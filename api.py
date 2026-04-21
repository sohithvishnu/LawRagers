import os

# --- 🛑 STOP TENSORFLOW INTERFERENCE ---
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
import torch
import ollama
import PyPDF2
import io
import sqlite3
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. SETUP CHROMADB & EMBEDDINGS ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
mps_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2", device=device
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

case_collection = chroma_client.get_collection(name="ny_case_law", embedding_function=mps_ef)
user_collection = chroma_client.get_or_create_collection(
    name="user_workspace", 
    embedding_function=mps_ef
)

# --- 2. SQLITE DATABASE SETUP (WITH CHECKPOINTS) ---
db_conn = sqlite3.connect('legal_sessions.db', check_same_thread=False)
cursor = db_conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        databases TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        role TEXT,
        content TEXT,
        graph_state TEXT DEFAULT '[]',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
''')

# Safely upgrade existing database if graph_state column is missing
try:
    cursor.execute("ALTER TABLE messages ADD COLUMN graph_state TEXT DEFAULT '[]'")
except sqlite3.OperationalError:
    pass # Column already exists

db_conn.commit()


# --- DATA MODELS ---
class SessionCreate(BaseModel):
    name: str
    description: str
    databases: str

class LegalArgument(BaseModel):
    session_id: str
    argument: str

class GenerateRequest(BaseModel):
    session_id: str
    argument: str
    context_text: str

class ChatMessage(BaseModel):
    session_id: str
    role: str
    content: str
    graph_state: Optional[str] = "[]"


# --- ENDPOINTS: SESSIONS & CHAT MEMORY ---
@app.post("/sessions")
def create_session(req: SessionCreate):
    session_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO sessions (id, name, description, databases) VALUES (?, ?, ?, ?)",
        (session_id, req.name, req.description, req.databases)
    )
    db_conn.commit()
    return {"id": session_id, "name": req.name}

@app.get("/sessions")
def get_sessions():
    cursor.execute("SELECT id, name, description, databases, created_at FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    sessions = [{"id": r[0], "name": r[1], "description": r[2], "databases": r[3], "created_at": r[4]} for r in rows]
    return {"sessions": sessions}

@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    cursor.execute("SELECT id, role, content, graph_state FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
    rows = cursor.fetchall()
    messages = [{"id": r[0], "role": r[1], "text": r[2], "graph_state": r[3]} for r in rows]
    return {"messages": messages}

@app.post("/messages")
def save_message(req: ChatMessage):
    msg_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO messages (id, session_id, role, content, graph_state) VALUES (?, ?, ?, ?, ?)",
        (msg_id, req.session_id, req.role, req.content, req.graph_state)
    )
    db_conn.commit()
    return {"status": "success", "id": msg_id}


# --- ENDPOINTS: ISOLATED FILE UPLOAD ---
@app.post("/upload")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    text = ""
    
    if file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = contents.decode('utf-8')
        
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    
    if paragraphs:
        ids = [f"{session_id}_{file.filename}_{i}" for i in range(len(paragraphs))]
        metadatas = [{"source": file.filename, "session_id": session_id} for _ in paragraphs]
        user_collection.add(documents=paragraphs, metadatas=metadatas, ids=ids)
        
    return {"status": "success", "filename": file.filename, "chunks_indexed": len(paragraphs)}


# --- ENDPOINTS: DUAL-RAG SEARCH & GENERATION WITH MEMORY ---
@app.post("/search")
def search_cases(req: LegalArgument):
    cases = []
    context_text = "--- BINDING PRECEDENT (NY CASE LAW) ---\n"
    
    case_results = case_collection.query(query_texts=[req.argument], n_results=5)
    if case_results['documents'] and len(case_results['documents'][0]) > 0:
        for i in range(len(case_results['documents'][0])):
            doc = case_results['documents'][0][i]
            meta = case_results['metadatas'][0][i]
            cases.append({
                "id": meta.get('case_name', f"Case {i}"),
                "date": meta.get('decision_date', ''),
                "text": doc,
                "distance": case_results['distances'][0][i]
            })
            context_text += f"\n[Case: {meta.get('case_name')}]\n{doc}\n"

    context_text += "\n--- USER UPLOADED DOCUMENTS ---\n"
    try:
        if user_collection.count() > 0:
            user_results = user_collection.query(
                query_texts=[req.argument], 
                n_results=3,
                where={"session_id": req.session_id} 
            )
            if user_results['documents'] and len(user_results['documents'][0]) > 0:
                for i in range(len(user_results['documents'][0])):
                    doc = user_results['documents'][0][i]
                    meta = user_results['metadatas'][0][i]
                    context_text += f"\n[Source: {meta.get('source')}]\n{doc}\n"
    except Exception:
        pass 

    return {"cases": cases, "context_text": context_text}

@app.post("/generate")
def generate_memo(req: GenerateRequest):
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC", (req.session_id,))
    history = cursor.fetchall()

    system_prompt = "You are a legal AI workspace. Analyze the user's argument or question based on the provided USER UPLOADED DOCUMENTS and BINDING PRECEDENT. Synthesize the facts of the upload with the rules of the precedent. Use IRAC format."
    messages = [{"role": "system", "content": system_prompt}]
    
    for row in history:
        messages.append({"role": row[0], "content": row[1]})
        
    current_prompt = f"USER QUERY:\n{req.argument}\n\n{req.context_text}"
    messages.append({"role": "user", "content": current_prompt})
    
    def stream_generator():
        for chunk in ollama.chat(model='llama3', messages=messages, stream=True):
            yield chunk['message']['content']
            
    return StreamingResponse(stream_generator(), media_type="text/plain")