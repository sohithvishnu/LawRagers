import os

# --- 🛑 STOP TENSORFLOW INTERFERENCE ---
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import torch
import ollama

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️", layout="wide")

# --- 2. SETUP BACKEND & EMBEDDINGS ---
@st.cache_resource
def get_db():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mps_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device=device
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_collection(
        name="ny_case_law", 
        embedding_function=mps_embedding_function
    )

collection = get_db()

# --- 3. LEGAL KNOWLEDGE SYSTEM (SYSTEM PROMPT) ---
LEGAL_SYSTEM_PROMPT = """
You are an expert New York appellate litigator. Your job is to validate the user's legal argument using ONLY the provided case law context.
You must structure your response using the IRAC method (Issue, Rule, Application, Conclusion).

STRICT GUARDRAILS:
1. Zero Hallucination: Do not cite any laws, cases, or concepts not explicitly found in the provided Context.
2. Citation: Always cite the case name when referencing a rule from the Context.
3. Candor to the Tribunal: If the provided cases contradict the user's argument, explicitly point out the flaw. If the context does not contain enough information to validate the argument, say so.
"""

# --- 4. CHAT MEMORY SETUP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. FRONTEND UI ---
st.title("⚖️ AI Legal Assistant (Powered by Llama 3)")
st.markdown("Draft your argument. The system will retrieve binding NY precedent and provide an IRAC-structured memo.")

# Display persistent chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_argument := st.chat_input("Draft your legal argument here..."):
    
    # 1. Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": user_argument})
    with st.chat_message("user"):
        st.markdown(user_argument)

    # 2. Retrieval Step
    with st.spinner("🔍 Searching NY Case Law Database..."):
        results = collection.query(
            query_texts=[user_argument],
            n_results=5
        )
    
    # 3. Format the Legal Context
    context_text = ""
    if results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            context_text += f"\n--- CASE: {meta['case_name']} ({meta['decision_date']}) ---\n{doc}\n"
    
    if not context_text:
        assistant_response = "I could not find any relevant precedents in the database to validate this argument."
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    else:
        # 4. LLM Generation Step
        with st.chat_message("assistant"):
            with st.spinner("✍️ Drafting IRAC Memo..."):
                
                # Combine prompt with retrieved context
                full_prompt = f"USER ARGUMENT:\n{user_argument}\n\nRETRIEVED PRECEDENT (CONTEXT):\n{context_text}"
                
                # Setup the message history for Ollama
                llm_messages = [{"role": "system", "content": LEGAL_SYSTEM_PROMPT}]
                # Feed in previous chat history to maintain memory
                for m in st.session_state.messages[:-1]: # exclude the current user prompt we just added
                    llm_messages.append({"role": m["role"], "content": m["content"]})
                # Add the current prompt with context
                llm_messages.append({"role": "user", "content": full_prompt})
                
                # Stream the response from Ollama
                response_stream = ollama.chat(
                    model='llama3',
                    messages=llm_messages,
                    stream=True,
                )
                
                # Streamlit magic to write text as it generates
                assistant_response = st.write_stream(chunk['message']['content'] for chunk in response_stream)
                
                # Show the citations used under the response
                with st.expander("📚 View Retrieved Cases Used for this Memo"):
                    st.text(context_text)

        # Save AI response to memory
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})