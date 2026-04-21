# ⚖️ Legal Scribe: AI-Powered Analysis Workspace

## 📖 Overview
**Legal Scribe** is a professional-grade legal research and analysis workspace. It leverages **Retrieval-Augmented Generation (RAG)** to connect binding New York precedent with a lawyer's private matter files.

Unlike traditional keyword-based research, Legal Scribe provides a **persistent, conversational, and visual environment**. It builds a cumulative "Semantic Network" of a case, allowing attorneys to visualize how precedents interconnect with their arguments while maintaining isolated, persistent sessions in a local SQLite database.

## ✨ Key Features
* **Isolated Matter Management:** A persistent "Lobby" for managing distinct legal matters. Each matter has its own isolated vector space to prevent data leakage between cases.
* **Dual-RAG Architecture:** Simultaneously queries the New York Court of Appeals (binding law) and private user uploads (PDF/Text) to provide contextually grounded legal memos.
* **Cumulative Knowledge Graph:** An interactive, Obsidian-inspired network that identifies "Anchor Cases"—precedents that are repeatedly relevant across a chat history.
* **Graph Checkpoints (Time Travel):** Every AI response saves a snapshot of the graph. Users can "rewind" the visual network to any point in the chat history.
* **Streaming IRAC Memo:** Character-by-character streaming of legal memos in Markdown format, following the industry-standard IRAC (Issue, Rule, Application, Conclusion) structure.
* **Persistent Memory:** Complete chat history and session metadata are stored in a local SQLite database for seamless continuity.

## 🛠️ Technology Stack
* **Frontend:** Expo (React Native for Web)
* **Backend API:** FastAPI (Python)
* **AI Engine:** Ollama (Llama 3)
* **Vector Database:** ChromaDB (Semantic Search)
* **Relational Database:** SQLite (Session & Chat History)
* **Hardware Acceleration:** PyTorch (MPS / Apple Silicon)
* **Document Parsing:** PyPDF2

---

## 🚀 Installation & Setup

### 1. Prerequisites
* **Python 3.9+**
* **Node.js & npm** (for Expo)
* **Ollama** installed and running on your Mac.

### 2. Install Backend Dependencies
Run the following in your project root:
```bash
pip install fastapi uvicorn chromadb torch sentence-transformers ollama python-multipart PyPDF2
```
### 3. Install Frontend Dependencies

Navigate to your legal-dashboard folder:

```bash
npx expo install react-native-svg react-native-markdown-display @expo/vector-icons expo-document-picker
```

### 🏗️ How to Run

Step 1: Start Ollama

Ensure Llama 3 is available and the server is listening:

``` bash
ollama run llama3
```
Step 2: Start the Python API (Backend)

In your project root:

```bash
python -m uvicorn api:app --reload
```
The backend will initialize the legal_sessions.db (SQLite) and the chroma_db (Vector) folders.

Step 3: Start the Expo Dashboard (Frontend)

In the legal-dashboard folder:

```bash
npx expo start -w
```
This will launch the Legal Scribe Lobby in your web browser.

### 📁 Workflow: Managing a Matter
The Lobby: Upon launch, select "Create New Matter" or pick an existing session from the persistent sidebar.

Configuration: Define your matter name, select your databases (e.g., NY Precedent), and upload your initial case files (PDF/Text).

The Workspace: * Pane 1 (Chat): Ask follow-up questions or draft arguments. Watch the AI stream its memo character-by-character.

Pane 2 (Graph): Visualize the semantic connections. Use "View Graph Checkpoint" on old messages to see the network's history.

Pane 3 (Viewer): Select nodes in the graph to read the full text of the court opinion or uploaded source.

### 🧪 Suggested Testing Scenario

Create Matter: "Smith Premises Liability"

Upload: A PDF of a witness deposition.

Query: "Based on the uploaded deposition and NY Law, does the 'Storm in Progress' doctrine apply to the slip and fall that occurred at 2:00 PM?"

Follow-up: "How does this change if the plaintiff testifies it had stopped snowing an hour prior?"