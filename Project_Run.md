# ⚖️ Legal RAG: Case Law Argument Validator (PoC)

## 📖 Overview
This project is a Proof of Concept (PoC) for an AI-powered legal validation tool utilizing Retrieval-Augmented Generation (RAG). 

Traditional legal research relies heavily on exact keyword matching (Boolean searches). This system allows a lawyer to input a drafted legal argument or thesis in plain English. The system then uses **semantic vector search** to understand the *concept* of the argument and retrieves specific, relevant binding precedents from the New York Court of Appeals to support or challenge the premise.

## ✨ Key Features
* **Semantic Search:** Finds relevant case law based on meaning, overcoming differences in judicial vocabulary (e.g., matching "precipitation" with "storm").
* **100% Local Vector Database:** Uses ChromaDB to store and query embeddings locally without relying on paid, external cloud databases.
* **Apple Silicon Accelerated:** Fully optimized to use Apple's Metal Performance Shaders (MPS), shifting the heavy embedding workload from the CPU to the Mac's built-in GPU.
* **Context-Aware Chunking:** Parses Harvard Caselaw Access Project (CAP) structured JSON files to separate non-binding headnotes from binding judicial opinions, chunking the text by paragraph to preserve legal context.



## 🛠️ Technology Stack
* **Frontend UI:** Streamlit
* **Vector Database:** ChromaDB
* **Embeddings:** `all-MiniLM-L6-v2` (via HuggingFace `sentence-transformers`)
* **Hardware Acceleration:** PyTorch (MPS/Apple Silicon)
* **Data Source:** Harvard Caselaw Access Project (New York Reports, 3d Series)

---

## 🚀 How to Install and Run

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment (like Anaconda or `venv`).

### 2. Install Dependencies
Run the following command in your terminal to install the required packages:
```bash
pip install streamlit chromadb torch sentence-transformers
```
*(Note: Our scripts automatically force the `transformers` library to bypass TensorFlow and use PyTorch to prevent local dependency conflicts).*

### 3. Data Preparation
This PoC requires raw JSON/JSONL data from the Harvard Caselaw Access Project.

1. Create a directory named `data/extracted_json` in the root of this project.
2. Download the New York Reports (`ny3d`) archive from Harvard CAP.
3. Extract the `.json` files representing the individual cases into the `data/extracted_json` folder.

### 4. Build the Vector Database (Backend)
Before you can search, you must embed the case law into the local vector database. Run the indexing script:

```bash
python build_index.py
```

**What this does:**
* Reads the raw JSON files.
* Targets the `casebody -> opinions` structure.
* Chunks the text by paragraph.
* Converts the text into vector embeddings using your Mac's GPU (MPS).
* Saves the vectors locally into a folder called `chroma_db`.

### 5. Launch the User Interface (Frontend)
Once the database is built, start the Streamlit web application:

```bash
streamlit run app.py
```

This will open a web browser at http://localhost:8501.

### 🧪 Example Test Queries
Try pasting these arguments into the UI to see the semantic search in action:

#### Premises Liability (Storm in Progress):

"A property owner is not liable for injuries caused by a slip and fall on snow or ice if the accident occurred while the winter storm was still ongoing."

#### Strict Liability (Dog Bites):

"To hold a landlord strictly liable for an injury caused by a tenant's domestic animal, the plaintiff must prove that the animal had vicious propensities and the landlord knew about them."


