import os

# --- 🛑 STOP TENSORFLOW INTERFERENCE ---
# These MUST be before importing streamlit or chromadb
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import torch

# --- 1. MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Legal RAG PoC", page_icon="⚖️", layout="wide")


# --- 2. Setup MPS and Backend Connection ---
@st.cache_resource
def get_db():
    # Setup MPS (Apple Silicon GPU Acceleration)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Force the app to use the same accelerated model as the indexer
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

# --- 3. Frontend UI ---
st.title("⚖️ Semantic Case Law Validator")
st.markdown(
    "Enter your drafted legal argument below. The system will retrieve conceptually relevant binding precedents from the NY Court of Appeals.")

# Input area for the lawyer's argument
user_argument = st.text_area("Drafted Legal Argument / Thesis:", height=150,
                             placeholder="e.g., A landlord cannot be held liable for injuries caused by a tenant's dog unless the landlord had prior knowledge of the dog's vicious propensities...")

# Search execution
if st.button("Validate Argument via Precedent"):
    if user_argument:
        with st.spinner("Searching localized vector database..."):

            # Query ChromaDB - 'n_results' dictates how many chunks to retrieve
            results = collection.query(
                query_texts=[user_argument],
                n_results=5
            )

            st.subheader("🔍 Retrieved Precedents")

            # Display results beautifully
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    doc_text = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]

                    with st.expander(f"📌 {metadata['case_name']} ({metadata['decision_date']})"):
                        st.write(f"**Relevance Distance:** {results['distances'][0][i]:.4f}")
                        st.write(doc_text)
                        st.caption(f"Source: {metadata['source_file']}")
            else:
                st.info("No relevant precedents found in the current database.")
    else:
        st.warning("Please enter an argument to search.")