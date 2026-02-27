# Legal RAG: Case Law Argument Validator (PoC)

## Project Overview
This project is a Proof of Concept (PoC) for an AI-powered legal validation tool utilizing Retrieval-Augmented Generation (RAG). It is designed to act as an advanced research assistant for legal professionals. 

Instead of relying on exact keyword matches, a lawyer can input a drafted legal argument or thesis. The system then uses semantic vector search to understand the *concept* of the argument, scans a localized database of historical case law, and retrieves specific, relevant precedents to support or challenge the argument.

## Core Benefits
* **Semantic Search:** Finds relevant case law based on the meaning of the argument, overcoming differences in judicial vocabulary.
* **Automated Precedent Discovery:** Reduces hours of manual research into minutes.
* **Counter-Argument Preparation:** Can be prompted to retrieve contradicting case law to help lawyers anticipate opposing arguments.

## Current Data Scope (Proof of Concept)
To ensure high performance, manageability, and accurate evaluation, the initial dataset is strictly scoped using data from the [Harvard Caselaw Access Project (CAP)](https://case.law/):
* **Jurisdiction:** State of New York
* **Court Level:** State Supreme Court and Appellate Courts only (Binding precedent)
* **Timeframe:** 2000 to 2020
* **Goal:** A manageable, high-quality dataset of modern legal language to test vector retrieval accuracy.

## System Architecture



1. **Data Ingestion:** Raw JSONL/Parquet files from Harvard CAP are loaded.
2. **Chunking & Embedding:** Legal texts are split into context-aware chunks (preserving headnotes and judicial opinions) and converted into vector embeddings.
3. **Vector Storage:** Embeddings are stored in a vector database (e.g., Pinecone, Qdrant, or Chroma).
4. **Retrieval (Hybrid Search):** User arguments are embedded and matched against the database using a combination of dense vector search (for concepts) and BM25 (for exact statutes).
5. **Generation:** An LLM synthesizes the retrieved chunks and validates the lawyer's argument with cited precedents.

## Critical Guardrails
* **Zero Hallucination Policy:** The LLM is strictly prompted to *only* generate responses based on the retrieved context. No outside knowledge is permitted for citation.
* **Good Law Validation:** Future iterations must include metadata tagging to ensure retrieved cases have not been overturned (Shepardizing).

---

## Next Steps & Recommendations (For Contributors)

### 1. Data Processing & Chunking
Standard text chunking will destroy legal context. We need to implement a chunking strategy that respects the structure of a legal opinion.
* **Action:** Write a Python script to parse the CAP JSON data, separate the "headnotes" (summaries) from the "opinions" (the judge's actual ruling), and chunk the text by paragraph rather than word count.

### 2. Database Setup
* **Action:** Select a Vector Database. For this PoC, a local instance of ChromaDB or an open-source tier of Qdrant will be sufficient and cost-effective.

### 3. Embedding Model Selection
* **Action:** Test legal-specific embedding models (e.g., `saul-base-uncased` or OpenAI's `text-embedding-3-large`) to see which accurately captures legal nuances best. 

### 4. Human-in-the-Loop Evaluation
* **Action:** Have our lawyer draft 5 to 10 standard arguments. Run them through the RAG pipeline and have the lawyer grade the retrieved precedents on a scale of 1-5 for relevance and accuracy.