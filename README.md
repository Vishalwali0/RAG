# RAGTry — Retrieval-Augmented Generation Example

A compact Retrieval-Augmented Generation (RAG) example that demonstrates document loading, chunking, embedding with SentenceTransformers, FAISS-based semantic retrieval, and generation via Google's Gemini model (accessed through `langchain_google_genai`).

---

## Overview

RAGTry is a small Python project that implements a practical RAG pipeline: it loads documents from a local `data/` folder, splits them into chunks, converts chunks to embeddings, stores vectors in a FAISS index, and uses semantic retrieval to assemble context for a generative LLM (Gemini) to produce grounded summaries.

Why this is useful: language models can hallucinate or lack domain-specific knowledge. RAG augments model input with retrieved, up-to-date, or private documents to produce more accurate, grounded answers.

Main technologies used (from the codebase): SentenceTransformers, FAISS, LangChain community loaders, and an adapter for Google's generative API (`langchain_google_genai`).

---

## Research Motivation

### Problem

Large language models are powerful but limited by their pretraining cutoff and can produce plausible-sounding but incorrect answers (hallucinations). They also do not have direct access to private or domain-specific documents unless those documents are supplied at inference time.

### Proposed Approach

This repository demonstrates a standard RAG approach:

Documents
    ↓
Document Processing (load + chunk)
    ↓
Embeddings (SentenceTransformers: `all-MiniLM-L6-v2`)
    ↓
FAISS Vector Store (persisted index + metadata)
    ↓
Semantic Retrieval (nearest neighbors)
    ↓
Context Construction (concatenate top-K chunks)
    ↓
Gemini (LLM generation via `langchain_google_genai`)

The LLM receives retrieved context so generated answers are grounded in the stored documents rather than solely in the model's pretraining.

---

## System Architecture

```mermaid
flowchart TD
  A[User Query] --> B[RAGSearch]
  B --> C[FaissVectorStore]
  C --> D[FAISS Index (faiss.index)]
  C --> E[Metadata (metadata.pkl)]
  B --> F[ChatGoogleGenerativeAI (Gemini)]
  D --> G[Top-K chunks]
  G --> F
  F --> H[Generated Summary]

  classDef storage fill:#f3f4f6,stroke:#333,stroke-width:1px;
  class D,E storage;
```

Components (code locations):

- Vector store & persistence: [src/vectorstore.py](src/vectorstore.py#L1)
- Document loading: [src/data_loader.py](src/data_loader.py#L1)
- Chunking & embeddings: [src/embedding.py](src/embedding.py#L1)
- RAG orchestration + LLM call: [src/search.py](src/search.py#L1)

---

## RAG Pipeline (Detailed)

### Step 1 — Document Loading

Documents are loaded from the `data/` directory using `load_all_documents(data_dir)` implemented in [src/data_loader.py](src/data_loader.py#L1). Supported file types (per code): PDF, TXT, CSV, XLSX, DOCX, JSON. The loader uses community LangChain loaders (`PyPDFLoader`, `TextLoader`, `CSVLoader`, `UnstructuredExcelLoader`, `Docx2txtLoader`, `JSONLoader`).

### Step 2 — Document Processing (Chunking)

Chunking is performed by `EmbeddingPipeline.chunk_documents()` in [src/embedding.py](src/embedding.py#L1). The code uses `RecursiveCharacterTextSplitter` with:

- `chunk_size = 1000`
- `chunk_overlap = 200`

This splits documents into overlapping chunks to preserve context across boundaries.

### Step 3 — Embeddings

Embeddings are produced with `SentenceTransformer` using the `all-MiniLM-L6-v2` model (this model name is configured in the code). Embeddings convert chunk text into fixed-length numeric vectors suitable for similarity search.

Why embeddings: semantic representations allow retrieval of text that is semantically similar to a query even when lexical overlap is small.

### Step 4 — Vector Database (FAISS)

The project uses FAISS for the vector index. Implementation notes (see [src/vectorstore.py](src/vectorstore.py#L1)):

- The index type: `faiss.IndexFlatL2` (an L2-distance flat index).
- Persistent artifacts: `faiss_store/faiss.index` and `faiss_store/metadata.pkl` (the latter stores chunk metadata and text).

FAISS stores dense vectors and supports fast nearest-neighbor search; metadata is kept in a parallel Python list and persisted with `pickle`.

### Step 5 — Retrieval

Queries are embedded using the same SentenceTransformer model and passed to FAISS via `FaissVectorStore.query()` which performs a nearest-neighbor search (`IndexFlatL2`) and returns the top-K results (default `top_k=5`, configurable when calling `search_and_summarize`).

If you query e.g. `"NoSQL"`, the pipeline:

1. Encodes `"NoSQL"` to a vector.
2. Performs L2 nearest-neighbor search in the FAISS index.
3. Returns the top-K chunk texts (from metadata) and distances.

### Step 6 — Context Construction

In [src/search.py](src/search.py#L1), retrieved chunk texts are concatenated with double newlines into a single `context` string. If no relevant documents are found, the code returns a short message.

### Step 7 — Generation (Gemini)

The repository initializes `ChatGoogleGenerativeAI` from `langchain_google_genai` (see [src/search.py](src/search.py#L1)). The default model string in the code is `gemini-3.7-flash`. The code sends a prompt that includes the original query and the retrieved context, and then calls `self.llm.invoke(prompt)` to obtain a generated summary.

The LLM therefore receives the retrieved context as part of the prompt; the generation step is augmentation + generation, not pure reliance on the LLM's pretraining.

---

## 6. Technical Methodology

### 6.1 Semantic Representation

Text chunks are represented as dense vectors via SentenceTransformers (`all-MiniLM-L6-v2`). These vectors capture semantic relationships so that nearby vectors correspond to semantically related text.

### 6.2 Similarity Search

FAISS performs nearest-neighbor lookup in vector space. The code constructs an `IndexFlatL2` index, meaning similarity is determined by L2 (Euclidean) distance between vectors. Lower distances indicate higher similarity.

### 6.3 Retrieval-Augmented Generation

Retrieval, augmentation, and generation are separated:

- Retrieval: nearest-neighbor search over vector store (FAISS).
- Augmentation: concatenation of retrieved chunk texts to form prompt context.
- Generation: passing the augmented prompt to Gemini (via `langchain_google_genai`) to produce a final answer.

---

## 7. Implementation Details

Key facts extracted from the repository:

- **Python**: project `pyproject.toml` specifies `requires-python = ">=3.14"` ([pyproject.toml](pyproject.toml#L1)).
- **Main libraries**: see `requirements.txt` — `langchain`, `langchain-core`, `langchain-community`, `pypdf`, `pymupdf`, `sentence-transformers`, `faiss-cpu`, `chromadb` ([requirements.txt](requirements.txt#L1)).
- **Embedding model**: `all-MiniLM-L6-v2` (used via `SentenceTransformer`).
- **Vector store**: FAISS (`IndexFlatL2`) and a pickled metadata list persisted under `faiss_store/`.
- **LLM**: Gemini (model name used: `gemini-3.7-flash`), invoked via `ChatGoogleGenerativeAI` in `langchain_google_genai`.
- **Environment**: `dotenv` is used to read `GEMINI_API_KEY` in [src/search.py](src/search.py#L1).
- **Persistence**: `faiss.index` and `metadata.pkl` in the `faiss_store` directory.

Technology summary:

| Component     | Technology / Code Reference                         | Purpose |
|--------------:|:---------------------------------------------------|:--------|
| Language      | Python (pyproject requires >=3.14)                  | Application logic |
| Document loaders | LangChain community loaders (`data_loader.py`)    | PDF, TXT, CSV, DOCX, XLSX, JSON ingestion |
| Chunking      | `RecursiveCharacterTextSplitter` (`embedding.py`)   | Text chunking |
| Embeddings    | `sentence-transformers / all-MiniLM-L6-v2`         | Semantic vectors |
| Vector Store  | FAISS (`IndexFlatL2`) (`vectorstore.py`)           | Nearest-neighbor search, persistence |
| LLM Adapter   | `langchain_google_genai` (`search.py`)             | Gemini API calls |

---

## 8. Project Structure

```
ragtry/
├── data/                  # place your PDFs, TXT, CSV, JSON, DOCX, XLSX here
├── faiss_store/           # generated: faiss.index + metadata.pkl
├── src/
│   ├── app.py             # example runner
│   ├── data_loader.py     # document loaders
│   ├── embedding.py       # chunking + embedding pipeline
│   ├── vectorstore.py     # FAISS vector store wrapper
│   ├── search.py          # RAG orchestration and Gemini calls
│   └── __init__.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

File responsibilities:

- [src/data_loader.py](src/data_loader.py#L1): loads documents from `data/` using LangChain community loaders.
- [src/embedding.py](src/embedding.py#L1): splits documents and generates embeddings.
- [src/vectorstore.py](src/vectorstore.py#L1): manages FAISS index and metadata persistence.
- [src/search.py](src/search.py#L1): performs retrieval, constructs prompt, and invokes Gemini.
- [src/app.py](src/app.py#L1): example script demonstrating a simple query flow.

---

## 9. Installation

Minimal steps to get started (cross-platform, Windows examples shown where convenient):

1. Clone the repository

```bash
git clone <your-repo-url>
cd ragtry
```

2. Create a virtual environment (Windows PowerShell example)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

4. (Optional) If you prefer `pyproject.toml` installation:

```powershell
pip install -e .
```

5. Prepare data: put PDFs, TXT, CSV, DOCX, XLSX, or JSON files under the `data/` folder.

6. Configure `GEMINI_API_KEY` in an `.env` file (see next section) and do not commit it to version control.

7. Run the example

```powershell
python src/app.py
```

---

## 10. Environment Variables

The project reads `GEMINI_API_KEY` using `python-dotenv` in [src/search.py](src/search.py#L1). Create a local `.env` file containing:

```
GEMINI_API_KEY=your_api_key_here
```

Never commit `.env` with secrets to your repository. Use secret managers for production.

---

## 11. Usage

Run the example in [src/app.py](src/app.py#L1) or use `RAGSearch` directly from [src/search.py](src/search.py#L1):

```python
from search import RAGSearch

rag = RAGSearch()
print(rag.search_and_summarize("NoSQL", top_k=3))
```

When executed, the system will:

1. Ensure a FAISS index exists in `faiss_store/` (it will build one from `data/` if missing).
2. Embed the query, perform a FAISS top-K search, collect chunk texts, and concatenate them.
3. Send a prompt containing the query and retrieved context to Gemini via `ChatGoogleGenerativeAI` and return the generated text.

Note: the exact response format depends on the `langchain_google_genai` adapter; the code attempts to read `response.content`.

---

## 12. Example Walkthrough

Query: `NoSQL`

Flow:

Query -> Query embedding -> FAISS top-K retrieval -> Retrieved chunks (texts) -> Context concatenation -> Gemini prompt -> Generated summary

This repository provides code to trace each of these steps; no quantitative evaluation harness is included.

---

## 13. Research / Experimental Perspective

**Objective**: Demonstrate an end-to-end RAG pipeline combining local document ingestion, sentence-transformer embeddings, FAISS nearest-neighbor retrieval, and a generative model (Gemini) to produce grounded answers.

**Method**: Off-the-shelf chunking, SentenceTransformers embeddings (`all-MiniLM-L6-v2`), FAISS `IndexFlatL2` nearest-neighbor retrieval, followed by prompt-based generation using `langchain_google_genai`.

**Configuration (from code)**:

- Embedding model: `all-MiniLM-L6-v2`
- LLM model identifier: `gemini-3.7-flash` (configurable in `RAGSearch`)
- Vector store: FAISS, `IndexFlatL2`
- `top_k`: default 5 in `FaissVectorStore.query()` and passed to `search_and_summarize()` (configurable)

**Observations**: Quantitative retrieval or answer-quality evaluation is not implemented in this repository.

---

## 14. Limitations

- Retrieval uses a fixed `IndexFlatL2` (no IVF, PQ, or HNSW approximation); may not scale to very large corpora.
- Chunking parameters are fixed in the code (`chunk_size=1000`, `chunk_overlap=200`).
- No reranking or cross-encoder re-ranking step is provided.
- No automated evaluation (precision/recall, human evaluation) is included.
- The code depends on the Gemini API (`langchain_google_genai`) and requires an API key and quota.
- Metadata is a simple pickled Python list — not a queryable metadata store.

---

## 15. Future Improvements (Suggested)

1. Add configurable chunking strategies and unit tests for chunking behavior.
2. Add a reranking model (cross-encoder) to improve result ordering.
3. Support approximate nearest neighbor indices (HNSW/IVF) for larger datasets.
4. Add structured metadata filtering (by source, date, or custom tags).
5. Add evaluation scripts for retrieval and answer quality.
6. Add streaming LLM responses and conversational memory.

---

## 16. Learning Outcomes

By studying this repo you can learn:

- Document ingestion with LangChain loaders
- Chunking strategies with text splitters
- Generating embeddings with SentenceTransformers
- Building and persisting FAISS indexes
- Performing semantic retrieval and assembling context
- Invoking a generative LLM with retrieved context

---

## 17. Security Notes

- Keep `GEMINI_API_KEY` and other secrets out of version control.
- `.env` files should be listed in `.gitignore` (verify locally).
- Consider using secret managers for production-deployed systems.

---

## 18. Dependencies (purpose)

- `langchain`, `langchain-core`, `langchain-community` — document loaders and helpers.
- `pypdf`, `pymupdf` — PDF loading backends.
- `sentence-transformers` — text embedding (SentenceTransformers).
- `faiss-cpu` — FAISS index for nearest-neighbor retrieval.
- `chromadb` — included in `requirements.txt` but not used by the current code (present but unused).

---

## 19. Conclusion

RAGTry is an instructive, code-first example of a Retrieval-Augmented Generation pipeline. It wires together document loaders, chunking, SentenceTransformers embeddings, FAISS-based retrieval, and an adapter to a generative LLM (Gemini). The implementation is intentionally simple and is a good starting point for experiments in retrieval strategies, reranking, and grounding generative models with local content.

---

If you want, I can now:

- Run a quick repository scan to confirm any missing imports or dependency gaps.
- Add a small demo script that builds the index and runs a user-specified query.
