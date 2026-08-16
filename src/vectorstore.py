import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from embedding import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.persist_dir = persist_dir

        # Create folder if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model

        # Load embedding model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):

        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} raw documents..."
        )

        # Create embedding pipeline
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Split documents into chunks
        chunks = emb_pipe.chunk_documents(documents)

        # Generate embeddings
        embeddings = emb_pipe.embed_chunks(chunks)

        # Store chunk text as metadata
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "unknown")
            }
            for chunk in chunks
        ]

        # Add embeddings to FAISS
        self.add_embeddings(
            np.array(embeddings).astype("float32"),
            metadatas
        )

        # Save vector store
        self.save()

        print(
            f"[INFO] Vector store built and saved to "
            f"{self.persist_dir}"
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None
    ):

        # Get embedding dimension
        dim = embeddings.shape[1]

        # Create FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        # Add embeddings
        self.index.add(embeddings)

        # Add metadata
        if metadatas:
            self.metadata.extend(metadatas)

        print(
            f"[INFO] Added {embeddings.shape[0]} "
            f"vectors to Faiss index."
        )

    def save(self):

        faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        # Save FAISS index
        faiss.write_index(
            self.index,
            faiss_path
        )

        # Save metadata
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(
            f"[INFO] Saved Faiss index and metadata "
            f"to {self.persist_dir}"
        )

    def load(self):

        faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        # Load FAISS index
        self.index = faiss.read_index(
            faiss_path
        )

        # Load metadata
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(
            f"[INFO] Loaded Faiss index and metadata "
            f"from {self.persist_dir}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):

        # Search FAISS
        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx, distance in zip(
            indices[0],
            distances[0]
        ):

            # Ignore invalid index
            if idx < 0:
                continue

            meta = (
                self.metadata[idx]
                if idx < len(self.metadata)
                else None
            )

            results.append(
                {
                    "index": int(idx),
                    "distance": float(distance),
                    "metadata": meta
                }
            )

        return results

    def query(
        self,
        query_text: str,
        top_k: int = 5
    ):

        print(
            f"[INFO] Querying vector store for: "
            f"'{query_text}'"
        )

        # Convert query into embedding
        query_embedding = self.model.encode(
            [query_text]
        ).astype("float32")

        # Search vector database
        return self.search(
            query_embedding,
            top_k=top_k
        )


# --- Example Usage ---

if __name__ == "__main__":

    from data_loader import load_all_documents

    # Load documents
    docs = load_all_documents("data")

    # Create vector store
    store = FaissVectorStore(
        "faiss_store"
    )

    # Build vector store
    store.build_from_documents(docs)

    # Load saved vector store
    store.load()

    # Test search
    results = store.query(
        "What is attention mechanism?",
        top_k=3
    )

    print("Search Results:")
    print(results)