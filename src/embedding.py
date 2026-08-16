from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from data_loader import load_all_documents


class EmbeddingPipeline:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:

        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Split documents into chunks
        chunks = splitter.split_documents(documents)

        print(
            f"[INFO] Split {len(documents)} documents "
            f"into {len(chunks)} chunks."
        )

        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:

        # Extract text from each chunk
        texts = [chunk.page_content for chunk in chunks]

        print(
            f"[INFO] Generating embeddings for {len(texts)} chunks..."
        )

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        return embeddings


# --- Example Usage ---

if __name__ == "__main__":

    # Load all documents from data folder
    docs = load_all_documents("data")

    # Create embedding pipeline
    emb_pipe = EmbeddingPipeline()

    # Split documents into smaller chunks
    chunks = emb_pipe.chunk_documents(docs)

    # Generate embeddings for chunks
    embeddings = emb_pipe.embed_chunks(chunks)

    # Print first embedding as an example
    print(
        "[INFO] Example embedding:",
        embeddings[0] if len(embeddings) > 0 else None
    )