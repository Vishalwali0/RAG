from data_loader import load_all_documents
from vectorstore import FaissVectorStore
from search import RAGSearch


# Example usage
if __name__ == "__main__":

    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")

    # Vector store is already created, so we only load it
    store.load()

    rag_search = RAGSearch()

    query = "What is a data-intensive application?"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("Summary:", summary)