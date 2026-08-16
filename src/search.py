import os
from dotenv import load_dotenv
from vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemini-3.7-flash"
    ):

        # Create vector store
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # Load or build vector store
        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):

            from data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()

        # Gemini API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=gemini_api_key
        )

        print(f"[INFO] Gemini LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Search vector database
        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # Get text from retrieved documents
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        # Combine retrieved text
        context = "\n\n".join(texts)

        # If nothing was found
        if not context:
            return "No relevant documents found."

        # Create Gemini prompt
        prompt = f"""
Summarize the following context for the query:

Query:
{query}

Context:
{context}

Summary:
"""

        # Send prompt to Gemini
        response = self.llm.invoke(prompt)

        # Extract Gemini answer
        if isinstance(response.content, list):
            return response.content[0]["text"]
        else:
            return response.content


# --- Example Usage ---

if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "What is a data-intensive application?"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("Summary:", summary)