
"""
rag.py — RAG pipeline using ChromaDB + Gemini
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DIR = Path("data/chroma")

# Resources are initialized only when first needed.
# Keeping these attributes allows tests to patch them.
client = None
embeddings = None
chroma_client = None
collection = None


def get_gemini_client():
    """Initialize Gemini only when an answer is requested."""
    global client

    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. "
                "Set it in your .env file or environment."
            )

        client = genai.Client(api_key=api_key)

    return client


def get_embeddings():
    """Load the embedding model only when retrieval is needed."""
    global embeddings

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    return embeddings


def get_collection():
    """Connect to the existing ChromaDB collection on demand."""
    global chroma_client, collection

    if collection is None:
        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )
        collection = chroma_client.get_collection(
            "physics_papers"
        )
        print(
            f"Loaded ChromaDB collection: "
            f"{collection.count()} chunks"
        )

    return collection


SYSTEM_PROMPT = """You are an expert space physics research assistant with deep knowledge
of geomagnetic storms, solar flares, space weather forecasting, and related ML techniques.

Answer questions strictly based on the provided research paper excerpts.
- Be precise and technical
- Always cite which paper a piece of information comes from
- If the context does not contain enough information, say so clearly
- Structure your answer with: Summary, Key Details, and Sources
"""


def retrieve(query: str, top_k: int = 5):
    """Retrieve top_k relevant chunks from ChromaDB."""
    query_vec = get_embeddings().embed_query(query)

    results = get_collection().query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "title": meta.get("title", "Unknown"),
            "authors": meta.get("authors", "Unknown"),
            "url": meta.get("url", ""),
            "distance": round(dist, 4),
        })

    return chunks


def build_prompt(query: str, chunks: list) -> str:
    """Build prompt with retrieved context."""
    context = ""

    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['title']} ---\n"
        context += f"Authors: {chunk['authors']}\n"
        context += f"URL: {chunk['url']}\n"
        context += f"{chunk['text']}\n"

    return f"""Based on the following research paper excerpts, answer the question below.

CONTEXT:
{context}

QUESTION: {query}

Instructions:
- Answer only from the provided context
- Cite sources by their title
- If context is insufficient, say so clearly
- Be concise and technical
"""


def ask(query: str, top_k: int = 5) -> dict:
    """Full RAG pipeline: retrieve → build prompt → generate answer."""
    print(f"\nQuery: {query}")
    print("Retrieving relevant chunks...")

    chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    print(f"Retrieved {len(chunks)} chunks. Generating answer...")

    response = get_gemini_client().models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )

    sources = list({c["title"] for c in chunks})

    return {
        "query": query,
        "answer": response.text,
        "sources": sources,
        "chunks": chunks,
    }


if __name__ == "__main__":
    result = ask(
        "What machine learning methods are used for geomagnetic storm prediction?"
    )

    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")

    for source in result["sources"]:
        print(f"  - {source}")
