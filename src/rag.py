"""
rag.py — RAG pipeline using ChromaDB + gemini-3.5-flash-lite
"""

import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DIR     = Path("data/chroma")

client        = genai.Client(api_key=GEMINI_API_KEY)
embeddings    = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection    = chroma_client.get_collection("physics_papers")

print(f"Loaded ChromaDB collection: {collection.count()} chunks")


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
    query_vec = embeddings.embed_query(query)
    results   = collection.query(
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
            "text":     doc,
            "title":    meta.get("title", "Unknown"),
            "authors":  meta.get("authors", "Unknown"),
            "url":      meta.get("url", ""),
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

    prompt = f"""Based on the following research paper excerpts, answer the question below.

CONTEXT:
{context}

QUESTION: {query}

Instructions:
- Answer only from the provided context
- Cite sources by their title
- If context is insufficient, say so clearly
- Be concise and technical
"""
    return prompt


def ask(query: str, top_k: int = 5) -> dict:
    """Full RAG pipeline: retrieve → build prompt → generate answer."""
    print(f"\nQuery: {query}")
    print("Retrieving relevant chunks...")

    chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    print(f"Retrieved {len(chunks)} chunks. Generating answer...")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    answer  = response.text
    sources = list({c["title"] for c in chunks})

    return {
        "query":   query,
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }


if __name__ == "__main__":
    result = ask("What machine learning methods are used for geomagnetic storm prediction?")
    print("\n" + "="*60)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s}")