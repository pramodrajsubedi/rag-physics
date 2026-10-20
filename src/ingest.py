"""
ingest.py: --- Fetch arXiv papers, extract text, embed with Gemini, store in Chroma DB
Run from project root: python src/ingest.py
"""

# import libraries
import os
import arxiv
import pymupdf
import chromadb
from pathlib import Path
from dotenv import load_dotenv
import urllib.request
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # Load environment variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get Gemini API key from environment variables
DATA_DIR = Path("data/papers")
CHROMA_DIR = Path("data/chroma")
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create data directory if it doesn't exist
CHROMA_DIR.mkdir(parents=True, exist_ok=True)  # Create Chroma directory if it doesn't exist

SEARCH_QUERIES = [
    "geomagnetic solar storm prediction",
    "space weather machine learning forecasting",
    "solar flare classification deep learning",
    "magnetosphere plasma physics neural network"
]

MAX_PAPERS = 10

def fetch_papers():
    """Fetch papers from arXiv based on search queries with robust backoff."""
    print("Fetching papers from arXiv...")
    client = arxiv.Client()
    papers = []
    seen = set()
 
    for query in SEARCH_QUERIES:
        print(f"   Querying: {query}")
        search = arxiv.Search(
            query=query,
            max_results=MAX_PAPERS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        
        # Retry loop for 429 errors from arXiv
        success = False
        retries = 3
        wait_time = 10  # Start with a 10-second penalty box wait if throttled
        
        for attempt in range(retries):
            try:
                for result in client.results(search):
                    if result.entry_id not in seen:
                        seen.add(result.entry_id)
                        papers.append(result)
                success = True
                break
            except Exception as e:
                print(f"   WARNING (Attempt {attempt+1}/{retries}) on query '{query}': {e}")
                print(f"   Waiting {wait_time}s for arXiv rate limit to clear...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
                
        if not success:
            print(f"   ERROR: Failed to fetch query '{query}' after {retries} attempts.")
 
        
 
    print(f"Found {len(papers)} unique papers.")
    return papers
 
 
def download_and_extract(papers):
    """Download PDFs and extract text."""
    docs = []
    for i, paper in enumerate(papers):
        pdf_path = DATA_DIR / f"{paper.entry_id.split('/')[-1]}.pdf"
 
        try:
            if not pdf_path.exists():
                print(f"   Downloading [{i+1}/{len(papers)}]: {paper.title[:60]}")
                urllib.request.urlretrieve(paper.pdf_url, str(pdf_path))
                time.sleep(2)  # Polite delay between downloads
            else:
                print(f"   Already exists: {paper.title[:60]}")
 
            doc = pymupdf.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
 
            if len(text.strip()) > 100:
                docs.append({
                    "text": text,
                    "title": paper.title,
                    "authors": ", ".join(a.name for a in paper.authors[:3]),
                    "url": paper.entry_id,
                    "id": paper.entry_id.split("/")[-1],
                })
 
        except Exception as e:
            print(f"   ERROR on {paper.title[:40]}: {e}")
 
    print(f"Extracted text from {len(docs)} papers.")
    return docs
 
 
def chunk_and_embed(docs):
    """Chunk text and embed into ChromaDB."""
    print("\nChunking and embedding...")
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
 
    # Updated to Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
 
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection("physics_papers")
 
    # Resume — skip already embedded chunks
    existing_ids = set(collection.get()["ids"])
    print(f"Already embedded: {len(existing_ids)} chunks")
 
    batch_texts = []
    batch_ids = []
    batch_meta = []
 
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        print(f"   {doc['title'][:60]} → {len(chunks)} chunks")
 
        for j, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{j}"
            if chunk_id not in existing_ids:
                batch_texts.append(chunk)
                batch_ids.append(chunk_id)
                batch_meta.append({
                    "title": doc["title"],
                    "authors": doc["authors"],
                    "url": doc["url"],
                    "chunk_index": j,
                })
 
    print(f"New chunks to embed: {len(batch_texts)}")
 
    if len(batch_texts) == 0:
        print("Nothing new to embed.")
        print(f"ChromaDB collection size: {collection.count()}")
        return
 
    batch_size = 10
    for start in range(0, len(batch_texts), batch_size):
        end = min(start + batch_size, len(batch_texts))
        batch = batch_texts[start:end]
 
        vecs = embeddings.embed_documents(batch)
        collection.add(
            ids=batch_ids[start:end],
            embeddings=vecs,
            documents=batch_texts[start:end],
            metadatas=batch_meta[start:end],
        )
        print(f"   Embedded {end}/{len(batch_texts)} chunks")
        time.sleep(6)  # Stay safely under rate limits
 
    print(f"\nChromaDB collection size: {collection.count()}")
    print(f"Saved to: {CHROMA_DIR}")
 
 
if __name__ == "__main__":
    papers = fetch_papers()
    docs = download_and_extract(papers)
    chunk_and_embed(docs)
    print("\nIngestion complete.")