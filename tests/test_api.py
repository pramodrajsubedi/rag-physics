import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

MOCK_RESULT = {
    "query":   "test question",
    "answer":  "test answer",
    "sources": ["Paper A", "Paper B"],
    "chunks":  [],
}

with patch("rag.collection", MagicMock(count=MagicMock(return_value=100))), \
     patch("rag.embeddings", MagicMock()), \
     patch("rag.client", MagicMock()):
    from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_root():
    r = client.get("/")
    assert r.status_code == 200