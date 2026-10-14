"""Our own RAG, built step by step (no LlamaIndex / LangChain).

1. read_files  2. split_text  3. embed_texts  4. store_chunks  5. rag_search
"""
import time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from .. import settings as S

_client = None
_pc = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=S.NVIDIA_API_KEY, base_url=S.NVIDIA_BASE_URL)
    return _client


def _pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=S.PINECONE_API_KEY)
    return _pc


# ---- Step 1: Read ---------------------------------------------------------
def read_files() -> list[dict]:
    """Read every .py file in the code project."""
    files = []
    for path in sorted(S.PROJECT_PATH.rglob("*.py")):
        if any(p in path.parts for p in (".venv", "venv", "__pycache__", ".git")):
            continue
        files.append({
            "file": str(path.relative_to(S.PROJECT_PATH)),
            "text": path.read_text(encoding="utf-8", errors="replace"),
        })
    return files


# ---- Step 2: Split --------------------------------------------------------
def split_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Cut text into overlapping chunks, preferring to break at a newline."""
    size = size or S.CHUNK_SIZE
    overlap = S.CHUNK_OVERLAP if overlap is None else overlap
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + size // 2, end)
            if nl != -1:
                end = nl + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---- Step 3: Embed --------------------------------------------------------
def embed_texts(texts: list[str], kind: str = "passage") -> list[list[float]]:
    """Embed texts with the NVIDIA NIM model. kind is 'passage' or 'query'."""
    vectors = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        try:
            resp = _openai().embeddings.create(
                model=S.NVIDIA_EMBED_MODEL,
                input=batch,
                extra_body={"input_type": kind, "truncate": "END"},
            )
        except Exception:
            # Some models do not accept input_type; retry without it.
            resp = _openai().embeddings.create(model=S.NVIDIA_EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return vectors


# ---- Step 4: Store --------------------------------------------------------
def _get_index():
    pc = _pinecone()
    if S.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=S.PINECONE_INDEX_NAME,
            dimension=S.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=S.PINECONE_CLOUD, region=S.PINECONE_REGION),
        )
        while not pc.describe_index(S.PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(S.PINECONE_INDEX_NAME)


def store_chunks(chunks: list[dict]) -> int:
    """Embed chunks and save numbers + text in Pinecone."""
    index = _get_index()
    vectors = embed_texts([c["text"] for c in chunks], kind="passage")
    records = [
        {
            "id": c["id"],
            "values": v,
            "metadata": {"file": c["file"], "text": c["text"]},
        }
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i:i + 100])
    return len(records)


# ---- Step 5: Search (this is the agent's tool) ----------------------------
def rag_search(question: str) -> str:
    """Search the code project for chunks of code that match a question.

    Use this to find WHERE something is (a function, a feature, a setting).
    Returns the best matching code chunks with their file names.
    """
    try:
        qvec = embed_texts([question], kind="query")[0]
        res = _get_index().query(vector=qvec, top_k=S.TOP_K, include_metadata=True)
    except Exception as e:
        return f"SEARCH ERROR: {e}"
    matches = res.get("matches", [])
    if not matches:
        return "NO RESULTS: nothing matched the question."
    files = []
    for m in matches:
        f = m["metadata"]["file"]
        if f not in files:
            files.append(f)
    parts = ["SEARCH RESULTS (these are relevant). Most relevant files: " + ", ".join(files)]
    for m in matches:
        md = m["metadata"]
        parts.append(f"File: {md['file']}\nCode:\n{md['text'][:600]}")
    return "\n\n---\n\n".join(parts)
