"""Central settings. Reads .env (kept inside code_helper/, never committed)."""
import os
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

# LiteLLM reads the NVIDIA key under its own name (same key).
if os.getenv("NVIDIA_API_KEY"):
    os.environ.setdefault("NVIDIA_NIM_API_KEY", os.environ["NVIDIA_API_KEY"])

# Folder of the Python project the helper studies (absolute path).
PROJECT_PATH = Path(os.getenv("PROJECT_PATH", "")).expanduser().resolve()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "code-helper")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_DIM = 2048          # must match nvidia/nemotron-3-embed-1b
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100")) # characters
TOP_K = 4
