"""Run once to fill Pinecone:  python code_helper/index_codebase.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_helper import settings as S
from code_helper.tools import rag


def main():
    if not S.PROJECT_PATH.is_dir():
        sys.exit(f"PROJECT_PATH is not a folder: {S.PROJECT_PATH}")
    chunks = []
    for f in rag.read_files():
        for i, text in enumerate(rag.split_text(f["text"])):
            chunks.append({"id": f"{f['file']}::{i}", "file": f["file"], "text": text})
    print(f"{len(chunks)} chunks (size={S.CHUNK_SIZE}, overlap={S.CHUNK_OVERLAP})")
    print(f"Stored {rag.store_chunks(chunks)} chunks in '{S.PINECONE_INDEX_NAME}'")


if __name__ == "__main__":
    main()
