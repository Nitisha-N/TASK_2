"""Test the search on its own:  python code_helper/test_search.py "where is the search done?" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from code_helper.tools.rag import rag_search

print(rag_search(" ".join(sys.argv[1:]) or "where is the search done?"))
