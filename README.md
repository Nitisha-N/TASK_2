# Multi-Agent Code Helper

This is a helper that answers questions about a small Python project (Naive_RAG).
It is a team of 5 AI agents made with Google ADK: one boss and four workers.
The boss reads your question and sends it to the right worker. The search part uses
a RAG pipeline that I wrote, with Pinecone as the database.

## The agents

| Agent | What it does | Tools |
|---|---|---|
| coordinator (boss) | Reads the question and picks a worker. It never answers itself. | sub_agents |
| code_searcher | Finds where something is in the code | rag_search |
| code_explainer | Reads one whole file and explains it | Filesystem MCP server (read_file, list_directory) |
| architecture_mapper | Shows which files import which, and what functions a file has | list_project_files, get_imports, get_functions, find_importers |
| onboarding_guide | Tells a new person what the project does and where to start | get_project_overview |

## The tools

I wrote 6 tools. The two MCP tools (read_file, list_directory) come from the server, so I
did not count them.

| Tool | Uses AI? | What it does |
|---|---|---|
| rag_search | Only for embeddings | Turns the question into numbers and gets the closest code chunks from Pinecone |
| list_project_files | No | Lists the .py files and how many lines each has |
| get_imports | No | Shows what one file imports (uses Python's ast module) |
| get_functions | No | Shows the functions and classes in one file (ast) |
| find_importers | No | Shows which files import a module |
| get_project_overview | No | Gives the README, the file list and the top of main.py |

## My RAG pipeline

I did not use LlamaIndex or LangChain. I wrote each step:

1. **Read** all the .py files in the project.
2. **Split** them into chunks that overlap a little.
3. **Embed** each chunk with nvidia/nemotron-3-embed-1b (NVIDIA NIM).
4. **Store** the numbers and the text in Pinecone (2048 dimensions, cosine).
5. **Search**: embed the question and ask Pinecone for the 4 closest chunks.

## Chunk size

I tried two sizes and asked both the same 3 questions.

| Setting | Number of chunks |
|---|---|
| 400 characters, overlap 50 | 127 |
| 800 characters, overlap 100 | 62 |

| Question | 400 / 50 | 800 / 100 |
|---|---|---|
| Where is the search done? | Weak. Only 1 of the 4 chunks had search code. The others were imports and docstrings. | Better. It found the real `index.query(...)` code in vector_store.py and the retrieval part of query_pipeline.py. |
| How are chunks stored in Pinecone? | Good, but the upsert code was split over 3 chunks. | Good. It found store_in_vector_db and the upsert function. |
| Where are the API keys loaded? | All 4 chunks in both runs came from config.py, including the load_environment_variables function, which is the code that loads the keys. | Same outcome.|

Both sizes were fine for small, specific questions. But 400 did badly on the bigger
question, because the code got cut into too many small pieces. With 800, more of each
function stayed together. The main job of the Code Searcher is "where is this?", so I
chose **800 with overlap 100**. The overlap is there so a line at the edge of a chunk
does not get lost.

I only tested 3 questions, so this is not a perfect test, but it was enough for me to choose.


## Why I used the MCP server

The Code Explainer uses `@modelcontextprotocol/server-filesystem`. Search only gives back
small pieces of code, but to explain a file you need the whole file. The server reads the
file exactly as it is. I also made it so it can only read the project folder.

## How to run it

1. Get the Naive_RAG project and put it in a folder called `code_project`, next to `code_helper`.
2. Copy `.env.example` to `.env`. Put in your own NVIDIA key and Pinecone key, and set
   `PROJECT_PATH` to the full path of `code_project`. Never upload your real `.env`.
3. Run these from the folder that has `code_helper` inside it:

```
pip install -r code_helper/requirements.txt
python code_helper/index_codebase.py
python code_helper/test_search.py "where is the search done?"
python -m google.adk.cli web
```

The first one installs everything. The second fills Pinecone (run it once). The third
tests the search on its own. The last one starts the web page, where you pick
`code_helper` and chat. If the `adk` command works for you, `adk web` does the same thing.

## Example questions

**1. "Where is the search done?"** (agent: code_searcher)
It said the search is in query_pipeline.py and vector_store.py, where the top-k chunks
come from Pinecone (or a local fallback). It also listed config.py, which only loads
settings, so that part was a small mistake.

**2. "Which files import config.py?"** (agent: architecture_mapper)
It answered: indexer.py, main.py, query_pipeline.py, vector_store.py.

**3. "Explain vector_store.py"** (agent: code_explainer)
It explained that the file handles Pinecone with a local JSON backup. It listed the
functions: cosine similarity, making the client, checking the index exists, upsert and query.

**4. "I am new. Where do I start?"** (agent: onboarding_guide)
It explained what the project does, how to run it, and which files to look at.

## A problem I hit and how I fixed it

At first the Architecture Mapper called `find_importers` over 200 times and never gave
an answer. The workers were also passing questions to each other by mistake. I fixed it
by telling every worker to call its tool once, then write the answer and stop. I also
turned off transfers between workers with `disallow_transfer_to_parent` and
`disallow_transfer_to_peers`.

A second problem was that the Code Searcher kept saying "I could not find that", even
though `rag_search` had returned the right code. The small model seemed to copy that
sentence from its instructions. I rewrote the instructions, and I made the search result
list the most relevant files first. After that it answered properly.

## What I learned

- Chunk size changes the search results more than I expected.
- Plain Python tools are better for structure questions (imports, functions), and
  search is better for "where is this?" questions.
- The instructions you give an agent matter a lot. A small model can follow them in a
  silly way, like repeating a tool call forever.

## What is in this repo

```
TASK_2/
├── README.md
└── code_helper/
    ├── __init__.py
    ├── agent.py              (the 5 agents)
    ├── settings.py           (reads the .env file)
    ├── index_codebase.py     (fills Pinecone)
    ├── test_search.py        (tests the search)
    ├── requirements.txt
    ├── .env.example          (fake keys, copy to .env)
    ├── .gitignore
    └── tools/
        ├── __init__.py
        ├── rag.py            (my RAG pipeline)
        └── analysis.py       (the plain Python tools)
```

The Naive_RAG project that the helper reads is not part of this repo.
