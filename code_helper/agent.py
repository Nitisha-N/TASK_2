"""Boss + 4 workers. ADK looks for `root_agent` in this file."""
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from . import settings as S
from .tools import (
    rag_search, list_project_files, get_imports,
    get_functions, find_importers, get_project_overview,
)

# One model object, shared by all agents.
model = LiteLlm(model=f"nvidia_nim/{S.NVIDIA_CHAT_MODEL}")

STOP_RULE = (
    "Call a tool only once. As soon as it returns a result, write the final "
    "answer in plain words and stop. Never call the same tool twice. "
)

code_searcher = LlmAgent(
    name="code_searcher",
    model=model,
    description="Finds WHERE something is in the code (a function, feature or setting) using semantic search.",
    instruction=(
        "Call rag_search once with the user's question. The results are relevant. "
        "Answer in 2 or 3 sentences: name the files listed after 'Most relevant files', "
        "and say what the code in those chunks does. "
        "Only if the result starts with 'NO RESULTS' or 'SEARCH ERROR', tell the user "
        "you could not find it. " + STOP_RULE
    ),
    tools=[rag_search],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# Filesystem MCP server: its tools appear on the agent. Only reads PROJECT_PATH.
file_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(S.PROJECT_PATH)],
        ),
        timeout=30,
    ),
    tool_filter=["list_directory", "read_file"],
)

code_explainer = LlmAgent(
    name="code_explainer",
    model=model,
    description="Reads ONE whole file and explains what it does, step by step. Use for 'explain X.py'.",
    instruction=(
        "The user names one file. Read ONLY that file with read_file, using the path "
        f"{S.PROJECT_PATH} plus a backslash and the file name. Read it once. "
        "Then explain its purpose and main functions in simple words. "
        "Do not read other files. " + STOP_RULE
    ),
    tools=[file_tools],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

architecture_mapper = LlmAgent(
    name="architecture_mapper",
    model=model,
    description="Shows how files connect: lists files, functions and imports, and which files import which.",
    instruction=(
        "Pick the ONE tool that fits the question: find_importers for 'which files "
        "import X', get_imports for 'what does X import', get_functions for 'what "
        "functions are in X', list_project_files for 'list the files'. "
        "Report exactly what the tool returns; do not guess. "
        "If the tool says nothing was found, reply 'I could not find that.' " + STOP_RULE
    ),
    tools=[list_project_files, get_imports, get_functions, find_importers],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

onboarding_guide = LlmAgent(
    name="onboarding_guide",
    model=model,
    description="Helps newcomers: what the project does, how to run it, and which file to read first.",
    instruction=(
        "Call get_project_overview once. Explain what the project does, how to run it, "
        "and a short reading order for the files. Keep it friendly and short. " + STOP_RULE
    ),
    tools=[get_project_overview],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

root_agent = LlmAgent(
    name="coordinator",
    model=model,
    description="Sends each question to the right worker.",
    instruction=(
        "Never answer yourself. Pick the best worker and hand over:\n"
        "- code_searcher: where is something?\n"
        "- code_explainer: explain one file\n"
        "- architecture_mapper: how files connect, imports, function lists\n"
        "- onboarding_guide: new here, what does it do, where to start"
    ),
    sub_agents=[code_searcher, code_explainer, architecture_mapper, onboarding_guide],
)