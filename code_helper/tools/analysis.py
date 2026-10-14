"""Plain-Python tools (no AI). They read the project with the ast module."""
import ast
from pathlib import Path

from .. import settings as S


def _py_files() -> list[Path]:
    return [
        p for p in sorted(S.PROJECT_PATH.rglob("*.py"))
        if not any(x in p.parts for x in (".venv", "venv", "__pycache__", ".git"))
    ]


def _resolve(file_name: str) -> Path | None:
    """Find a file inside the project (blocks paths outside it)."""
    for p in _py_files():
        if p.name == file_name or str(p.relative_to(S.PROJECT_PATH)) == file_name:
            return p
    return None


def list_project_files() -> str:
    """List every Python file in the project with its line count."""
    files = _py_files()
    if not files:
        return "No Python files found. Check PROJECT_PATH."
    lines = []
    for p in files:
        n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        lines.append(f"{p.relative_to(S.PROJECT_PATH)} ({n} lines)")
    return "\n".join(lines)


def get_imports(file_name: str) -> str:
    """List what one file imports (for example 'main.py'). Use it to see a file's dependencies."""
    path = _resolve(file_name)
    if not path:
        return f"File not found: {file_name}"
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(("." * node.level) + (node.module or ""))
    return f"{file_name} imports: " + (", ".join(sorted(set(found))) or "nothing")


def get_functions(file_name: str) -> str:
    """List the classes and functions defined in one file, with line numbers and first docstring line."""
    path = _resolve(file_name)
    if not path:
        return f"File not found: {file_name}"
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            doc = (ast.get_docstring(node) or "").split("\n")[0]
            out.append(f"line {node.lineno}: {kind} {node.name}  {doc}".rstrip())
    return "\n".join(sorted(out, key=lambda s: int(s.split(":")[0][5:]))) or "No functions or classes."


def find_importers(module_name: str) -> str:
    """Find which files import a given module (for example 'config' or 'config.py')."""
    target = module_name.removesuffix(".py").split(".")[-1]
    hits = []
    for p in _py_files():
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[-1] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[-1]] + [a.name for a in node.names]
            if target in names:
                hits.append(str(p.relative_to(S.PROJECT_PATH)))
                break
    return (f"Files that import {target}: " + ", ".join(hits)) if hits else f"No file imports {target}."


def get_project_overview() -> str:
    """Get the README (if any), the file list and the top of main.py. Use it to explain what the project does and how to start."""
    parts = []
    for name in ("README.md", "readme.md", "README.txt"):
        r = S.PROJECT_PATH / name
        if r.exists():
            parts.append(f"README:\n{r.read_text(encoding='utf-8', errors='replace')[:3000]}")
            break
    parts.append("FILES:\n" + list_project_files())
    main = _resolve("main.py")
    if main:
        parts.append("main.py (first 60 lines):\n" +
                     "\n".join(main.read_text(encoding="utf-8", errors="replace").splitlines()[:60]))
    return "\n\n".join(parts)
