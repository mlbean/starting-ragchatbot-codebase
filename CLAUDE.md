# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager

Always use `uv` to run the server, Python files, and manage dependencies. Never use `pip` or `python` directly.

- `uv run python <file>.py`
- `uv run uvicorn ...`
- `uv sync` to install dependencies

## Commands

**Install dependencies:**
```bash
uv sync
```

**Run the server** (from repo root):
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is served at `http://localhost:8000` (web UI) and `http://localhost:8000/docs` (FastAPI docs).

**Environment setup:** copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY`.

## Architecture

This is a full-stack RAG chatbot: a FastAPI backend serving both the API and the static frontend from a single port.

**Request flow:**
1. Browser (`frontend/`) sends POST `/api/query` with a user message
2. `backend/app.py` receives it and calls `RAGSystem.query()`
3. `RAGSystem` (`rag_system.py`) builds a prompt and calls `AIGenerator.generate_response()` with a `search_course_content` tool available
4. Claude (via `ai_generator.py`) decides whether to call the search tool; if so, `ToolManager` routes the call to `CourseSearchTool`
5. `CourseSearchTool` (`search_tools.py`) calls `VectorStore.search()` which queries ChromaDB
6. ChromaDB returns semantically relevant chunks; Claude synthesizes a final answer
7. Sources and session history are updated; response returned to browser

**Key components:**

| File | Responsibility |
|------|----------------|
| `backend/rag_system.py` | Orchestrator — wires all components together |
| `backend/ai_generator.py` | Anthropic API calls; handles the tool-use loop (one round-trip max) |
| `backend/vector_store.py` | ChromaDB wrapper with two collections: `course_catalog` (course metadata) and `course_content` (chunked text) |
| `backend/document_processor.py` | Parses `.txt`/`.pdf`/`.docx` course files into `Course`/`Lesson`/`CourseChunk` models |
| `backend/search_tools.py` | `CourseSearchTool` (Anthropic tool definition + execution) and `ToolManager` registry |
| `backend/session_manager.py` | In-memory conversation history (trimmed to `MAX_HISTORY=2` exchanges) |
| `backend/config.py` | Single `Config` dataclass loaded from environment |
| `backend/models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |

**Document format** (`docs/*.txt`): course files must start with:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>
Lesson 1: <title>
Lesson Link: <url>
<content...>
Lesson 2: ...
```
The title is also used as the unique ChromaDB ID, so titles must be unique across all documents.

**ChromaDB storage** is persisted at `backend/chroma_db/`. On server startup, all files in `../docs` are loaded (skipping courses already in the store). To force a full rebuild, call `VectorStore.clear_all_data()` before loading.

**Tool-use pattern:** `AIGenerator` makes at most two API calls per query — one initial call (which may return a `tool_use` stop reason) and one follow-up after tool results are injected. Tools are not passed on the follow-up call to prevent chaining.
