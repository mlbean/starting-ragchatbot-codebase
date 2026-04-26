"""Shared pytest fixtures for the RAG backend test suite.

Mocks the Anthropic client and (where appropriate) the VectorStore so the
tests are fast, hermetic, and never make real API calls or touch ChromaDB.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


# ---------- SearchResults fixtures ----------


@pytest.fixture
def sample_search_results() -> SearchResults:
    """Three chunks across two unique (course, lesson) pairs.

    Used to verify formatting and source-deduplication behavior.
    """
    return SearchResults(
        documents=[
            "MCP is a protocol for tool use.",
            "It standardizes how models talk to tools.",
            "Retrieval augments generation with external knowledge.",
        ],
        metadata=[
            {"course_title": "MCP Basics", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "MCP Basics", "lesson_number": 1, "chunk_index": 1},
            {"course_title": "RAG Fundamentals", "lesson_number": 2, "chunk_index": 0},
        ],
        distances=[0.1, 0.2, 0.3],
    )


@pytest.fixture
def empty_search_results() -> SearchResults:
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results() -> SearchResults:
    return SearchResults.empty("Search error: boom")


# ---------- VectorStore mock ----------


@pytest.fixture
def fake_vector_store() -> MagicMock:
    """A MagicMock standing in for VectorStore.

    Tests configure `.search.return_value` per-case. `get_lesson_link` returns
    a deterministic fake URL keyed on (course_title, lesson_number).
    """
    store = MagicMock(spec=VectorStore)

    def _lesson_link(course_title, lesson_number):
        if lesson_number is None:
            return None
        return f"https://example.com/{course_title.replace(' ', '_')}/lesson/{lesson_number}"

    store.get_lesson_link.side_effect = _lesson_link
    return store


# ---------- Anthropic response/client helpers ----------


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, tool_input: dict, tool_use_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use", name=name, input=tool_input, id=tool_use_id
    )


@pytest.fixture
def make_text_response():
    """Factory for a fake `messages.create` response with stop_reason='end_turn'."""

    def _make(text: str) -> SimpleNamespace:
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[_text_block(text)],
        )

    return _make


@pytest.fixture
def make_tool_use_response():
    """Factory for a fake `messages.create` response with stop_reason='tool_use'.

    `tool_calls` is a list of (tool_name, tool_input, tool_use_id) tuples.
    """

    def _make(tool_calls):
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _tool_use_block(name, inp, tid) for (name, inp, tid) in tool_calls
            ],
        )

    return _make


@pytest.fixture
def mock_anthropic_client():
    """Mock anthropic.Anthropic client whose `messages.create` is a MagicMock.

    Tests typically do:
        mock_anthropic_client.messages.create.side_effect = [resp1, resp2]
    so a tool_use response is followed by a final text response.
    """
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = MagicMock()
    return client


@pytest.fixture
def ai_generator_with_mock_client(mock_anthropic_client) -> AIGenerator:
    """An AIGenerator whose underlying anthropic client is mocked out."""
    gen = AIGenerator(api_key="test-key", model="claude-test")
    gen.client = mock_anthropic_client
    return gen


# ---------- ToolManager / RAGSystem fixtures ----------


class RecordingToolManager:
    """Stand-in for ToolManager that records `execute_tool` calls."""

    def __init__(self, tool_definitions=None, tool_results=None):
        self._definitions = tool_definitions or [
            {
                "name": "search_course_content",
                "description": "test",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        # tool_results: dict[name] -> str OR a single str applied to all.
        self._tool_results = tool_results or {}
        self.calls = []  # list of (name, kwargs)

    def get_tool_definitions(self):
        return self._definitions

    def execute_tool(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if isinstance(self._tool_results, str):
            return self._tool_results
        return self._tool_results.get(name, f"result-for-{name}")

    def get_last_sources(self):
        return []

    def reset_sources(self):
        pass


@pytest.fixture
def recording_tool_manager() -> RecordingToolManager:
    return RecordingToolManager()


@pytest.fixture
def rag_config():
    """Minimal config-like object for RAGSystem construction.

    Uses the real config dataclass so attribute access matches production.
    Overrides paths so any accidental ChromaDB hit lands in a tmp location
    (the vector store is mocked anyway, but belt-and-braces).
    """
    from config import Config

    cfg = Config()
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-test"
    cfg.CHROMA_PATH = "/tmp/_rag_tests_should_not_use_this"
    return cfg


@pytest.fixture
def rag_system_with_mocks(
    rag_config, fake_vector_store, mock_anthropic_client, monkeypatch
):
    """A real RAGSystem with VectorStore and the Anthropic client mocked.

    The session manager and tool manager are real so we can verify the wiring
    end-to-end (history fetching, source propagation, tool registration).
    """
    # Patch VectorStore so RAGSystem.__init__ doesn't open a real Chroma DB.
    monkeypatch.setattr(
        "rag_system.VectorStore", lambda *a, **kw: fake_vector_store
    )
    rag = RAGSystem(rag_config)
    # Replace the Anthropic client with the mock.
    rag.ai_generator.client = mock_anthropic_client
    return rag
