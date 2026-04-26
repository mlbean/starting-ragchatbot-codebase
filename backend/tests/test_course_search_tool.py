"""Tests for CourseSearchTool.execute() output and source tracking."""

from unittest.mock import ANY

import pytest

from search_tools import CourseSearchTool
from vector_store import SearchResults


@pytest.fixture
def tool(fake_vector_store):
    return CourseSearchTool(fake_vector_store)


def test_forwards_all_params_to_vector_store(tool, fake_vector_store, empty_search_results):
    fake_vector_store.search.return_value = empty_search_results
    tool.execute(query="x", course_name="MCP", lesson_number=2)
    fake_vector_store.search.assert_called_once_with(
        query="x", course_name="MCP", lesson_number=2
    )


def test_successful_results_format(tool, fake_vector_store, sample_search_results):
    fake_vector_store.search.return_value = sample_search_results
    out = tool.execute(query="anything")
    assert isinstance(out, str)
    # Each chunk gets a header line
    assert "[MCP Basics - Lesson 1]" in out
    assert "[RAG Fundamentals - Lesson 2]" in out
    # Chunks are separated by blank lines
    assert "\n\n" in out


def test_results_without_lesson_number_omits_lesson_in_header(tool, fake_vector_store):
    results = SearchResults(
        documents=["intro chunk"],
        metadata=[{"course_title": "Solo", "lesson_number": None, "chunk_index": 0}],
        distances=[0.1],
    )
    fake_vector_store.search.return_value = results
    out = tool.execute(query="x")
    assert "[Solo]" in out
    assert "Lesson" not in out


def test_empty_results_no_filters(tool, fake_vector_store, empty_search_results):
    fake_vector_store.search.return_value = empty_search_results
    assert tool.execute(query="x") == "No relevant content found."


def test_empty_results_with_course_filter(tool, fake_vector_store, empty_search_results):
    fake_vector_store.search.return_value = empty_search_results
    out = tool.execute(query="x", course_name="MCP")
    assert out == "No relevant content found in course 'MCP'."


def test_empty_results_with_lesson_filter(tool, fake_vector_store, empty_search_results):
    fake_vector_store.search.return_value = empty_search_results
    out = tool.execute(query="x", lesson_number=2)
    assert out == "No relevant content found in lesson 2."


def test_empty_results_with_lesson_zero_still_mentions_lesson(
    tool, fake_vector_store, empty_search_results
):
    """Lesson 0 is a valid lesson number; the empty-results message should
    mention it. Will fail if `if lesson_number:` is used (truthy check) instead
    of `if lesson_number is not None:`."""
    fake_vector_store.search.return_value = empty_search_results
    out = tool.execute(query="x", lesson_number=0)
    assert out == "No relevant content found in lesson 0."


def test_error_short_circuits_formatting(tool, fake_vector_store, error_search_results):
    fake_vector_store.search.return_value = error_search_results
    out = tool.execute(query="x")
    assert out == "Search error: boom"
    # Lesson link lookup should not have been called because formatting was skipped.
    fake_vector_store.get_lesson_link.assert_not_called()


def test_last_sources_populated_and_deduped(
    tool, fake_vector_store, sample_search_results
):
    fake_vector_store.search.return_value = sample_search_results
    tool.execute(query="x")
    # Three chunks, two unique (course, lesson) pairs.
    assert len(tool.last_sources) == 2
    texts = [s["text"] for s in tool.last_sources]
    assert texts == ["MCP Basics - Lesson 1", "RAG Fundamentals - Lesson 2"]


def test_last_sources_includes_lesson_links(
    tool, fake_vector_store, sample_search_results
):
    fake_vector_store.search.return_value = sample_search_results
    tool.execute(query="x")
    links = {s["text"]: s["link"] for s in tool.last_sources}
    assert links["MCP Basics - Lesson 1"] == "https://example.com/MCP_Basics/lesson/1"
    assert (
        links["RAG Fundamentals - Lesson 2"]
        == "https://example.com/RAG_Fundamentals/lesson/2"
    )


def test_chunk_without_lesson_has_no_link(tool, fake_vector_store):
    results = SearchResults(
        documents=["intro chunk"],
        metadata=[{"course_title": "Solo", "lesson_number": None, "chunk_index": 0}],
        distances=[0.1],
    )
    fake_vector_store.search.return_value = results
    tool.execute(query="x")
    assert tool.last_sources == [{"text": "Solo", "link": None}]
    fake_vector_store.get_lesson_link.assert_not_called()


def test_tool_definition_shape(tool):
    defn = tool.get_tool_definition()
    assert defn["name"] == "search_course_content"
    schema = defn["input_schema"]
    assert "query" in schema["properties"]
    assert schema["required"] == ["query"]
    # Optional params present but not required.
    assert "course_name" in schema["properties"]
    assert "lesson_number" in schema["properties"]
