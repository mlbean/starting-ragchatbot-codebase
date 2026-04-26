"""End-to-end RAGSystem.query() tests for content questions.

Uses a real RAGSystem with the VectorStore and Anthropic client mocked so the
ToolManager wiring, source propagation, and history handling are exercised."""

import pytest


def _set_responses(mock_client, responses):
    mock_client.messages.create.side_effect = list(responses)


def test_query_returns_answer_and_sources(
    rag_system_with_mocks,
    mock_anthropic_client,
    fake_vector_store,
    sample_search_results,
    make_tool_use_response,
    make_text_response,
):
    fake_vector_store.search.return_value = sample_search_results
    _set_responses(
        mock_anthropic_client,
        [
            make_tool_use_response(
                [("search_course_content", {"query": "what is MCP"}, "tu_1")]
            ),
            make_text_response("MCP is a tool-use protocol."),
        ],
    )

    answer, sources = rag_system_with_mocks.query("what is MCP")
    assert answer == "MCP is a tool-use protocol."
    assert isinstance(sources, list) and len(sources) == 2
    assert all("text" in s and "link" in s for s in sources)


def test_sources_come_from_search_tool(
    rag_system_with_mocks,
    mock_anthropic_client,
    fake_vector_store,
    sample_search_results,
    make_tool_use_response,
    make_text_response,
):
    fake_vector_store.search.return_value = sample_search_results
    _set_responses(
        mock_anthropic_client,
        [
            make_tool_use_response(
                [("search_course_content", {"query": "x"}, "tu_1")]
            ),
            make_text_response("answer"),
        ],
    )
    _, sources = rag_system_with_mocks.query("x")
    texts = [s["text"] for s in sources]
    assert "MCP Basics - Lesson 1" in texts
    assert "RAG Fundamentals - Lesson 2" in texts


def test_sources_reset_between_queries(
    rag_system_with_mocks,
    mock_anthropic_client,
    fake_vector_store,
    sample_search_results,
    make_tool_use_response,
    make_text_response,
):
    """A second query that doesn't use the tool must return empty sources."""
    fake_vector_store.search.return_value = sample_search_results
    _set_responses(
        mock_anthropic_client,
        [
            # First query uses the tool.
            make_tool_use_response(
                [("search_course_content", {"query": "x"}, "tu_1")]
            ),
            make_text_response("answer-1"),
            # Second query: plain text, no tool use.
            make_text_response("answer-2"),
        ],
    )
    _, first_sources = rag_system_with_mocks.query("first")
    assert first_sources, "expected sources from first query"

    _, second_sources = rag_system_with_mocks.query("second")
    assert second_sources == [], (
        "sources from a previous query leaked into a query that did not "
        "invoke the search tool"
    )


def test_tool_definitions_include_both_tools(
    rag_system_with_mocks,
    mock_anthropic_client,
    make_text_response,
):
    _set_responses(mock_anthropic_client, [make_text_response("ok")])
    rag_system_with_mocks.query("anything")
    kwargs = mock_anthropic_client.messages.create.call_args.kwargs
    tool_names = {t["name"] for t in kwargs["tools"]}
    assert tool_names == {"search_course_content", "get_course_outline"}


def test_session_id_integrates_history(
    rag_system_with_mocks,
    mock_anthropic_client,
    make_text_response,
):
    sid = rag_system_with_mocks.session_manager.create_session()
    _set_responses(
        mock_anthropic_client,
        [make_text_response("first answer"), make_text_response("second answer")],
    )
    rag_system_with_mocks.query("first question?", session_id=sid)
    rag_system_with_mocks.query("second question?", session_id=sid)

    second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
    system = second_call_kwargs["system"]
    assert "Previous conversation:" in system
    assert "first question?" in system
    assert "first answer" in system


def test_no_session_id_skips_history(
    rag_system_with_mocks,
    mock_anthropic_client,
    make_text_response,
):
    _set_responses(mock_anthropic_client, [make_text_response("ok")])
    rag_system_with_mocks.query("hello")
    system = mock_anthropic_client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:" not in system


def test_user_message_contains_query(
    rag_system_with_mocks,
    mock_anthropic_client,
    make_text_response,
):
    _set_responses(mock_anthropic_client, [make_text_response("ok")])
    rag_system_with_mocks.query("what is MCP?")
    messages = mock_anthropic_client.messages.create.call_args.kwargs["messages"]
    user_text = messages[0]["content"]
    assert "what is MCP?" in user_text
