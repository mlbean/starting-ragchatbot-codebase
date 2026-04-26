"""Tests for AIGenerator's tool-calling behavior."""

import pytest


def test_no_tools_returns_text(
    ai_generator_with_mock_client, mock_anthropic_client, make_text_response
):
    mock_anthropic_client.messages.create.return_value = make_text_response("hello")
    out = ai_generator_with_mock_client.generate_response("hi")
    assert out == "hello"
    assert mock_anthropic_client.messages.create.call_count == 1


def test_tools_provided_sets_tool_choice_auto(
    ai_generator_with_mock_client,
    mock_anthropic_client,
    make_text_response,
    recording_tool_manager,
):
    mock_anthropic_client.messages.create.return_value = make_text_response("ok")
    tools = recording_tool_manager.get_tool_definitions()
    ai_generator_with_mock_client.generate_response(
        "q", tools=tools, tool_manager=recording_tool_manager
    )
    kwargs = mock_anthropic_client.messages.create.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == {"type": "auto"}


def test_conversation_history_appended_to_system(
    ai_generator_with_mock_client, mock_anthropic_client, make_text_response
):
    mock_anthropic_client.messages.create.return_value = make_text_response("ok")
    ai_generator_with_mock_client.generate_response(
        "q", conversation_history="User: hi\nAssistant: hello"
    )
    system = mock_anthropic_client.messages.create.call_args.kwargs["system"]
    assert system.endswith("Previous conversation:\nUser: hi\nAssistant: hello")


def test_tool_use_round_trip(
    ai_generator_with_mock_client,
    mock_anthropic_client,
    make_tool_use_response,
    make_text_response,
    recording_tool_manager,
):
    """First call returns tool_use; AIGenerator must execute the tool, send a
    follow-up call with the tool_result, and return the final text answer."""
    recording_tool_manager._tool_results = {"search_course_content": "tool-said-this"}
    first = make_tool_use_response(
        [("search_course_content", {"query": "what is MCP"}, "tu_abc")]
    )
    final = make_text_response("MCP is a protocol")
    mock_anthropic_client.messages.create.side_effect = [first, final]

    out = ai_generator_with_mock_client.generate_response(
        "what is MCP",
        tools=recording_tool_manager.get_tool_definitions(),
        tool_manager=recording_tool_manager,
    )
    assert out == "MCP is a protocol"

    # Tool was executed with the right arguments.
    assert recording_tool_manager.calls == [
        ("search_course_content", {"query": "what is MCP"})
    ]

    # Two API calls were made.
    assert mock_anthropic_client.messages.create.call_count == 2

    # Follow-up call must have the assistant tool_use turn followed by a
    # user turn carrying a tool_result with the same id.
    second_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
    messages = second_kwargs["messages"]
    assert len(messages) >= 3
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    tool_results = messages[-1]["content"]
    assert isinstance(tool_results, list) and len(tool_results) == 1
    assert tool_results[0]["type"] == "tool_result"
    assert tool_results[0]["tool_use_id"] == "tu_abc"
    assert tool_results[0]["content"] == "tool-said-this"


def test_final_call_drops_tools(
    ai_generator_with_mock_client,
    mock_anthropic_client,
    make_tool_use_response,
    make_text_response,
    recording_tool_manager,
):
    """The follow-up call after tool execution must NOT include `tools` /
    `tool_choice` — preventing infinite tool-call chaining."""
    first = make_tool_use_response(
        [("search_course_content", {"query": "x"}, "tu_1")]
    )
    final = make_text_response("done")
    mock_anthropic_client.messages.create.side_effect = [first, final]

    ai_generator_with_mock_client.generate_response(
        "x",
        tools=recording_tool_manager.get_tool_definitions(),
        tool_manager=recording_tool_manager,
    )
    second_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
    assert "tools" not in second_kwargs
    assert "tool_choice" not in second_kwargs


def test_multiple_tool_use_blocks_all_executed(
    ai_generator_with_mock_client,
    mock_anthropic_client,
    make_tool_use_response,
    make_text_response,
    recording_tool_manager,
):
    recording_tool_manager._tool_results = {
        "search_course_content": "search-result",
        "get_course_outline": "outline-result",
    }
    first = make_tool_use_response(
        [
            ("search_course_content", {"query": "a"}, "tu_1"),
            ("get_course_outline", {"course_title": "MCP"}, "tu_2"),
        ]
    )
    final = make_text_response("combined")
    mock_anthropic_client.messages.create.side_effect = [first, final]

    ai_generator_with_mock_client.generate_response(
        "compound",
        tools=recording_tool_manager.get_tool_definitions(),
        tool_manager=recording_tool_manager,
    )

    names_called = [c[0] for c in recording_tool_manager.calls]
    assert names_called == ["search_course_content", "get_course_outline"]

    second_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
    tool_results = second_kwargs["messages"][-1]["content"]
    assert [t["tool_use_id"] for t in tool_results] == ["tu_1", "tu_2"]
    assert [t["content"] for t in tool_results] == ["search-result", "outline-result"]


def test_end_turn_skips_tool_execution(
    ai_generator_with_mock_client,
    mock_anthropic_client,
    make_text_response,
    recording_tool_manager,
):
    mock_anthropic_client.messages.create.return_value = make_text_response("plain")
    ai_generator_with_mock_client.generate_response(
        "q",
        tools=recording_tool_manager.get_tool_definitions(),
        tool_manager=recording_tool_manager,
    )
    assert recording_tool_manager.calls == []
    assert mock_anthropic_client.messages.create.call_count == 1
