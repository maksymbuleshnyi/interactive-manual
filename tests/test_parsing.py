"""Tests for tolerant LLM-response JSON parsing and the Claude adapter shape."""

import json

import pytest

from interactive_interfaces.utils.io import parse_json_response


def test_parses_plain_json_object():
    assert parse_json_response('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_parses_plain_json_list():
    assert parse_json_response('[{"x": 1}, {"x": 2}]') == [{"x": 1}, {"x": 2}]


def test_strips_json_code_fence():
    fenced = '```json\n{"ok": true}\n```'
    assert parse_json_response(fenced) == {"ok": True}


def test_strips_bare_code_fence():
    fenced = '```\n[1, 2, 3]\n```'
    assert parse_json_response(fenced) == [1, 2, 3]


def test_tolerates_surrounding_whitespace():
    assert parse_json_response('  \n {"k": "v"}\n  ') == {"k": "v"}


def test_malformed_json_still_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("not json at all")


def test_claude_adapter_constructs_and_matches_protocol():
    pytest.importorskip("anthropic")
    from interactive_interfaces.models.base import LLMClient
    from interactive_interfaces.models.claude_llm import ClaudeLLM

    # api_key is supplied so construction needs no environment variable; no
    # network call is made here.
    llm = ClaudeLLM(api_key="test-key-not-used", model="claude-opus-4-7")
    assert llm.name == "claude:claude-opus-4-7"
    assert llm.model == "claude-opus-4-7"
    assert isinstance(llm, LLMClient)  # structural Protocol check
