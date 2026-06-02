"""Tests for the AnthropicMessagesTransport.

Behavioral tests that require the real anthropic transport implementation.
"""

import json
import pytest
from types import SimpleNamespace

from agent.transports import get_transport
from agent.transports.types import NormalizedResponse


@pytest.fixture
def transport():
    """Load the real Anthropic transport by registering the plugin."""
    from hermes_agent_anthropic import register as _anthro_register
    from agent.plugin_registries import registries

    class _Ctx:
        def register_transport(self, api_mode, obj):
            from agent.transports import register_transport
            register_transport(api_mode, obj)
        def register_provider_resolver(self, name, fn):
            registries.register_provider_resolver(name, fn)
        def register_provider_services(self, name, services):
            registries.register_provider_services(name, services)
        def register_credential_pool_hook(self, name, hook):
            registries.register_credential_pool_hook(name, hook)
        def register_pricing_provider(self, name, entries):
            registries.register_pricing_provider(name, entries)
        def register_provider_overlay(self, entry):
            registries.register_provider_overlay(entry)
        def __getattr__(self, name):
            if name.startswith("register_"):
                return lambda *a, **kw: None
            raise AttributeError(name)

    _anthro_register(_Ctx())
    return get_transport("anthropic_messages")



class TestAnthropicTransportBehavioral:

    # (fixture defined at module level above)

    def test_api_mode(self, transport):
        assert transport.api_mode == "anthropic_messages"

    def test_convert_tools_simple(self, transport):
        tools = [{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test",
                "parameters": {"type": "object", "properties": {}},
            }
        }]
        result = transport.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert "input_schema" in result[0]

    def test_validate_response_none(self, transport):
        assert transport.validate_response(None) is False

    def test_validate_response_empty_content(self, transport):
        r = SimpleNamespace(content=[])
        assert transport.validate_response(r) is False

    def test_validate_response_empty_content_with_end_turn_is_valid(self, transport):
        r = SimpleNamespace(content=[], stop_reason="end_turn")
        assert transport.validate_response(r) is True

    def test_validate_response_empty_content_with_tool_use_is_invalid(self, transport):
        r = SimpleNamespace(content=[], stop_reason="tool_use")
        assert transport.validate_response(r) is False

    def test_validate_response_valid(self, transport):
        r = SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])
        assert transport.validate_response(r) is True

    def test_map_finish_reason(self, transport):
        assert transport.map_finish_reason("end_turn") == "stop"
        assert transport.map_finish_reason("tool_use") == "tool_calls"
        assert transport.map_finish_reason("max_tokens") == "length"
        assert transport.map_finish_reason("stop_sequence") == "stop"
        assert transport.map_finish_reason("refusal") == "content_filter"
        assert transport.map_finish_reason("model_context_window_exceeded") == "length"
        assert transport.map_finish_reason("unknown") == "stop"

    def test_extract_cache_stats_none_usage(self, transport):
        r = SimpleNamespace(usage=None)
        assert transport.extract_cache_stats(r) is None

    def test_extract_cache_stats_with_cache(self, transport):
        usage = SimpleNamespace(cache_read_input_tokens=100, cache_creation_input_tokens=50)
        r = SimpleNamespace(usage=usage)
        result = transport.extract_cache_stats(r)
        assert result == {"cached_tokens": 100, "creation_tokens": 50}

    def test_extract_cache_stats_zero(self, transport):
        usage = SimpleNamespace(cache_read_input_tokens=0, cache_creation_input_tokens=0)
        r = SimpleNamespace(usage=usage)
        assert transport.extract_cache_stats(r) is None

    def test_normalize_response_text(self, transport):
        """Test normalization of a simple text response."""
        r = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello world")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            model="claude-sonnet-4-6",
        )
        nr = transport.normalize_response(r)
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == "Hello world"
        assert nr.tool_calls is None or nr.tool_calls == []
        assert nr.finish_reason == "stop"

    def test_normalize_response_tool_calls(self, transport):
        """Test normalization of a tool-use response."""
        r = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_123",
                    name="terminal",
                    input={"command": "ls"},
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            model="claude-sonnet-4-6",
        )
        nr = transport.normalize_response(r)
        assert nr.finish_reason == "tool_calls"
        assert len(nr.tool_calls) == 1
        tc = nr.tool_calls[0]
        assert tc.name == "terminal"
        assert tc.id == "toolu_123"
        assert '"command"' in tc.arguments

    def test_normalize_response_thinking(self, transport):
        """Test normalization preserves thinking content."""
        r = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Let me think..."),
                SimpleNamespace(type="text", text="The answer is 42"),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=15),
            model="claude-sonnet-4-6",
        )
        nr = transport.normalize_response(r)
        assert nr.content == "The answer is 42"
        assert nr.reasoning == "Let me think..."

    def test_build_kwargs_returns_dict(self, transport):
        """Test build_kwargs produces a usable kwargs dict."""
        messages = [{"role": "user", "content": "Hello"}]
        kw = transport.build_kwargs(
            model="claude-sonnet-4-6",
            messages=messages,
            max_tokens=1024,
        )
        assert isinstance(kw, dict)
        assert "model" in kw
        assert "max_tokens" in kw
        assert "messages" in kw

    def test_convert_messages_extracts_system(self, transport):
        """Test convert_messages separates system from messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, msgs = transport.convert_messages(messages)
        # System should be extracted
        assert system is not None
        # Messages should only have user
        assert len(msgs) >= 1
