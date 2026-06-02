"""Tests for the transport ABC, registry, and AnthropicTransport."""

import pytest
from types import SimpleNamespace

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse
from agent.transports import get_transport, register_transport, _REGISTRY


# ── ABC contract tests ──────────────────────────────────────────────────

class TestProviderTransportABC:
    """Verify the ABC contract is enforceable."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ProviderTransport()

    def test_concrete_must_implement_all_abstract(self):
        class Incomplete(ProviderTransport):
            @property
            def api_mode(self):
                return "test"
        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_concrete(self):
        class Minimal(ProviderTransport):
            @property
            def api_mode(self):
                return "test_minimal"
            def convert_messages(self, messages, **kw):
                return messages
            def convert_tools(self, tools):
                return tools
            def build_kwargs(self, model, messages, tools=None, **params):
                return {"model": model, "messages": messages}
            def normalize_response(self, response, **kw):
                return NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")

        t = Minimal()
        assert t.api_mode == "test_minimal"
        assert t.validate_response(None) is True  # default
        assert t.extract_cache_stats(None) is None  # default
        assert t.map_finish_reason("end_turn") == "end_turn"  # default passthrough


# ── Registry tests ───────────────────────────────────────────────────────

class TestTransportRegistry:

    def test_get_unregistered_returns_none(self):
        assert get_transport("nonexistent_mode") is None

    def test_anthropic_registered_via_plugin(self):
        """Anthropic transport is registered by the anthropic plugin, not a core module."""
        # Register a mock transport via the registry API — core tests must
        # not import hermes_agent_* packages.
        from agent.plugin_registries import registries
        from agent.transports.base import ProviderTransport
        from agent.transports import register_transport

        class _MockAnthropicTransport(ProviderTransport):
            @property
            def api_mode(self):
                return "anthropic_messages"
            def convert_messages(self, messages, **kw):
                return messages
            def convert_tools(self, tools, **kw):
                return tools
            def convert_response(self, resp, **kw):
                return resp
            def build_kwargs(self, model, messages, tools=None, **params):
                return {}
            def normalize_response(self, response, **kw):
                return NormalizedResponse(content=None, tool_calls=None, finish_reason="stop")

        register_transport("anthropic_messages", _MockAnthropicTransport)
        t = get_transport("anthropic_messages")
        assert t is not None
        assert t.api_mode == "anthropic_messages"

    def test_discovers_missing_transport_when_registry_partially_populated(self):
        """Importing one transport directly must not hide other valid api_modes."""
        import agent.transports.chat_completions  # noqa: F401
        t = get_transport("codex_responses")
        assert t is not None
        assert t.api_mode == "codex_responses"

    def test_register_and_get(self):
        class DummyTransport(ProviderTransport):
            @property
            def api_mode(self):
                return "dummy_test"
            def convert_messages(self, messages, **kw):
                return messages
            def convert_tools(self, tools):
                return tools
            def build_kwargs(self, model, messages, tools=None, **params):
                return {}
            def normalize_response(self, response, **kw):
                return NormalizedResponse(content=None, tool_calls=None, finish_reason="stop")

        register_transport("dummy_test", DummyTransport)
        t = get_transport("dummy_test")
        assert t.api_mode == "dummy_test"
        # Cleanup
        _REGISTRY.pop("dummy_test", None)


# ── AnthropicTransport tests ────────────────────────────────────────────

class TestAnthropicTransport:
    """Core transport registry checks for the anthropic transport slot.

    Full behavioral tests (convert_tools, validate_response, normalize_response,
    build_kwargs, etc.) live in plugins/model-providers/anthropic/tests/ where
    the real transport implementation is available.
    """

    @pytest.fixture
    def transport(self):
        # Register a mock anthropic transport via the registry API.
        from agent.transports.base import ProviderTransport
        from agent.transports import register_transport

        class _MockAnthropicTransport(ProviderTransport):
            @property
            def api_mode(self):
                return "anthropic_messages"
            def convert_messages(self, messages, **kw):
                return messages
            def convert_tools(self, tools, **kw):
                return tools
            def convert_response(self, resp, **kw):
                return resp
            def build_kwargs(self, model, messages, tools=None, **params):
                return {}
            def normalize_response(self, response, **kw):
                return NormalizedResponse(content=None, tool_calls=None, finish_reason="stop")

        register_transport("anthropic_messages", _MockAnthropicTransport)
        return get_transport("anthropic_messages")

    def test_api_mode(self, transport):
        assert transport.api_mode == "anthropic_messages"
