"""Tests for agent.auxiliary_client._try_custom_endpoint's anthropic_messages branch.

When a user configures a custom endpoint with ``api_mode: anthropic_messages``
(e.g. MiniMax, Zhipu GLM, LiteLLM in Anthropic-proxy mode), auxiliary tasks
(compression, web_extract, session_search, title generation) must use the
native Anthropic transport rather than being silently downgraded to an
OpenAI-wire client that speaks the wrong protocol.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_fake_anthropic_namespace(build_side_effect=None, build_return=None):
    """Return a fake provider namespace dict and fake client for registry patching.

    _try_custom_endpoint() resolves build_anthropic_client through
    registries.get_provider_namespace("anthropic"), not via a direct module
    import, so we must patch that path (not hermes_agent_anthropic.adapter).
    """
    fake_client = MagicMock(name="anthropic_client")
    if build_side_effect is not None:
        mock_build = MagicMock(side_effect=build_side_effect)
    else:
        mock_build = MagicMock(return_value=build_return or fake_client)

    from agent.anthropic_aux import AnthropicAuxiliaryClient as _AC
    fake_ns = {
        "build_anthropic_client": mock_build,
        "AnthropicAuxiliaryClient": _AC,
        "AsyncAnthropicAuxiliaryClient": MagicMock(),
    }
    return fake_ns, fake_client, mock_build


def test_custom_endpoint_anthropic_messages_builds_anthropic_wrapper():
    """api_mode=anthropic_messages → returns AnthropicAuxiliaryClient, not OpenAI."""
    from agent.auxiliary_client import _try_custom_endpoint
    from agent.anthropic_aux import AnthropicAuxiliaryClient

    fake_ns, fake_client, _ = _make_fake_anthropic_namespace()

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=(
            "https://api.minimax.io/anthropic",
            "minimax-key",
            "anthropic_messages",
        ),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="claude-sonnet-4-6",
    ), patch(
        "agent.plugin_registries.registries",
    ) as mock_reg:
        mock_reg.get_provider_namespace.return_value = fake_ns
        mock_reg.get_provider_service.side_effect = lambda p, n: fake_ns.get(n) if p == "anthropic" else None
        client, model = _try_custom_endpoint()

    assert isinstance(client, AnthropicAuxiliaryClient), (
        "Custom endpoint with api_mode=anthropic_messages must return the "
        f"native Anthropic wrapper, got {type(client).__name__}"
    )
    assert model == "claude-sonnet-4-6"
    # Wrapper should NOT be marked as OAuth — third-party endpoints are
    # always API-key authenticated.
    assert client.api_key == "minimax-key"
    assert client.base_url == "https://api.minimax.io/anthropic"


def test_custom_endpoint_anthropic_messages_falls_back_when_sdk_missing():
    """Graceful degradation when anthropic SDK is unavailable."""
    from agent.auxiliary_client import _try_custom_endpoint

    import_error = ImportError("anthropic package not installed")
    fake_ns, _, _ = _make_fake_anthropic_namespace(build_side_effect=import_error)

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=("https://api.minimax.io/anthropic", "k", "anthropic_messages"),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="claude-sonnet-4-6",
    ), patch(
        "agent.plugin_registries.registries",
    ) as mock_reg:
        mock_reg.get_provider_namespace.return_value = fake_ns
        mock_reg.get_provider_service.side_effect = lambda p, n: fake_ns.get(n) if p == "anthropic" else None
        client, model = _try_custom_endpoint()

    # Should fall back to an OpenAI-wire client rather than returning
    # (None, None) — the tool still needs to do *something*.
    assert client is not None
    assert model == "claude-sonnet-4-6"
    # OpenAI client, not AnthropicAuxiliaryClient.
    from agent.anthropic_aux import AnthropicAuxiliaryClient
    assert not isinstance(client, AnthropicAuxiliaryClient)


def test_custom_endpoint_chat_completions_still_uses_openai_wire():
    """Regression: default path (no api_mode) must remain OpenAI client."""
    from agent.auxiliary_client import _try_custom_endpoint
    from agent.anthropic_aux import AnthropicAuxiliaryClient

    with patch(
        "agent.auxiliary_client._resolve_custom_runtime",
        return_value=("https://api.example.com/v1", "key", None),
    ), patch(
        "agent.auxiliary_client._read_main_model",
        return_value="my-model",
    ):
        client, model = _try_custom_endpoint()

    assert client is not None
    assert model == "my-model"
    assert not isinstance(client, AnthropicAuxiliaryClient)
