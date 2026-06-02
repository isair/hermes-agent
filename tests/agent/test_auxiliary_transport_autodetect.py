"""Tests for transport auto-detection in agent.auxiliary_client.

Auxiliary clients must pick the correct wire protocol (OpenAI
chat.completions vs native Anthropic Messages) based on the endpoint,
regardless of which resolve_provider_client branch built them.

Regression target (April 2026): Kimi Coding Plan's ``api.kimi.com/coding``
endpoint only speaks Anthropic Messages — sending ``kimi-for-coding`` over
chat.completions returns 404 "resource_not_found_error".  The named
``kimi-coding`` provider branch in resolve_provider_client used to build a
plain OpenAI client, so title generation / vision / compression /
web_extract all failed on Kimi Coding Plan users.

NOTE: Core tests must NEVER import from hermes_agent_* plugin packages.
All plugin behaviour is simulated through the registry mock namespace
provided by the conftest.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
        "KIMI_API_KEY", "KIMI_CODING_API_KEY", "KIMI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Helpers — get services from the registry mock namespace (not plugin imports)
# ---------------------------------------------------------------------------

from agent.anthropic_aux import AnthropicAuxiliaryClient as _CoreAnthropicAuxiliaryClient
def _get_anthropic_service(name):
    """Look up an anthropic service from the registry (mock namespace)."""
    from agent.plugin_registries import registries
    return registries.get_provider_service("anthropic", name)


# ---------------------------------------------------------------------------
# URL detection helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected,label", [
    ("https://api.kimi.com/coding/v1", True, "Kimi Coding Plan /v1"),
    ("https://api.kimi.com/coding", True, "Kimi Coding Plan no /v1"),
    ("https://api.moonshot.ai/v1", False, "Moonshot legacy"),
    ("https://api.minimax.io/anthropic", True, "MiniMax /anthropic"),
    ("https://litellm.example.com/v1/anthropic", True, "/anthropic suffix"),
    ("https://api.anthropic.com", True, "native Anthropic"),
    ("https://api.anthropic.com/v1", True, "native Anthropic /v1"),
    ("https://openrouter.ai/api/v1", False, "OpenRouter"),
    ("https://api.openai.com/v1", False, "OpenAI"),
    ("https://inference-api.nousresearch.com/v1", False, "Nous"),
    ("", False, "empty"),
    (None, False, "None"),
])
def testendpoint_speaks_anthropic_messages(url, expected, label):
    endpoint_speaks = _get_anthropic_service("endpoint_speaks_anthropic_messages")
    assert endpoint_speaks(url) is expected, (
        f"{label}: {url!r} should be {expected}"
    )


# ---------------------------------------------------------------------------
# maybe_wrap_anthropic decision table
# ---------------------------------------------------------------------------

def test_maybe_wrap_anthropic_rewraps_kimi_coding_url():
    """Plain OpenAI client pointed at api.kimi.com/coding gets rewrapped."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient
    from agent.plugin_registries import registries

    plain_client = MagicMock(name="plain_openai")
    fake_anthropic = MagicMock(name="anthropic_sdk_client")

    with patch.dict(registries._provider_services, {"anthropic": {
        **registries._provider_services.get("anthropic", {}),
        "build_anthropic_client": MagicMock(return_value=fake_anthropic),
        "resolve_anthropic_token": MagicMock(return_value="sk-test"),
        "_is_oauth_token": lambda k: False,
        "build_anthropic_kwargs": MagicMock(return_value={}),
    }}):
        result = maybe_wrap(
            plain_client, "kimi-for-coding", "sk-kimi-test",
            "https://api.kimi.com/coding", api_mode=None,
        )
    assert isinstance(result, AnthropicAuxiliaryClient)


def test_maybe_wrap_anthropic_rewraps_slash_anthropic_url():
    """Plain OpenAI client pointed at any /anthropic URL gets rewrapped."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient
    from agent.plugin_registries import registries

    plain_client = MagicMock(name="plain_openai")
    fake_anthropic = MagicMock(name="anthropic_sdk_client")

    with patch.dict(registries._provider_services, {"anthropic": {
        **registries._provider_services.get("anthropic", {}),
        "build_anthropic_client": MagicMock(return_value=fake_anthropic),
        "resolve_anthropic_token": MagicMock(return_value="sk-test"),
        "_is_oauth_token": lambda k: False,
        "build_anthropic_kwargs": MagicMock(return_value={}),
    }}):
        result = maybe_wrap(
            plain_client, "MiniMax-M2.7", "mm-key",
            "https://api.minimax.io/anthropic", api_mode=None,
        )
    assert isinstance(result, AnthropicAuxiliaryClient)


def test_maybe_wrap_anthropic_skips_openai_wire_urls():
    """OpenRouter / OpenAI / Moonshot-legacy stay as plain OpenAI clients."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient

    plain_client = MagicMock(name="plain_openai")
    # No patch on build_anthropic_client — if the function tried to call it,
    # we'd get an AttributeError-style failure. The point is it shouldn't.
    result = maybe_wrap(
        plain_client, "claude-sonnet-4.6", "sk-or-test",
        "https://openrouter.ai/api/v1", api_mode=None,
    )
    assert result is plain_client
    assert not isinstance(result, AnthropicAuxiliaryClient)


def test_maybe_wrap_anthropic_respects_explicit_chat_completions():
    """api_mode=chat_completions overrides URL heuristics."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient

    plain_client = MagicMock(name="plain_openai")
    result = maybe_wrap(
        plain_client, "kimi-for-coding", "sk-kimi-test",
        "https://api.kimi.com/coding",
        api_mode="chat_completions",  # explicit override
    )
    assert result is plain_client, "Explicit chat_completions must bypass wrap"
    assert not isinstance(result, AnthropicAuxiliaryClient)


def test_maybe_wrap_anthropic_honors_explicit_anthropic_messages():
    """api_mode=anthropic_messages wraps even when URL wouldn't trigger."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient
    from agent.plugin_registries import registries

    plain_client = MagicMock(name="plain_openai")
    fake_anthropic = MagicMock(name="anthropic_sdk_client")

    with patch.dict(registries._provider_services, {"anthropic": {
        **registries._provider_services.get("anthropic", {}),
        "build_anthropic_client": MagicMock(return_value=fake_anthropic),
        "resolve_anthropic_token": MagicMock(return_value="sk-test"),
        "_is_oauth_token": lambda k: False,
        "build_anthropic_kwargs": MagicMock(return_value={}),
    }}):
        result = maybe_wrap(
            plain_client, "model-name", "some-key",
            "https://opaque.internal/v1",  # URL alone wouldn't trigger
            api_mode="anthropic_messages",
        )
    assert isinstance(result, AnthropicAuxiliaryClient)


def test_maybe_wrap_anthropic_double_wrap_safe():
    """Already-wrapped AnthropicAuxiliaryClient passes through unchanged."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient

    already_wrapped = MagicMock(spec=AnthropicAuxiliaryClient)
    result = maybe_wrap(
        already_wrapped, "model", "key",
        "https://api.kimi.com/coding", api_mode=None,
    )
    assert result is already_wrapped


def test_maybe_wrap_anthropic_codex_client_passes_through():
    """CodexAuxiliaryClient is never re-dispatched."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient
    from agent.auxiliary_client import CodexAuxiliaryClient

    codex_client = MagicMock(spec=CodexAuxiliaryClient)
    result = maybe_wrap(
        codex_client, "model", "key",
        "https://api.kimi.com/coding", api_mode=None,
    )
    assert result is codex_client
    assert not isinstance(result, AnthropicAuxiliaryClient)


def test_maybe_wrap_anthropic_sdk_missing_falls_back():
    """ImportError on anthropic SDK returns plain client with warning."""
    maybe_wrap = _get_anthropic_service("maybe_wrap_anthropic")
    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient
    from agent.plugin_registries import registries

    plain_client = MagicMock(name="plain_openai")

    def _raise_import(*args, **kwargs):
        raise ImportError("no anthropic SDK")

    # Mock at the registry boundary — simulate SDK missing by making
    # build_anthropic_client raise ImportError when called.
    with patch.dict(registries._provider_services, {
        "anthropic": {**registries._provider_services.get("anthropic", {}),
                      "build_anthropic_client": _raise_import}
    }):
        result = maybe_wrap(
            plain_client, "kimi-for-coding", "sk-kimi-test",
            "https://api.kimi.com/coding", api_mode=None,
        )

    assert result is plain_client
    assert not isinstance(result, AnthropicAuxiliaryClient)


# ---------------------------------------------------------------------------
# Integration: resolve_provider_client for named kimi-coding provider
# ---------------------------------------------------------------------------

def test_resolve_provider_client_kimi_coding_wraps_anthropic(monkeypatch, tmp_path):
    """End-to-end: resolve_provider_client('kimi-coding', 'kimi-for-coding')
    must return AnthropicAuxiliaryClient because /coding speaks Anthropic.

    This is the primary regression guard: the bug that caused title
    generation 404s on every Kimi Coding Plan user after the "main model
    for every user" aux design shipped.
    """
    from agent.auxiliary_client import resolve_provider_client
    from agent.plugin_registries import registries

    AnthropicAuxiliaryClient = _CoreAnthropicAuxiliaryClient

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # sk-kimi- prefix triggers /coding endpoint auto-detection
    monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test123")

    mock_client = MagicMock()
    with patch.dict(registries._provider_services, {"anthropic": {
        **registries._provider_services.get("anthropic", {}),
        "build_anthropic_client": MagicMock(return_value=mock_client),
        "resolve_anthropic_token": MagicMock(return_value="sk-test"),
        "_is_oauth_token": lambda k: False,
        "build_anthropic_kwargs": MagicMock(return_value={}),
    }}):
        client, model = resolve_provider_client("kimi-coding", "kimi-for-coding")

    assert client is not None, "Should resolve a client"
    assert isinstance(client, AnthropicAuxiliaryClient), (
        "Kimi Coding Plan endpoint (api.kimi.com/coding) speaks Anthropic "
        "Messages — aux client MUST be AnthropicAuxiliaryClient, got "
        f"{type(client).__name__}"
    )
    assert "kimi.com/coding" in str(client.base_url)
