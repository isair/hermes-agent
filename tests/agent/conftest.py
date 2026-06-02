"""Agent test conftest — pre-populates the registry with safe mock stubs.

Unit tests in tests/agent/ import core modules directly without going through
the normal startup sequence (PluginManager.discover_and_load()).  Any code path
that calls registries.get_provider_service("anthropic", ...) would return None
and either crash or silently degrade.

This conftest installs a minimal mock anthropic namespace in the registry before
each test, so that:
  - resolve_auxiliary_client(), maybe_wrap_anthropic(), etc. don't crash
  - Tests that want to verify specific behaviour can override individual keys
    with their own patch.dict / mock_anthropic_provider context manager
  - The anthropic SDK never actually needs to be installed in the test env

IMPORTANT: Core tests must NEVER import from hermes_agent_* plugin packages.
All plugin behaviour is simulated through the registry mock namespace.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

__all__ = ["mock_anthropic_provider"]


def _mock_endpoint_speaks_anthropic_messages(base_url: str) -> bool:
    """Functional mock — detects Anthropic-wire endpoints by URL pattern.

    Reproduces the real plugin's logic without importing it.
    """
    if not base_url:
        return False
    normalized = base_url.lower().rstrip("/")
    if normalized.endswith("/anthropic"):
        return True
    # api.anthropic.com
    if "api.anthropic.com" in normalized:
        return True
    # kimi coding plan
    if "api.kimi.com" in normalized and "/coding" in normalized:
        return True
    return False


def _mock_is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """Functional mock — detects Anthropic-compat endpoints.

    Reproduces the real plugin's logic: named compat providers OR /anthropic URL suffix.
    """
    _COMPAT_PROVIDERS = frozenset({"minimax", "minimax-oauth", "minimax-cn"})
    if provider in _COMPAT_PROVIDERS:
        return True
    url_lower = (base_url or "").lower()
    return "/anthropic" in url_lower


def _mock_convert_openai_images_to_anthropic(messages: list) -> list:
    """Functional mock — converts OpenAI image_url blocks to Anthropic image blocks."""
    converted = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue
        new_content = []
        changed = False
        for block in content:
            if block.get("type") == "image_url":
                image_url_val = (block.get("image_url") or {}).get("url", "")
                if image_url_val.startswith("data:"):
                    header, _, b64data = image_url_val.partition(",")
                    media_type = "image/png"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url_val,
                        },
                    })
                changed = True
            else:
                new_content.append(block)
        converted.append({**msg, "content": new_content} if changed else msg)
    return converted


def _mock_maybe_wrap_anthropic(client_obj, model, api_key, base_url, api_mode=None):
    """Functional mock for maybe_wrap_anthropic — wraps when endpoint is Anthropic-wire.

    Reproduces the real plugin's wrapping logic without importing it.
    Uses the real AnthropicAuxiliaryClient from core (no SDK dependency).
    """
    # Already wrapped — don't double-wrap
    from agent.anthropic_aux import (AnthropicAuxiliaryClient,
                                      AsyncAnthropicAuxiliaryClient)
    if isinstance(client_obj, (AnthropicAuxiliaryClient, AsyncAnthropicAuxiliaryClient)):
        return client_obj

    # Check for other specialized adapters we should never re-dispatch
    try:
        from agent.auxiliary_client import CodexAuxiliaryClient
        if isinstance(client_obj, CodexAuxiliaryClient):
            return client_obj
    except ImportError:
        pass

    # Explicit non-anthropic api_mode wins over URL heuristics
    if api_mode and api_mode != "anthropic_messages":
        return client_obj

    should_wrap = (
        api_mode == "anthropic_messages"
        or _mock_endpoint_speaks_anthropic_messages(base_url)
    )
    if not should_wrap:
        return client_obj

    # Use the registry's build_anthropic_client to construct a real(ish) client
    from agent.plugin_registries import registries
    build_fn = registries.get_provider_service("anthropic", "build_anthropic_client")
    if build_fn is None:
        return client_obj

    try:
        real_client = build_fn(api_key, base_url)
    except Exception:
        return client_obj

    return AnthropicAuxiliaryClient(
        real_client, model, api_key, base_url, is_oauth=False,
    )


def _make_base_anthropic_namespace() -> dict:
    """Build a minimal anthropic service namespace with safe mock stubs.

    Wire-format code (build_anthropic_kwargs, convert_messages_to_anthropic,
    AnthropicAuxiliaryClient, etc.) has moved to core modules and is no
    longer looked up via the registry.  Only SDK-dependent orchestration
    (maybe_wrap_anthropic, is_anthropic_compat_endpoint, client building,
    auth) still needs mock stubs here.
    """
    mock_client = MagicMock(name="anthropic_client")
    mock_client.base_url = "https://api.anthropic.com/v1"
    mock_client.api_key = "sk-ant-mock"

    def _resolve_token():
        """Return token from env vars if set — mimics the real resolve_anthropic_token."""
        import os
        return (os.environ.get("ANTHROPIC_TOKEN")
                or os.environ.get("ANTHROPIC_API_KEY"))

    return {
        # SDK-dependent client building
        "build_anthropic_client": MagicMock(return_value=mock_client),
        "build_anthropic_bedrock_client": MagicMock(return_value=mock_client),
        "resolve_anthropic_token": _resolve_token,
        "_is_oauth_token": lambda k: bool(k) and not (k or "").startswith("sk-ant-api"),
        "is_claude_code_token_valid": MagicMock(return_value=False),
        "read_claude_code_credentials": MagicMock(return_value=None),
        "write_claude_code_credentials": MagicMock(),
        "refresh_oauth_token": MagicMock(return_value=None),
        "run_hermes_oauth_login_pure": MagicMock(return_value=("mock-token", None)),
        "_HERMES_OAUTH_FILE": MagicMock(),
        # Resolve / endpoint detection (still plugin-provided, still needs mocking)
        "maybe_wrap_anthropic": _mock_maybe_wrap_anthropic,
        "endpoint_speaks_anthropic_messages": _mock_endpoint_speaks_anthropic_messages,
        "is_anthropic_compat_endpoint": _mock_is_anthropic_compat_endpoint,
        "convert_openai_images_to_anthropic": _mock_convert_openai_images_to_anthropic,
        "ANTHROPIC_DEFAULT_BASE_URL": "https://api.anthropic.com",
        "_ANTHROPIC_COMPAT_PROVIDERS": frozenset(),
        "resolve_auxiliary_client": MagicMock(return_value=(mock_client, "claude-3-5-sonnet-20241022")),
    }


@contextmanager
def mock_anthropic_provider(**overrides):
    """Patch the anthropic registry namespace. Use in core tests instead of
    patching hermes_agent_anthropic.* directly.

    Usage:
        with mock_anthropic_provider(build_anthropic_client=my_mock):
            result = resolve_provider_client(...)
    """
    from agent.plugin_registries import registries
    base = _make_base_anthropic_namespace()
    base.update(overrides)
    with patch.dict(registries._provider_services, {"anthropic": base}):
        yield base


@pytest.fixture(autouse=True)
def _seed_anthropic_registry():
    """Install mock anthropic namespace before each test, restore after.

    Uses patch.dict so it's guaranteed to restore even when plugin tests
    in other directories (which use the real plugin) run before us in the
    same process. Function-scoped (not session) so it re-seeds after each
    plugin test that overwrites the registry.

    Also clears _provider_resolvers["anthropic"] so a real plugin registration
    that leaked from another test file doesn't affect core unit tests.

    Also blocks _ensure_plugins_discovered() so that code paths that lazily
    trigger plugin loading (e.g. get_plugin_auxiliary_tasks via
    _resolve_task_provider_model) don't overwrite the mock namespace.
    """
    from unittest.mock import patch
    from agent.plugin_registries import registries
    ns = _make_base_anthropic_namespace()
    # Guard registries.register_provider_services so that if discover_and_load()
    # fires during a test (e.g. via get_plugin_auxiliary_tasks in
    # _resolve_task_provider_model), it can't overwrite our mock anthropic
    # namespace.  We only block "anthropic" — other providers / hooks proceed
    # normally so tests like test_context_engine.py still work.
    _orig_register = registries.register_provider_services

    def _guarded_register(name, services):
        if name == "anthropic":
            return  # mock namespace wins — don't let the real plugin clobber it
        return _orig_register(name, services)

    _orig_resolver = registries._provider_resolvers.pop("anthropic", None)
    with patch.dict(registries._provider_services, {"anthropic": ns}), \
         patch.object(registries, "register_provider_services", _guarded_register):
        yield
    # Restore resolver (None means "not registered", which is correct for
    # core unit tests; plugin tests that need the real resolver load it themselves)
    if _orig_resolver is not None:
        registries._provider_resolvers["anthropic"] = _orig_resolver
    else:
        registries._provider_resolvers.pop("anthropic", None)
