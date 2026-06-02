"""Shared fixtures for anthropic plugin tests.

Registers the anthropic plugin in the singleton registry before each test
and provides the ``agent`` fixture used by integration tests.
"""

import sys
from pathlib import Path

from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    """Remove sys.path entries that would shadow the real ``anthropic`` SDK.

    pytest adds ``plugins/model-providers/`` to ``sys.path`` because
    ``plugins/model-providers/anthropic/__init__.py`` (a provider profile)
    exists.  This makes ``import anthropic`` find the plugin directory
    instead of the installed SDK package, causing ``AttributeError:
    module 'anthropic' has no attribute 'Anthropic'``.

    We remove the conflicting entry, evict any wrong cached import, and
    force-import the real SDK so sys.modules["anthropic"] is correct even
    after pytest re-adds the conflicting path during collection.
    """
    import importlib
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent  # main/
    _bad = str(_repo_root / "plugins" / "model-providers")
    while _bad in sys.path:
        sys.path.remove(_bad)
    # Evict wrong import
    if "anthropic" in sys.modules and not hasattr(sys.modules["anthropic"], "Anthropic"):
        del sys.modules["anthropic"]
    # Force-import the real SDK now (before pytest re-adds the bad path)
    # so sys.modules["anthropic"] points to the real package.
    try:
        import anthropic as _real_anthropic  # noqa: F401
        if not hasattr(_real_anthropic, "Anthropic"):
            raise ImportError("wrong anthropic module loaded")
    except ImportError:
        # Try explicit import from venv
        import importlib.util as _ilu
        for _p in sys.path:
            _candidate = Path(_p) / "anthropic" / "__init__.py"
            if _candidate.exists() and (_candidate.parent / "_client.py").exists():
                _spec = _ilu.spec_from_file_location("anthropic", _candidate)
                if _spec and _spec.loader:
                    _mod = _ilu.module_from_spec(_spec)
                    sys.modules["anthropic"] = _mod
                    _spec.loader.exec_module(_mod)
                    break



class _FullCtx:
    """Plugin context that wires up all registry hooks the anthropic plugin uses.

    Uses the real registries for provider_services, provider_resolver,
    credential_pool_hook, transport, and pricing so plugin internals work
    correctly.  Everything else is a no-op so the fixture doesn't depend on
    parts of the system (platform, TTS, etc.) that aren't under test.
    """

    def register_provider_services(self, name, services):
        from agent.plugin_registries import registries
        registries.register_provider_services(name, services)

    def register_provider_resolver(self, name, resolver):
        from agent.plugin_registries import registries
        registries.register_provider_resolver(name, resolver)

    def register_credential_pool_hook(self, name, hook):
        from agent.plugin_registries import registries
        registries.register_credential_pool_hook(name, hook)

    def register_transport(self, api_mode, transport_cls):
        from agent.plugin_registries import registries
        registries._transports[api_mode] = transport_cls

    def register_pricing_provider(self, name, fn):
        from agent.plugin_registries import registries
        registries.register_pricing_provider(name, fn)

    def register_provider_overlay(self, entry):
        from agent.plugin_registries import registries
        registries.register_provider_overlay(entry)

    # Catch-all no-op for every other register_* method (platform, TTS,
    # tools, hooks, skills, etc.) so the fixture never crashes when the
    # plugin calls something we don't need to wire up for unit tests.
    def __getattr__(self, name):
        if name.startswith("register_"):
            return lambda *a, **kw: None
        raise AttributeError(name)


@pytest.fixture(autouse=True)
def _register_anthropic_plugin():
    """Register the real anthropic plugin for the duration of each test,
    then restore the registry to its prior state afterwards.

    Calls the plugin's ``register()`` against a full context so that all
    registry hooks (services, resolver, transport, pricing, etc.) are
    populated.  patch.dict on each affected registry dict guarantees clean
    teardown even across conftest scopes.
    """
    from agent.plugin_registries import registries

    # Snapshot current state so we can restore after the test.
    _prev_services = dict(registries._provider_services)
    _prev_resolvers = dict(registries._provider_resolvers)
    _prev_cph = dict(registries._credential_pool_hooks)
    _prev_transports = dict(registries._transports) if hasattr(registries, "_transports") else {}
    _prev_pricing = dict(registries._pricing_providers) if hasattr(registries, "_pricing_providers") else {}
    _prev_overlays = dict(registries._provider_overlays) if hasattr(registries, "_provider_overlays") else {}

    ctx = _FullCtx()
    try:
        from hermes_agent_anthropic import register as _reg  # type: ignore[import]
        _reg(ctx)
    except ImportError:
        pass

    yield

    # Restore — remove keys the plugin added, put back what was there before.
    for d, prev in [
        (registries._provider_services, _prev_services),
        (registries._provider_resolvers, _prev_resolvers),
        (registries._credential_pool_hooks, _prev_cph),
    ]:
        d.clear()
        d.update(prev)
    for attr, prev in [
        ("_transports", _prev_transports),
        ("_pricing_providers", _prev_pricing),
        ("_provider_overlays", _prev_overlays),
    ]:
        if hasattr(registries, attr):
            getattr(registries, attr).clear()
            getattr(registries, attr).update(prev)


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    from run_agent import AIAgent
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a
