"""Shared fixtures for bedrock plugin tests.

Registers the bedrock plugin in the singleton registry before each test.
"""
import pytest


class _FullCtx:
    """Plugin context that wires up all registry hooks."""

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

    def register_pricing_provider(self, name, entries):
        from agent.plugin_registries import registries
        registries.register_pricing_provider(name, entries)

    def register_provider_overlay(self, entry):
        from agent.plugin_registries import registries
        registries.register_provider_overlay(entry)

    def __getattr__(self, name):
        if name.startswith("register_"):
            return lambda *a, **kw: None
        raise AttributeError(name)


@pytest.fixture(autouse=True)
def _register_bedrock_plugin():
    """Register the real bedrock plugin for the duration of each test."""
    from agent.plugin_registries import registries
    from hermes_cli import providers as _prov

    _prev_services = dict(registries._provider_services)
    _prev_resolvers = dict(registries._provider_resolvers)
    _prev_cph = dict(registries._credential_pool_hooks)
    _prev_overlays = dict(registries._provider_overlays)
    _prev_hermes_overlays = dict(_prov.HERMES_OVERLAYS)
    _prev_aliases = dict(_prov.ALIASES)
    _prev_merged = _prov._plugin_overlays_merged

    ctx = _FullCtx()
    try:
        from hermes_agent_bedrock import register as _reg
        _reg(ctx)
    except ImportError:
        pass
    try:
        from hermes_agent_anthropic import register as _ant_reg
        _ant_reg(ctx)
    except ImportError:
        pass

    # Force a re-merge so plugin-registered overlays and aliases
    # appear in HERMES_OVERLAYS / ALIASES for the test.
    _prov._plugin_overlays_merged = False
    _prov._merge_plugin_overlays()

    yield

    for d, prev in [
        (registries._provider_services, _prev_services),
        (registries._provider_resolvers, _prev_resolvers),
        (registries._credential_pool_hooks, _prev_cph),
        (registries._provider_overlays, _prev_overlays),
    ]:
        d.clear()
        d.update(prev)
    _prov.HERMES_OVERLAYS.clear()
    _prov.HERMES_OVERLAYS.update(_prev_hermes_overlays)
    _prov.ALIASES.clear()
    _prov.ALIASES.update(_prev_aliases)
    _prov._plugin_overlays_merged = _prev_merged
