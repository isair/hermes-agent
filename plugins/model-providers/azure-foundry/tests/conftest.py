"""Shared fixtures for azure-foundry plugin tests.

Registers the azure plugin in the singleton registry before each test.
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
def _register_azure_plugin():
    """Register the real azure plugin for the duration of each test."""
    from agent.plugin_registries import registries

    _prev_services = dict(registries._provider_services)
    _prev_resolvers = dict(registries._provider_resolvers)
    _prev_cph = dict(registries._credential_pool_hooks)

    ctx = _FullCtx()
    try:
        from hermes_agent_azure import register as _reg
        _reg(ctx)
    except ImportError:
        pass
    # azure-foundry tests for Anthropic Messages mode need the anthropic plugin too
    try:
        from hermes_agent_anthropic import register as _anthro_reg
        _anthro_reg(ctx)
    except ImportError:
        pass

    yield

    for d, prev in [
        (registries._provider_services, _prev_services),
        (registries._provider_resolvers, _prev_resolvers),
        (registries._credential_pool_hooks, _prev_cph),
    ]:
        d.clear()
        d.update(prev)
