"""run_agent test conftest — same pattern as tests/agent/conftest.py."""

from unittest.mock import MagicMock, patch
import pytest

import pytest


@pytest.fixture(autouse=True)
def _register_anthropic_transport():
    """Register the real Anthropic transport so get_transport('anthropic_messages') works."""
    from agent.plugin_registries import registries
    from agent.transports import register_transport

    prev = registries._transports.copy()
    try:
        from hermes_agent_anthropic import register as _reg

        class _Ctx:
            def register_transport(self, api_mode, cls):
                registries._transports[api_mode] = cls
            def __getattr__(self, n):
                if n.startswith("register_"):
                    return lambda *a, **kw: None
                raise AttributeError(n)

        _reg(_Ctx())
    except ImportError:
        pass
    yield
    registries._transports.clear()
    registries._transports.update(prev)
