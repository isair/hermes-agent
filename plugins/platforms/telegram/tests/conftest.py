"""Shared fixtures for telegram plugin tests.

Provides the ``_ensure_telegram_mock`` helper that guarantees a minimal mock
of the ``telegram`` package is registered in :data:`sys.modules` **before**
any test file triggers ``from hermes_agent_telegram import ...``.
"""

import sys
from unittest.mock import MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    # Prevent pytest from interpreting auto-generated mock attributes as
    # plugin specs.  Without this, ``mod.pytest_plugins`` returns a child
    # MagicMock which trips _get_plugin_specs_as_list().
    mod.pytest_plugins = None
    # Suppress xunit-style setup/teardown names so pytest's
    # _get_first_non_fixture_func() never picks up auto-generated child
    # mocks as real fixtures.  Without this, Package.setup() finds a
    # MagicMock setUpModule, tries to inspect .__code__, and blows up.
    for _xunit_name in (
        "setUpModule", "setup_module",
        "tearDownModule", "teardown_module",
        "setUpClass", "setup_class",
        "tearDownClass", "teardown_class",
        "setUpFunction", "setup_function",
        "tearDownFunction", "teardown_function",
        "setUpMethod", "setup_method",
        "tearDownMethod", "teardown_method",
    ):
        try:
            delattr(mod, _xunit_name)
        except AttributeError:
            pass
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


# Auto-apply at collection time so every test file sees the mock.
_ensure_telegram_mock()
