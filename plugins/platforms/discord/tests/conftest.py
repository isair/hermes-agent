"""Shared fixtures for discord plugin tests.

Registers ``hermes_agent_discord`` as a importable package backed by the
local ``adapter.py`` so that tests can ``import hermes_agent_discord.adapter``
without the package being installed in the venv.
"""

import importlib
import sys
import types
from pathlib import Path

_DISCORD_PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _ensure_hermes_agent_discord():
    """Make ``hermes_agent_discord`` importable from the local adapter.py."""
    if "hermes_agent_discord" in sys.modules:
        return

    # Create a package module pointing at the plugin root
    pkg = types.ModuleType("hermes_agent_discord")
    pkg.__path__ = [str(_DISCORD_PLUGIN_DIR)]
    pkg.__package__ = "hermes_agent_discord"
    sys.modules["hermes_agent_discord"] = pkg

    # Make sure the adapter submodule resolves to the local adapter.py
    if "hermes_agent_discord.adapter" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "hermes_agent_discord.adapter",
            str(_DISCORD_PLUGIN_DIR / "adapter.py"),
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["hermes_agent_discord.adapter"] = mod
        spec.loader.exec_module(mod)


_ensure_hermes_agent_discord()
