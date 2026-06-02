#!/usr/bin/env python3
"""Check that core code and core tests never import from plugin packages.

Core (agent/, hermes_cli/, tools/, run_agent.py, etc.) must interact with
plugins exclusively through the registry layer (``registries.get_provider_service``,
``registries.register_*``).  Direct imports from ``hermes_agent_*`` packages
couple core to plugin internals and break the plugin isolation boundary.

Allowed locations for plugin imports:
  - ``plugins/`` (the plugin packages themselves)
  - ``tests/plugins/`` (plugin-specific tests)
  - ``tests/e2e/`` (end-to-end integration tests that load the full system)
  - ``tests/gateway/`` (gateway integration tests)
  - ``tests/tools/`` (tool integration tests — these test plugin-provided tools)
  - ``hermes_cli/plugins.py`` (the plugin loader itself — it MUST import plugins)

Exit 0 on success, 1 on violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

# Directories where ``from hermes_agent_* import`` / ``import hermes_agent_*``
# is FORBIDDEN.
FORBIDDEN_DIRS: list[Path] = [
    ROOT / "agent",
    ROOT / "hermes_cli",
    ROOT / "tools",
    ROOT / "cron",
    ROOT / "gateway",
    ROOT / "acp_adapter",
    ROOT / "tui_gateway",
    ROOT / "ui-tui",       # Ink/TUI frontend
    ROOT / "batch_runner.py",
    ROOT / "run_agent.py",
    ROOT / "model_tools.py",
    ROOT / "cli.py",
    ROOT / "toolsets.py",
]

# Directories where plugin imports are ALLOWED (no check).
ALLOWED_DIRS: list[Path] = [
    ROOT / "plugins",
    ROOT / "tests" / "plugins",
    ROOT / "tests" / "e2e",
    ROOT / "tests" / "gateway",
    ROOT / "tests" / "tools",
]

# Specific files where plugin imports are allowed even inside a forbidden dir.
ALLOWED_FILES: set[str] = {
    # The plugin loader itself must import plugin packages.
    "hermes_cli/plugins.py",
    # Tests that register real plugin hooks via the registry (correct pattern —
    # they import the function solely to inject it into the registry, not to
    # call it directly).
    "tests/agent/test_credential_pool.py",
}

# Regex matching a plugin import line.
PLUGIN_IMPORT_RE = re.compile(
    r'^\s*(?:from|import)\s+hermes_agent_\w+',
    re.MULTILINE,
)

# ── Implementation ──────────────────────────────────────────────────────────

def _is_in_allowed_dir(path: Path) -> bool:
    """Return True if *path* is inside an allowed directory."""
    for allowed in ALLOWED_DIRS:
        try:
            path.relative_to(allowed)
            return True
        except ValueError:
            pass
    return False


def _is_allowed_file(path: Path) -> bool:
    """Return True if *path* is an explicitly allowed file."""
    rel = str(path.relative_to(ROOT))
    return rel in ALLOWED_FILES


def _check_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, line_content) for violating lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    violations: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if PLUGIN_IMPORT_RE.match(line):
            # Allow noqa comments (F401-style) or explicit "noqa: plugin-import"
            if "noqa" in line:
                continue
            violations.append((i, line.strip()))
    return violations


def main() -> int:
    all_violations: list[tuple[Path, int, str]] = []

    # Check individual files at root level
    for path in FORBIDDEN_DIRS:
        if path.is_file() and path.suffix == ".py":
            if _is_allowed_file(path):
                continue
            for lineno, line in _check_file(path):
                all_violations.append((path, lineno, line))

    # Check directories
    for dir_path in FORBIDDEN_DIRS:
        if not dir_path.is_dir():
            continue
        for py_file in dir_path.rglob("*.py"):
            if _is_in_allowed_dir(py_file):
                continue
            if _is_allowed_file(py_file):
                continue
            for lineno, line in _check_file(py_file):
                all_violations.append((py_file, lineno, line))

    # Check tests/agent/ and tests/agent/transports/ specifically
    # (these are core unit tests that must NOT import plugins)
    for test_dir in [
        ROOT / "tests" / "agent",
        ROOT / "tests" / "agent" / "transports",
    ]:
        if not test_dir.is_dir():
            continue
        for py_file in test_dir.rglob("*.py"):
            if _is_in_allowed_dir(py_file):
                continue
            if _is_allowed_file(py_file):
                continue
            for lineno, line in _check_file(py_file):
                all_violations.append((py_file, lineno, line))

    if not all_violations:
        print("✓ No plugin imports found in core code or core tests")
        return 0

    print("✗ Plugin imports found in core code or core tests:\n")
    for path, lineno, line in sorted(all_violations):
        rel = path.relative_to(ROOT)
        print(f"  {rel}:{lineno}: {line}")
    print(
        f"\n{len(all_violations)} violation(s). Core must interact with plugins "
        "through the registry layer, not direct imports."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
