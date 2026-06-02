"""Custom PEP 517 build backend for hermes-agent.

At wheel build time, rewrites [project.optional-dependencies] so that
plugin extras (e.g. ``anthropic = ["hermes-agent-anthropic"]``) are
inlined with the actual deps from each plugin's pyproject.toml.

In the source repo (and on Nix), uv resolves workspace members natively
so this backend is NOT used — it's only invoked when building a wheel
for PyPI publication.

Usage in pyproject.toml::

    [build-system]
    requires = ["setuptools>=61.0"]
    build-backend = "_build_backend"
    backend-path = ["."]

How it works:
1.  ``build_wheel`` intercepts the call before setuptools sees pyproject.toml.
2.  It reads the workspace member dirs from [tool.uv.workspace].members.
3.  For each member, it reads the member's pyproject.toml and extracts
    ``project.dependencies`` (excluding the ``hermes-agent`` base dep).
4.  It rewrites the main pyproject.toml's optional-dependencies to inline
    those deps instead of the workspace member references.
5.  It writes a temporary pyproject.toml, delegates to
    ``setuptools.build_meta.build_wheel``, then restores the original.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import tomllib

# The original setuptools backend we delegate to.
_BACKEND = "setuptools.build_meta"


def _load_pyproject(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _save_pyproject(path: Path, data: dict) -> None:
    """Write a pyproject.toml. Uses a simple serializer since we only
    need to preserve the structure enough for setuptools to parse."""
    import tomli_w
    with path.open("wb") as f:
        tomli_w.dump(data, f)


def _inline_plugin_deps(root: Path, data: dict) -> dict:
    """Rewrite optional-dependencies to inline plugin deps.

    Maps each plugin extra (e.g. ``anthropic = ["hermes-agent-anthropic"]``)
    to the actual deps from that plugin's pyproject.toml, minus the
    ``hermes-agent`` base dependency.
    """
    opt_deps = data.get("project", {}).get("optional-dependencies", {})
    workspace = data.get("tool", {}).get("uv", {}).get("workspace", {})
    members = workspace.get("members", [])

    # Build a map: package name → (member_dir, pyproject_data)
    pkg_to_deps: dict[str, list[str]] = {}
    for member_glob in members:
        for member_dir in sorted(root.glob(member_glob)):
            pptoml = member_dir / "pyproject.toml"
            if not pptoml.exists():
                continue
            member_data = _load_pyproject(pptoml)
            pkg_name = member_data.get("project", {}).get("name", "")
            if not pkg_name:
                continue
            # Extract deps, excluding the base hermes-agent dependency
            raw_deps = member_data.get("project", {}).get("dependencies", [])
            filtered = [
                d for d in raw_deps
                if not d.replace(" ", "").startswith("hermes-agent")
            ]
            pkg_to_deps[pkg_name] = filtered

    # Rewrite optional-dependencies
    new_opt_deps = {}
    for extra_name, specs in opt_deps.items():
        new_specs = []
        for spec in specs:
            # Check if this spec references a workspace member package
            if spec in pkg_to_deps:
                # Inline the plugin's deps
                new_specs.extend(pkg_to_deps[spec])
            else:
                new_specs.append(spec)
        new_opt_deps[extra_name] = new_specs

    data["project"]["optional-dependencies"] = new_opt_deps

    # Remove [tool.uv] section — it's not valid in a published wheel
    if "uv" in data.get("tool", {}):
        del data["tool"]["uv"]

    return data


# ---------------------------------------------------------------------------
# PEP 517 hooks
# ---------------------------------------------------------------------------

def build_wheel(wheel_directory: str, config_settings: dict[str, Any] | None = None, metadata_directory: str | None = None) -> str:
    """Build a wheel with inlined plugin deps."""
    root = Path.cwd()
    pyproject_path = root / "pyproject.toml"

    # Read and rewrite
    data = _load_pyproject(pyproject_path)
    data = _inline_plugin_deps(root, data)

    # Write a temporary pyproject.toml, build, then restore
    backup = pyproject_path.with_suffix(".toml.bak")
    shutil.copy2(pyproject_path, backup)
    try:
        _save_pyproject(pyproject_path, data)

        # Delegate to setuptools
        import importlib
        backend = importlib.import_module(_BACKEND)
        return backend.build_wheel(wheel_directory, config_settings)
    finally:
        shutil.copy2(backup, pyproject_path)
        backup.unlink()


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    """Build an sdist — no rewriting needed."""
    import importlib
    backend = importlib.import_module(_BACKEND)
    return backend.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return ["setuptools>=61.0", "tomli_w"]


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    return ["setuptools>=61.0"]


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    """Prepare metadata with inlined plugin deps."""
    root = Path.cwd()
    pyproject_path = root / "pyproject.toml"

    data = _load_pyproject(pyproject_path)
    data = _inline_plugin_deps(root, data)

    backup = pyproject_path.with_suffix(".toml.bak")
    shutil.copy2(pyproject_path, backup)
    try:
        _save_pyproject(pyproject_path, data)

        import importlib
        backend = importlib.import_module(_BACKEND)
        return backend.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    finally:
        shutil.copy2(backup, pyproject_path)
        backup.unlink()


def build_editable(wheel_directory: str, config_settings: dict[str, Any] | None = None, metadata_directory: str | None = None) -> str:
    """Build an editable install — no rewriting needed (dev mode)."""
    import importlib
    backend = importlib.import_module(_BACKEND)
    kwargs: dict[str, Any] = {"config_settings": config_settings}
    if metadata_directory is not None:
        kwargs["metadata_directory"] = metadata_directory
    return backend.build_editable(wheel_directory, **kwargs)


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    return ["setuptools>=61.0"]
