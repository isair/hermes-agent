"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import tomllib


def _load_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def _load_package_data():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        tool = tomllib.load(handle)["tool"]
    return tool["setuptools"]["package-data"]


def test_matrix_extra_not_in_all():
    """The [matrix] extra pulls `mautrix[encryption]` -> `python-olm`,
    which has Linux-only wheels and no native build path on Windows or
    modern macOS (archived libolm, C++ errors with Clang 21+).

    With matrix in [all], `uv sync --locked` on Windows tried to build
    python-olm from sdist and failed on `make`. The [matrix] extra is
    excluded from [all] — users opt in via `pip install hermes-agent[matrix]`.
    """
    optional_dependencies = _load_optional_dependencies()

    assert "matrix" in optional_dependencies, (
        "[matrix] extra must still exist for explicit `pip install hermes-agent[matrix]`"
    )
    matrix_in_all = [
        dep for dep in optional_dependencies["all"]
        if "matrix" in dep
    ]
    assert not matrix_in_all, (
        f"matrix must not appear in [all] — it's an opt-in plugin. Found: "
        f"{matrix_in_all}"
    )


def test_plugin_extras_are_workspace_member_refs():
    """Every plugin extra in pyproject.toml should reference a workspace
    member package (e.g. ``hermes-agent-anthropic``), not inline dep specs.

    This ensures the single source of truth for plugin deps is the plugin's
    own pyproject.toml, not the main package's extras.
    """
    optional_dependencies = _load_optional_dependencies()

    # Extras that are known plugin workspace members
    plugin_extras = {
        "anthropic", "bedrock", "azure-identity",
        "discord", "exa", "firecrawl", "parallel",
        "honcho", "hindsight",
        "fal", "tts", "stt",
        "daytona", "modal",
        "telegram", "slack", "dingtalk", "feishu", "matrix",
        "dashboard",
        # "mistral" is intentionally excluded — it's an inline dep spec
        # (mistralai==2.4.8), not a workspace member plugin package.
    }

    for extra in plugin_extras:
        if extra not in optional_dependencies:
            continue
        specs = optional_dependencies[extra]
        for spec in specs:
            assert spec.startswith("hermes-agent-"), (
                f"[{extra}] extra should reference a workspace member, "
                f"not an inline dep spec. Got: {spec}"
            )


def test_dingtalk_extra_includes_qrcode_for_qr_auth():
    """DingTalk's QR-code device-flow auth (hermes_cli/dingtalk_auth.py)
    needs the qrcode package — verify it's in the dingtalk plugin's deps."""
    pyproject_path = Path(__file__).resolve().parents[1] / "plugins" / "platforms" / "dingtalk" / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    deps = project.get("dependencies", [])
    assert any("qrcode" in d for d in deps), (
        f"hermes-agent-dingtalk should depend on qrcode. deps: {deps}"
    )


def test_feishu_extra_includes_qrcode_for_qr_login():
    """Feishu's QR login flow needs the qrcode package — verify it's in
    the feishu plugin's deps."""
    pyproject_path = Path(__file__).resolve().parents[1] / "plugins" / "platforms" / "feishu" / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    deps = project.get("dependencies", [])
    assert any("qrcode" in d for d in deps), (
        f"hermes-agent-feishu should depend on qrcode. deps: {deps}"
    )


def test_dashboard_plugin_manifests_and_assets_are_packaged():
    """Bundled dashboard plugins need their manifests and built assets in
    wheel installs so /api/dashboard/plugins can discover them outside a
    source checkout."""
    package_data = _load_package_data()
    plugin_data = package_data["plugins"]

    assert "*/dashboard/manifest.json" in plugin_data
    assert "*/dashboard/dist/*" in plugin_data
    assert "*/dashboard/dist/**/*" in plugin_data
