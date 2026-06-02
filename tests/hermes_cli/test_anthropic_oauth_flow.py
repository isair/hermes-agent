"""Tests for Anthropic OAuth setup flow behavior."""

import pytest
from hermes_cli.config import load_env, save_env_value


def _register_anthropic_mocks(monkeypatch, *, run_oauth_setup_token=None, read_claude_code_credentials=None, is_claude_code_token_valid=None):
    """Temporarily inject mock callables into the anthropic provider registry namespace."""
    from agent.plugin_registries import registries

    ns = registries._provider_services.setdefault("anthropic", {})
    updates = {
        "run_oauth_setup_token": run_oauth_setup_token,
        "read_claude_code_credentials": read_claude_code_credentials,
        "is_claude_code_token_valid": is_claude_code_token_valid,
    }
    for k, v in updates.items():
        if v is not None:
            monkeypatch.setitem(ns, k, v)


def test_run_anthropic_oauth_flow_prefers_claude_code_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _register_anthropic_mocks(
        monkeypatch,
        run_oauth_setup_token=lambda: "sk-ant...etup",
        read_claude_code_credentials=lambda: {
            "accessToken": "cc-access-token",
            "refreshToken": "cc-refresh-token",
            "expiresAt": 9999999999999,
        },
        is_claude_code_token_valid=lambda creds: True,
    )

    from hermes_cli.main import _run_anthropic_oauth_flow

    save_env_value("ANTHROPIC_TOKEN", "stale-env-token")
    assert _run_anthropic_oauth_flow(save_env_value) is True

    env_vars = load_env()
    assert env_vars["ANTHROPIC_TOKEN"] == ""
    assert env_vars["ANTHROPIC_API_KEY"] == ""
    output = capsys.readouterr().out
    assert "Claude Code credentials linked" in output


def test_run_anthropic_oauth_flow_manual_token_still_persists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _register_anthropic_mocks(
        monkeypatch,
        run_oauth_setup_token=lambda: None,
        read_claude_code_credentials=lambda: None,
        is_claude_code_token_valid=lambda creds: False,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "sk-ant...oken")
    monkeypatch.setattr(
        "hermes_cli.secret_prompt.masked_secret_prompt",
        lambda _prompt="": "sk-ant...oken",
    )

    from hermes_cli.main import _run_anthropic_oauth_flow

    assert _run_anthropic_oauth_flow(save_env_value) is True

    env_vars = load_env()
    assert env_vars["ANTHROPIC_TOKEN"] == "sk-ant...oken"
    output = capsys.readouterr().out
    assert "Setup-token saved" in output
