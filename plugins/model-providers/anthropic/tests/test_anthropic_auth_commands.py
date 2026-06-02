"""Anthropic-specific auth command tests moved from tests/hermes_cli/test_auth_commands.py."""

from __future__ import annotations

import base64
import json

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _jwt_with_email(email: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"email": email}).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_auth_add_anthropic_oauth_persists_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    token = _jwt_with_email("claude@example.com")
    monkeypatch.setattr(
        "hermes_agent_anthropic.adapter.run_hermes_oauth_login_pure",
        lambda: {
            "access_token": token,
            "refresh_token": "refresh-token",
            "expires_at_ms": 1711234567000,
        },
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "anthropic"
        auth_type = "oauth"
        api_key = None
        label = None

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["anthropic"]
    entry = next(item for item in entries if item["source"] == "manual:hermes_pkce")
    assert entry["label"] == "claude@example.com"
    assert entry["source"] == "manual:hermes_pkce"
    assert entry["refresh_token"] == "refresh-token"
    assert entry["expires_at_ms"] == 1711234567000


def test_seed_from_singletons_respects_hermes_pkce_suppression(tmp_path, monkeypatch):
    """anthropic hermes_pkce must not re-seed from ~/.hermes/.anthropic_oauth.json when suppressed."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import yaml
    (hermes_home / "config.yaml").write_text(yaml.dump({"model": {"provider": "anthropic", "model": "claude"}}))
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"anthropic": ["hermes_pkce"]},
    }))

    # Stub the readers so only hermes_pkce is "available"; claude_code returns None
    import hermes_agent_anthropic as aa
    monkeypatch.setattr(aa, "read_hermes_oauth_credentials", lambda: {
        "accessToken": "tok", "refreshToken": "r", "expiresAt": 9999999999000,
    })
    monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)

    from agent.credential_pool import _seed_from_singletons
    entries = []
    changed, active = _seed_from_singletons("anthropic", entries)
    # hermes_pkce suppressed, claude_code returns None → nothing should be seeded
    assert entries == []
    assert "hermes_pkce" not in active
