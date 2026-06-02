"""Anthropic credential pool hook.

Handles provider-specific pool operations: syncing from ~/.claude/.credentials.json,
refreshing OAuth tokens, and deciding which sources to include in the pool.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from typing import Any, Optional

logger = logging.getLogger(__name__)


def sync_from_credentials_file(entry: Any) -> Any:
    """Sync a claude_code pool entry from ~/.claude/.credentials.json if tokens differ.

    OAuth refresh tokens are single-use. When something external (e.g.
    Claude Code CLI, or another profile's pool) refreshes the token, it
    writes the new pair to ~/.claude/.credentials.json. The pool entry's
    refresh token becomes stale. This method detects that and syncs.

    Returns the (possibly updated) entry.
    """
    if entry.source != "claude_code":
        return entry
    try:
        from agent.plugin_registries import registries
        read_claude_code_credentials = registries.get_provider_service("anthropic", "read_claude_code_credentials")
        if read_claude_code_credentials is None:
            return entry
        creds = read_claude_code_credentials()
        if not creds:
            return entry
        file_refresh = creds.get("refreshToken", "")
        file_access = creds.get("accessToken", "")
        file_expires = creds.get("expiresAt", 0)
        if file_refresh and file_refresh != entry.refresh_token:
            logger.debug("Pool entry %s: syncing tokens from credentials file (refresh token changed)", entry.id)
            return replace(
                entry,
                access_token=file_access,
                refresh_token=file_refresh,
                expires_at_ms=file_expires,
                last_status=None,
                last_status_at=None,
                last_error_code=None,
            )
    except Exception as exc:
        logger.debug("Failed to sync from credentials file: %s", exc)
    return entry


def refresh_oauth(entry: Any, pool: Any) -> Any:
    """Refresh an anthropic OAuth token and return the updated entry.

    Handles:
    - Standard OAuth refresh via ``refresh_anthropic_oauth_pure``
    - Writing back to ~/.claude/.credentials.json for claude_code entries
    - Retry with synced token from credentials file on refresh failure

    Returns the updated entry, or the original entry on failure.
    """
    from agent.plugin_registries import registries

    refresh_anthropic_oauth_pure = registries.get_provider_service("anthropic", "refresh_anthropic_oauth_pure")
    if refresh_anthropic_oauth_pure is None:
        return entry

    try:
        refreshed = refresh_anthropic_oauth_pure(
            entry.refresh_token,
            use_json=entry.source.endswith("hermes_pkce"),
        )
        updated = replace(
            entry,
            access_token=refreshed["access_token"],
            refresh_token=refreshed["refresh_token"],
            expires_at_ms=refreshed["expires_at_ms"],
        )
        # Keep ~/.claude/.credentials.json in sync
        if entry.source == "claude_code":
            try:
                _write_claude_code_credentials = registries.get_provider_service("anthropic", "_write_claude_code_credentials")
                if _write_claude_code_credentials is not None:
                    _write_claude_code_credentials(
                        refreshed["access_token"],
                        refreshed["refresh_token"],
                        refreshed["expires_at_ms"],
                    )
            except Exception as wexc:
                logger.debug("Failed to write refreshed token to credentials file: %s", wexc)
        return updated
    except Exception as exc:
        logger.debug("Credential refresh failed for anthropic/%s: %s", entry.id, exc)
        # The refresh token may have been consumed by another process.
        # Check if ~/.claude/.credentials.json has a newer token pair.
        if entry.source == "claude_code":
            synced = sync_from_credentials_file(entry)
            if synced.refresh_token != entry.refresh_token:
                logger.debug("Retrying refresh with synced token from credentials file")
                try:
                    refreshed = refresh_anthropic_oauth_pure(
                        synced.refresh_token,
                        use_json=synced.source.endswith("hermes_pkce"),
                    )
                    updated = replace(
                        synced,
                        access_token=refreshed["access_token"],
                        refresh_token=refreshed["refresh_token"],
                        expires_at_ms=refreshed["expires_at_ms"],
                        last_status="OK",
                        last_status_at=None,
                        last_error_code=None,
                    )
                    try:
                        _write_claude_code_credentials = registries.get_provider_service("anthropic", "_write_claude_code_credentials")
                        if _write_claude_code_credentials is not None:
                            _write_claude_code_credentials(
                                refreshed["access_token"],
                                refreshed["refresh_token"],
                                refreshed["expires_at_ms"],
                            )
                    except Exception:
                        pass
                    return updated
                except Exception:
                    pass
        return entry


def needs_refresh(entry: Any) -> bool:
    """Check if an anthropic OAuth entry needs a token refresh."""
    if entry.expires_at_ms is None:
        return False
    return int(entry.expires_at_ms) <= int(time.time() * 1000) + 120_000


def should_include_in_pool(source: str) -> bool:
    """Which anthropic credential sources should be pooled."""
    return source in {"claude_code", "hermes_pkce"}


def source_priority(source: str) -> int:
    """Priority ordering for anthropic credential sources (lower = preferred)."""
    _PRIORITIES = {
        "claude_code": 3,
        "hermes_pkce": 2,
    }
    return _PRIORITIES.get(source, 99)


def discover_credentials(entries: list, provider: str, is_suppressed: Any) -> tuple:
    """Discover external anthropic credentials and upsert into pool entries.

    Returns (changed: bool, active_sources: set).
    """
    from agent.plugin_registries import registries

    changed = False
    active_sources = set()

    # Only auto-discover external credentials (Claude Code, Hermes PKCE)
    # when the user has explicitly configured anthropic as their provider.
    # Without this gate, auxiliary client fallback chains silently read
    # ~/.claude/.credentials.json without user consent.  See PR #4210.
    try:
        from hermes_cli.auth import is_provider_explicitly_configured
        if not is_provider_explicitly_configured("anthropic"):
            return changed, active_sources
    except ImportError:
        pass

    # API-key vs OAuth is a user-visible choice at `hermes setup` ("Claude
    # Pro/Max subscription" vs "Anthropic API key").  The signal that the
    # user picked the API-key path is: ANTHROPIC_API_KEY set in the env,
    # AND no OAuth env vars set — `save_anthropic_api_key()` writes the
    # API key and zeros ANTHROPIC_TOKEN; `save_anthropic_oauth_token()`
    # does the inverse.  When that signal is present we MUST NOT seed
    # autodiscovered OAuth tokens (~/.claude/.credentials.json from the
    # Claude Code CLI, hermes_pkce creds from a previous OAuth login)
    # into the anthropic pool — otherwise rotation on a 401/429 silently
    # flips the session onto an OAuth credential, which forces the Claude
    # Code identity injection, `mcp_` tool-name rewrite, and claude-cli
    # User-Agent header.  Users who explicitly opted into the API-key path
    # are explicitly opting OUT of that masquerade.  Prefer ~/.hermes/.env
    # over os.environ for the same reason `_seed_from_env` does — that's
    # the authoritative file that `hermes setup` writes.
    try:
        from hermes_cli.config import load_env
    except ImportError:
        load_env = None  # type: ignore[assignment]

    _env_file = load_env() if load_env is not None else {}

    def _env_val(key: str) -> str:
        return (_env_file.get(key) or os.environ.get(key) or "").strip()

    anthropic_api_key = _env_val("ANTHROPIC_API_KEY")
    anthropic_oauth_env = (
        _env_val("ANTHROPIC_TOKEN") or _env_val("CLAUDE_CODE_OAUTH_TOKEN")
    )
    api_key_path_explicit = bool(anthropic_api_key and not anthropic_oauth_env)

    if api_key_path_explicit:
        # Prune any stale autodiscovered OAuth entries that may have been
        # seeded into the on-disk pool during a previous OAuth session.
        # Without this, switching OAuth -> API key at setup leaves the
        # OAuth entries dormant in auth.json forever and rotation on a
        # transient 401 could revive them.
        retained = [
            entry for entry in entries
            if entry.source not in {"hermes_pkce", "claude_code"}
        ]
        if len(retained) != len(entries):
            entries[:] = retained
            changed = True
        return changed, active_sources

    read_claude_code_credentials = registries.get_provider_service("anthropic", "read_claude_code_credentials")
    read_hermes_oauth_credentials = registries.get_provider_service("anthropic", "read_hermes_oauth_credentials")
    if read_claude_code_credentials is None or read_hermes_oauth_credentials is None:
        return changed, active_sources

    # Import pool helpers
    try:
        from agent.credential_pool import _upsert_entry, label_from_token, AUTH_TYPE_OAUTH
    except ImportError:
        return changed, active_sources

    for source_name, creds in (
        ("hermes_pkce", read_hermes_oauth_credentials()),
        ("claude_code", read_claude_code_credentials()),
    ):
        if creds and creds.get("accessToken"):
            if is_suppressed(provider, source_name):
                continue
            active_sources.add(source_name)
            changed |= _upsert_entry(
                entries,
                provider,
                source_name,
                {
                    "source": source_name,
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": creds.get("accessToken", ""),
                    "refresh_token": creds.get("refreshToken"),
                    "expires_at_ms": creds.get("expiresAt"),
                    "label": label_from_token(creds.get("accessToken", ""), source_name),
                },
            )
    return changed, active_sources


# Env var scan order for anthropic — prefer OAuth tokens over API keys
ANTHROPIC_ENV_VAR_ORDER = [
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
]


def detect_auth_type(token: str) -> str:
    """Determine auth type for an anthropic token.

    OAuth tokens don't start with 'sk-ant-api'; API keys do.
    """
    from agent.credential_pool import AUTH_TYPE_OAUTH, AUTH_TYPE_API_KEY
    if not token.startswith("sk-ant-api"):
        return AUTH_TYPE_OAUTH
    return AUTH_TYPE_API_KEY
