"""Anthropic provider resolver for auxiliary client construction.

Handles ALL provider-specific logic for building auxiliary clients:
credential resolution (pool, env var, OAuth), client construction,
base URL detection, and transport wrapping.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from utils import base_url_hostname

logger = logging.getLogger(__name__)

_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"

_ANTHROPIC_COMPAT_PROVIDERS = frozenset({"minimax", "minimax-oauth", "minimax-cn"})


# ---------------------------------------------------------------------------
# Endpoint detection helpers
# ---------------------------------------------------------------------------

def endpoint_speaks_anthropic_messages(base_url: str) -> bool:
    """True if the endpoint at ``base_url`` speaks Anthropic Messages protocol.

    Covers:
    - Any URL ending in ``/anthropic``
    - ``api.kimi.com/coding`` (Kimi Coding Plan)
    - ``api.anthropic.com`` (native Anthropic)
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    if normalized.endswith("/anthropic"):
        return True
    hostname = base_url_hostname(normalized)
    if hostname == "api.anthropic.com":
        return True
    if hostname == "api.kimi.com" and "/coding" in normalized:
        return True
    return False


def is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """Detect if an endpoint expects Anthropic-format content blocks."""
    if provider in _ANTHROPIC_COMPAT_PROVIDERS:
        return True
    url_lower = (base_url or "").lower()
    return "/anthropic" in url_lower


def convert_openai_images_to_anthropic(messages: list) -> list:
    """Convert OpenAI ``image_url`` content blocks to Anthropic ``image`` blocks."""
    converted = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue
        new_content = []
        changed = False
        for block in content:
            if block.get("type") == "image_url":
                image_url_val = (block.get("image_url") or {}).get("url", "")
                if image_url_val.startswith("data:"):
                    header, _, b64data = image_url_val.partition(",")
                    media_type = "image/png"
                    if ":" in header and ";" in header:
                        media_type = header.split(":", 1)[1].split(";", 1)[0]
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })
                else:
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url_val,
                        },
                    })
                changed = True
            else:
                new_content.append(block)
        converted.append({**msg, "content": new_content} if changed else msg)
    return converted


# ---------------------------------------------------------------------------
# Transport wrapping
# ---------------------------------------------------------------------------

def _safe_isinstance(obj: Any, maybe_type: Any) -> bool:
    """Return False instead of raising when a patched symbol is not a type."""
    try:
        return isinstance(obj, maybe_type)
    except TypeError:
        return False


def maybe_wrap_anthropic(
    client_obj: Any,
    model: str,
    api_key: str,
    base_url: str,
    api_mode: Optional[str] = None,
) -> Any:
    """Rewrap a plain OpenAI client in ``AnthropicAuxiliaryClient`` when
    the endpoint actually speaks Anthropic Messages.

    Returns ``client_obj`` unchanged when it's already a specialized adapter
    or the endpoint is OpenAI-wire.
    """
    from agent.anthropic_aux import AnthropicAuxiliaryClient

    # Already wrapped — don't double-wrap.
    if _safe_isinstance(client_obj, AnthropicAuxiliaryClient):
        return client_obj

    # Check for other specialized adapters we should never re-dispatch.
    try:
        from agent.auxiliary_client import CodexAuxiliaryClient
        if _safe_isinstance(client_obj, CodexAuxiliaryClient):
            return client_obj
    except ImportError:
        pass
    try:
        from agent.gemini_native_adapter import GeminiNativeClient
        if _safe_isinstance(client_obj, GeminiNativeClient):
            return client_obj
    except ImportError:
        pass
    try:
        from agent.copilot_acp_client import CopilotACPClient
        if _safe_isinstance(client_obj, CopilotACPClient):
            return client_obj
    except ImportError:
        pass

    # Explicit non-anthropic api_mode wins over URL heuristics.
    if api_mode and api_mode != "anthropic_messages":
        return client_obj

    should_wrap = (
        api_mode == "anthropic_messages"
        or endpoint_speaks_anthropic_messages(base_url)
    )
    if not should_wrap:
        return client_obj

    from agent.plugin_registries import registries
    build_anthropic_client = registries.get_provider_service("anthropic", "build_anthropic_client")
    if build_anthropic_client is None:
        logger.warning(
            "Endpoint %s speaks Anthropic Messages but the anthropic SDK is "
            "not installed — falling back to OpenAI-wire (will likely 404).",
            base_url,
        )
        return client_obj

    try:
        real_client = build_anthropic_client(api_key, base_url)
    except Exception as exc:
        logger.warning(
            "Failed to build Anthropic client for %s (%s) — falling back to "
            "OpenAI-wire client.", base_url, exc,
        )
        return client_obj

    logger.debug(
        "Auxiliary transport: wrapping client in AnthropicAuxiliaryClient "
        "(model=%s, base_url=%s, api_mode=%s)",
        model, base_url[:60] if base_url else "", api_mode or "auto-detected",
    )
    return AnthropicAuxiliaryClient(
        real_client, model, api_key, base_url, is_oauth=False,
    )


# ---------------------------------------------------------------------------
# Pool helpers (thin wrappers over core pool functions)
# ---------------------------------------------------------------------------

def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s: %s", provider, exc)
        return False, None
    if not pool or not pool.has_credentials():
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


def _get_aux_model_for_provider(provider_id: str) -> str:
    """Return the cheap auxiliary model for a provider."""
    try:
        from providers import get_provider_profile
        _p = get_provider_profile(provider_id)
        if _p and _p.default_aux_model:
            return _p.default_aux_model
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# The resolver: called by core's resolve_provider_client()
# ---------------------------------------------------------------------------

def resolve_auxiliary_client(
    *,
    model: str | None = None,
    explicit_api_key: str | None = None,
    explicit_base_url: str | None = None,
    async_mode: bool = False,
    is_vision: bool = False,
    main_runtime: dict | None = None,
    api_mode: str | None = None,
) -> tuple[Any, str] | tuple[None, None]:
    """Resolve an auxiliary client for the Anthropic provider.

    Returns ``(client, default_model)`` or ``(None, None)`` if unavailable.
    """
    from agent.plugin_registries import registries
    from agent.anthropic_aux import (
        AnthropicAuxiliaryClient,
        AsyncAnthropicAuxiliaryClient,
    )

    _anthropic = registries.get_provider_namespace("anthropic")
    build_anthropic_client = _anthropic.get("build_anthropic_client")
    resolve_anthropic_token = _anthropic.get("resolve_anthropic_token")
    if build_anthropic_client is None or resolve_anthropic_token is None:
        return None, None

    pool_present, entry = _select_pool_entry("anthropic")
    if pool_present:
        if entry is None:
            return None, None
        token = explicit_api_key or _pool_runtime_api_key(entry)
    else:
        entry = None
        token = explicit_api_key or resolve_anthropic_token()
    if not token:
        return None, None

    # Allow base URL override from config.yaml model.base_url, but only
    # when the configured provider is anthropic.
    base_url = _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL) if pool_present else _ANTHROPIC_DEFAULT_BASE_URL
    if explicit_base_url:
        base_url = explicit_base_url.strip().rstrip("/")
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == "anthropic":
                cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
                if cfg_base_url:
                    base_url = cfg_base_url
    except Exception:
        pass

    _is_oauth_token = _anthropic.get("_is_oauth_token")
    is_oauth = _is_oauth_token(token) if _is_oauth_token else False
    default_model = model or _get_aux_model_for_provider("anthropic") or "claude-haiku-4-5-20251001"
    logger.debug("Auxiliary client: Anthropic native (%s) at %s (oauth=%s)", default_model, base_url, is_oauth)
    try:
        real_client = build_anthropic_client(token, base_url)
    except ImportError:
        return None, None

    client = AnthropicAuxiliaryClient(real_client, default_model, token, base_url, is_oauth=is_oauth)

    if async_mode:
        client = AsyncAnthropicAuxiliaryClient(client)

    return client, default_model
