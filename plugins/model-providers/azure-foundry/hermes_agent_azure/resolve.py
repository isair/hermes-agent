"""Azure Foundry provider resolver for auxiliary client construction.

Handles ALL provider-specific logic for building auxiliary clients:
Entra ID auth, static API key, base URL resolution, api_mode routing
(chat_completions, codex_responses, anthropic_messages).
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

logger = logging.getLogger(__name__)


def _extract_url_query_params(url: str):
    """Extract query params from URL, return (clean_url, default_query dict or None)."""
    parsed = urlparse(url)
    if parsed.query:
        clean = urlunparse(parsed._replace(query=""))
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return clean, params
    return url, None


def _normalize_resolved_model(model: str, provider: str) -> str:
    """Normalize model name for a given provider."""
    return str(model or "").strip()


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
    """Resolve an Azure Foundry auxiliary client via the runtime resolver.

    Mirrors the anthropic/bedrock resolver shape but delegates to
    ``hermes_cli.runtime_provider._resolve_azure_foundry_runtime`` —
    the same resolver the main agent uses — so:

    * ``auth_mode: api_key`` (default) gets the static
      ``AZURE_FOUNDRY_API_KEY`` string.
    * ``auth_mode: entra_id`` gets a callable bearer-token provider
      (``Callable[[], str]`` from the azure identity adapter).
    * Per-model ``api_mode`` auto-routing for GPT-5.x / o-series /
      codex models works.
    * ``model.entra.{tenant_id,client_id,authority,scope}`` config
      fields propagate.
    * Non-default ``model.base_url`` overrides are honored.

    Returns ``(client, model)`` or ``(None, None)`` on failure.
    """
    from openai import OpenAI

    try:
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        from hermes_cli.auth import AuthError
        from hermes_cli.config import load_config
    except ImportError:
        return None, None

    try:
        cfg = load_config()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
    except Exception:
        model_cfg = {}

    try:
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg=model_cfg,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            target_model=model,
        )
    except AuthError as exc:
        logger.debug("Auxiliary azure-foundry: %s", exc)
        return None, None
    except Exception as exc:
        logger.debug("Auxiliary azure-foundry runtime error: %s", exc)
        return None, None

    api_key = runtime.get("api_key")
    base_url = str(runtime.get("base_url", "") or "")
    runtime_api_mode = api_mode or runtime.get("api_mode") or "chat_completions"

    _has_key = bool(api_key) if not callable(api_key) else True
    if not _has_key or not base_url:
        return None, None

    final_model = _normalize_resolved_model(
        model or str(model_cfg.get("default") or ""),
        "azure-foundry",
    )
    if not final_model:
        logger.debug(
            "Auxiliary azure-foundry: no model resolved (model=%r, default=%r)",
            model, model_cfg.get("default"),
        )
        return None, None

    extra: dict[str, Any] = {}
    _clean_base, _dq = _extract_url_query_params(base_url)
    if _dq:
        extra["default_query"] = _dq

    client = OpenAI(api_key=api_key, base_url=_clean_base, **extra)

    if runtime_api_mode == "codex_responses":
        from agent.auxiliary_client import CodexAuxiliaryClient
        return CodexAuxiliaryClient(client, final_model), final_model

    if runtime_api_mode == "anthropic_messages":
        from agent.plugin_registries import registries
        maybe_wrap = registries.get_provider_service("anthropic", "maybe_wrap_anthropic")
        if maybe_wrap is not None:
            return maybe_wrap(
                client, final_model, api_key,
                base_url, runtime_api_mode,
            ), final_model

    return client, final_model
