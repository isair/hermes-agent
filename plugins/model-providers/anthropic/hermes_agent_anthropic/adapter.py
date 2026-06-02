"""Anthropic Messages API adapter for Hermes Agent.

Translates between Hermes's internal OpenAI-style message format and
Anthropic's Messages API. Follows the same pattern as the codex_responses
adapter — all provider-specific logic is isolated here.

Auth supports:
  - Regular API keys (sk-ant-api*) → x-api-key header
  - OAuth setup-tokens (sk-ant-oat*) → Bearer auth + beta header
  - Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json) → Bearer auth
"""

import copy
import json
import logging
import os
import platform
import secrets
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_home
from typing import Any, Dict, List, Optional, Tuple
from utils import base_url_host_matches, normalize_proxy_env_vars

# NOTE: `import anthropic` is deliberately NOT at module top — the SDK pulls
# ~220 ms of imports (anthropic.types, anthropic.lib.tools._beta_runner, etc.)
# and the 3 usage sites (build_anthropic_client, build_anthropic_bedrock_client,
# read_claude_code_credentials_from_keychain) are all on cold user-triggered
# paths. Access via the `_get_anthropic_sdk()` accessor below, which caches
# the module after the first call and returns None on ImportError.
_anthropic_sdk: Any = ...  # sentinel — None means "tried and missing"


def _get_anthropic_sdk():
    """Return the ``anthropic`` SDK module, importing lazily. None if not installed."""
    global _anthropic_sdk
    if _anthropic_sdk is ...:
        try:
            import anthropic as _sdk
            _anthropic_sdk = _sdk
        except ImportError:
            _anthropic_sdk = None
    return _anthropic_sdk

logger = logging.getLogger(__name__)

THINKING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}
# Hermes effort → Anthropic adaptive-thinking effort (output_config.effort).
# Anthropic exposes 5 levels on 4.7+: low, medium, high, xhigh, max.
# Opus/Sonnet 4.6 only expose 4 levels: low, medium, high, max — no xhigh.
# We preserve xhigh as xhigh on 4.7+ (the recommended default for coding/
# agentic work) and downgrade it to max on pre-4.7 adaptive models (which
# is the strongest level they accept).  "minimal" is a legacy alias that
# maps to low on every model.  See:
# https://platform.claude.com/docs/en/about-claude/models/migration-guide
ADAPTIVE_EFFORT_MAP = {
    "max":     "max",
    "xhigh":   "xhigh",
    "high":    "high",
    "medium":  "medium",
    "low":     "low",
    "minimal": "low",
}

# Models that accept the "xhigh" output_config.effort level.  Opus 4.7 added
# xhigh as a distinct level between high and max; older adaptive-thinking
# models (4.6) reject it with a 400.  Keep this substring list in sync with
# the Anthropic migration guide as new model families ship.
_XHIGH_EFFORT_SUBSTRINGS = ("4-7", "4.7")

# Models where extended thinking is deprecated/removed (4.6+ behavior: adaptive
# is the only supported mode; 4.7 additionally forbids manual thinking entirely
# and drops temperature/top_p/top_k).
_ADAPTIVE_THINKING_SUBSTRINGS = ("4-6", "4.6", "4-7", "4.7")

# Models where temperature/top_p/top_k return 400 if set to non-default values.
# This is the Opus 4.7 contract; future 4.x+ models are expected to follow it.
_NO_SAMPLING_PARAMS_SUBSTRINGS = ("4-7", "4.7")
_FAST_MODE_SUPPORTED_SUBSTRINGS = ("opus-4-6", "opus-4.6")

# ── Max output token limits per Anthropic model ───────────────────────
# Source: Anthropic docs + Cline model catalog.  Anthropic's API requires
# max_tokens as a mandatory field.  Previously we hardcoded 16384, which
# starves thinking-enabled models (thinking tokens count toward the limit).
_ANTHROPIC_OUTPUT_LIMITS = {
    # Claude 4.7
    "claude-opus-4-7":   128_000,
    # Claude 4.6
    "claude-opus-4-6":   128_000,
    "claude-sonnet-4-6":  64_000,
    # Claude 4.5
    "claude-opus-4-5":    64_000,
    "claude-sonnet-4-5":  64_000,
    "claude-haiku-4-5":   64_000,
    # Claude 4
    "claude-opus-4":      32_000,
    "claude-sonnet-4":    64_000,
    # Claude 3.7
    "claude-3-7-sonnet": 128_000,
    # Claude 3.5
    "claude-3-5-sonnet":   8_192,
    "claude-3-5-haiku":    8_192,
    # Claude 3
    "claude-3-opus":       4_096,
    "claude-3-sonnet":     4_096,
    "claude-3-haiku":      4_096,
    # Third-party Anthropic-compatible providers
    "minimax":            131_072,
    # Qwen models via DashScope Anthropic-compatible endpoint
    # DashScope enforces max_tokens ∈ [1, 65536]
    "qwen3":               65_536,
}

# For any model not in the table, assume the highest current limit.
# Future Anthropic models are unlikely to have *less* output capacity.
_ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000


def _get_anthropic_max_output(model: str) -> int:
    """Look up the max output token limit for an Anthropic model.

    Uses substring matching against _ANTHROPIC_OUTPUT_LIMITS so date-stamped
    model IDs (claude-sonnet-4-5-20250929) and variant suffixes (:1m, :fast)
    resolve correctly.  Longest-prefix match wins to avoid e.g. "claude-3-5"
    matching before "claude-3-5-sonnet".

    Normalizes dots to hyphens so that model names like
    ``anthropic/claude-opus-4.6`` match the ``claude-opus-4-6`` table key.
    """
    m = model.lower().replace(".", "-")
    best_key = ""
    best_val = _ANTHROPIC_DEFAULT_OUTPUT_LIMIT
    for key, val in _ANTHROPIC_OUTPUT_LIMITS.items():
        if key in m and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val


def _resolve_positive_anthropic_max_tokens(value) -> Optional[int]:
    """Return ``value`` floored to a positive int, or ``None`` if it is not a
    finite positive number. Ported from openclaw/openclaw#66664.

    Anthropic's Messages API rejects ``max_tokens`` values that are 0,
    negative, non-integer, or non-finite with HTTP 400. Python's ``or``
    idiom (``max_tokens or fallback``) correctly catches ``0`` but lets
    negative ints and fractional floats (``-1``, ``0.5``) through to the
    API, producing a user-visible failure instead of a local error.
    """
    # Booleans are a subclass of int — exclude explicitly so ``True`` doesn't
    # silently become 1 and ``False`` doesn't become 0.
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        import math
        if not math.isfinite(value):
            return None
    except Exception:
        return None
    floored = int(value)  # truncates toward zero for floats
    return floored if floored > 0 else None


def _resolve_anthropic_messages_max_tokens(
    requested,
    model: str,
    context_length: Optional[int] = None,
) -> int:
    """Resolve the ``max_tokens`` budget for an Anthropic Messages call.

    Prefers ``requested`` when it is a positive finite number; otherwise
    falls back to the model's output ceiling. Raises ``ValueError`` if no
    positive budget can be resolved (should not happen with current model
    table defaults, but guards against a future regression where
    ``_get_anthropic_max_output`` could return ``0``).

    Separately, callers apply a context-window clamp — this resolver does
    not, to keep the positive-value contract independent of endpoint
    specifics.

    Ported from openclaw/openclaw#66664 (resolveAnthropicMessagesMaxTokens).
    """
    resolved = _resolve_positive_anthropic_max_tokens(requested)
    if resolved is not None:
        return resolved
    fallback = _get_anthropic_max_output(model)
    if fallback > 0:
        return fallback
    raise ValueError(
        f"Anthropic Messages adapter requires a positive max_tokens value for "
        f"model {model!r}; got {requested!r} and no model default resolved."
    )


def _supports_adaptive_thinking(model: str) -> bool:
    """Return True for Claude 4.6+ models that support adaptive thinking."""
    return any(v in model for v in _ADAPTIVE_THINKING_SUBSTRINGS)


def _supports_xhigh_effort(model: str) -> bool:
    """Return True for models that accept the 'xhigh' adaptive effort level.

    Opus 4.7 introduced xhigh as a distinct level between high and max.
    Pre-4.7 adaptive models (Opus/Sonnet 4.6) only accept low/medium/high/max
    and reject xhigh with an HTTP 400. Callers should downgrade xhigh→max
    when this returns False.
    """
    return any(v in model for v in _XHIGH_EFFORT_SUBSTRINGS)


def _forbids_sampling_params(model: str) -> bool:
    """Return True for models that 400 on any non-default temperature/top_p/top_k.

    Opus 4.7 explicitly rejects sampling parameters; later Claude releases are
    expected to follow suit.  Callers should omit these fields entirely rather
    than passing zero/default values (the API rejects anything non-null).
    """
    return any(v in model for v in _NO_SAMPLING_PARAMS_SUBSTRINGS)


def _supports_fast_mode(model: str) -> bool:
    """Return True for models that support Anthropic Fast Mode (speed=fast).

    Per Anthropic docs, fast mode is currently supported on Opus 4.6 only.
    Sending ``speed: "fast"`` to any other Claude model (including Opus 4.7)
    returns HTTP 400. This guard prevents silently 400'ing when stale config
    or older callers leave fast mode enabled across a model upgrade.
    """
    return any(v in model for v in _FAST_MODE_SUPPORTED_SUBSTRINGS)


# Beta headers for enhanced features that are safe on ordinary/native Anthropic
# requests. As of Opus 4.7 (2026-04-16), these are GA on Claude 4.6+ — the
# beta headers are still accepted (harmless no-op) but not required. Kept
# here so older Claude (4.5, 4.1) + compatible endpoints that still gate on
# the headers continue to get the enhanced features.
#
# Do NOT include ``context-1m-2025-08-07`` here. Anthropic returns HTTP 400
# ("long context beta is not yet available for this subscription") for
# accounts without the long-context beta, which breaks normal short auxiliary
# calls like title generation/session summarization.
#
# ``context-1m-2025-08-07`` is still required to unlock the 1M context window
# on Claude Opus 4.6/4.7 and Sonnet 4.6 when served via AWS Bedrock or Azure
# AI Foundry. Add it only for those endpoint-specific paths below.
_COMMON_BETAS = [
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]
# MiniMax's Anthropic-compatible endpoints fail tool-use requests when
# the fine-grained tool streaming beta is present.  Omit it so tool calls
# fall back to the provider's default response path.
_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
# 1M context beta. Native Anthropic does not get this by default because some
# subscriptions reject it, but Bedrock/Azure still need it for 1M context.
_CONTEXT_1M_BETA = "context-1m-2025-08-07"

# Fast mode beta — enables the ``speed: "fast"`` request parameter for
# significantly higher output token throughput on Opus 4.6 (~2.5x).
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
_FAST_MODE_BETA = "fast-mode-2026-02-01"

# Additional beta headers required for OAuth/subscription auth.
# Matches what Claude Code (and pi-ai / OpenCode) send.
_OAUTH_ONLY_BETAS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
]

# Claude Code identity — required for OAuth requests to be routed correctly.
# Without these, Anthropic's infrastructure intermittently 500s OAuth traffic.
# The version must stay reasonably current — Anthropic rejects OAuth requests
# when the spoofed user-agent version is too far behind the actual release.
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.74"
_claude_code_version_cache: Optional[str] = None


def _detect_claude_code_version() -> str:
    """Detect the installed Claude Code version, fall back to a static constant.

    Anthropic's OAuth infrastructure validates the user-agent version and may
    reject requests with a version that's too old.  Detecting dynamically means
    users who keep Claude Code updated never hit stale-version 400s.
    """
    import subprocess as _sp

    for cmd in ("claude", "claude-code"):
        try:
            result = _sp.run(
                [cmd, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output is like "2.1.74 (Claude Code)" or just "2.1.74"
                version = result.stdout.strip().split()[0]
                if version and version[0].isdigit():
                    return version
        except Exception:
            pass
    return _CLAUDE_CODE_VERSION_FALLBACK


_CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
_MCP_TOOL_PREFIX = "mcp_"


def _get_claude_code_version() -> str:
    """Lazily detect the installed Claude Code version when OAuth headers need it."""
    global _claude_code_version_cache
    if _claude_code_version_cache is None:
        _claude_code_version_cache = _detect_claude_code_version()
    return _claude_code_version_cache


def _is_oauth_token(key: str) -> bool:
    """Check if the key is an Anthropic OAuth/setup token.

    Positively identifies Anthropic OAuth tokens by their key format:
    - ``sk-ant-`` prefix (but NOT ``sk-ant-api``) → setup tokens, managed keys
    - ``eyJ`` prefix → JWTs from the Anthropic OAuth flow
    - ``cc-`` prefix → Claude Code OAuth access tokens (from CLAUDE_CODE_OAUTH_TOKEN)

    Non-Anthropic keys (MiniMax, Alibaba, etc.) don't match any pattern
    and correctly return False.
    """
    if not key:
        return False
    # Regular Anthropic Console API keys — x-api-key auth, never OAuth
    if key.startswith("sk-ant-api"):
        return False
    # Anthropic-issued tokens (setup-tokens sk-ant-oat-*, managed keys)
    if key.startswith("sk-ant-"):
        return True
    # JWTs from Anthropic OAuth flow
    if key.startswith("eyJ"):
        return True
    # Claude Code OAuth access tokens (opaque, from CLAUDE_CODE_OAUTH_TOKEN)
    if key.startswith("cc-"):
        return True
    return False


def _normalize_base_url_text(base_url) -> str:
    """Normalize SDK/base transport URL values to a plain string for inspection.

    Some client objects expose ``base_url`` as an ``httpx.URL`` instead of a raw
    string.  Provider/auth detection should accept either shape.
    """
    if not base_url:
        return ""
    return str(base_url).strip()


def _is_third_party_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for non-Anthropic endpoints using the Anthropic Messages API.

    Third-party proxies (Microsoft Foundry, AWS Bedrock, self-hosted) authenticate
    with their own API keys via x-api-key, not Anthropic OAuth tokens. OAuth
    detection should be skipped for these endpoints.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False  # No base_url = direct Anthropic API
    normalized = normalized.rstrip("/").lower()
    if "anthropic.com" in normalized:
        return False  # Direct Anthropic API — OAuth applies
    return True  # Any other endpoint is a third-party proxy


def _is_kimi_coding_endpoint(base_url: str | None) -> bool:
    """Return True for Kimi's /coding endpoint that requires claude-code UA."""
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    return normalized.rstrip("/").lower().startswith("https://api.kimi.com/coding")


# Model-name prefixes that identify the Kimi / Moonshot family.  Covers
# - official slugs: ``kimi-k2.5``, ``kimi_thinking``, ``moonshot-v1-8k``
# - common release lines: ``k1.5-...``, ``k2-thinking``, ``k25-...``, ``k2.5-...``
# Matched case-insensitively against the post-``normalize_model_name`` form,
# so a caller's ``provider/vendor/model`` slug is handled the same as a
# bare name.
_KIMI_FAMILY_MODEL_PREFIXES = (
    "kimi-", "kimi_",
    "moonshot-", "moonshot_",
    "k1.", "k1-",
    "k2.", "k2-",
    "k25", "k2.5",
)


def _model_name_is_kimi_family(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    m = model.strip().lower()
    if not m:
        return False
    # Strip vendor prefix (e.g. ``moonshotai/kimi-k2.5`` → ``kimi-k2.5``)
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m.startswith(_KIMI_FAMILY_MODEL_PREFIXES)


def _is_kimi_family_endpoint(base_url: str | None, model: str | None = None) -> bool:
    """Return True for any Kimi / Moonshot Anthropic-Messages-speaking endpoint.

    Broader than ``_is_kimi_coding_endpoint`` — matches:

    - Kimi's official ``/coding`` URL (legacy check, preserved)
    - Any ``api.kimi.com`` / ``moonshot.ai`` / ``moonshot.cn`` host
    - Custom or proxied endpoints whose *model* name is in the Kimi / Moonshot
      family (``kimi-*``, ``moonshot-*``, ``k1.*``, ``k2.*``, …).  Users with
      ``api_mode: anthropic_messages`` on a private gateway fronting Kimi
      fall into this branch — the upstream still enforces Kimi's thinking
      semantics (reasoning_content required on every replayed tool-call
      message) regardless of the gateway's hostname.

    Used to decide whether to drop Anthropic's ``thinking`` kwarg and to
    preserve unsigned reasoning_content-derived thinking blocks on replay.
    See hermes-agent#13848, #17057.
    """
    if _is_kimi_coding_endpoint(base_url):
        return True
    for _domain in ("api.kimi.com", "moonshot.ai", "moonshot.cn"):
        if base_url_host_matches(base_url or "", _domain):
            return True
    if _model_name_is_kimi_family(model):
        return True
    return False


def _is_deepseek_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for DeepSeek's Anthropic-compatible endpoint.

    DeepSeek's ``/anthropic`` route speaks the Anthropic Messages protocol
    but, when thinking mode is enabled, requires the ``thinking`` blocks
    from prior assistant turns to round-trip on subsequent requests — the
    generic third-party path strips them and triggers HTTP 400::

        The content[].thinking in the thinking mode must be passed back
        to the API.

    Per DeepSeek's published compatibility matrix the blocks are unsigned
    (no Anthropic-proprietary signature, no ``redacted_thinking`` support),
    so this endpoint is handled with the same strip-signed / keep-unsigned
    policy used for Kimi's ``/coding`` endpoint.  The match is pinned to
    the ``/anthropic`` path so the OpenAI-compatible ``api.deepseek.com``
    base URL (which never reaches this adapter) is not misclassified.
    See hermes-agent#16748.
    """
    if not base_url_host_matches(base_url or "", "api.deepseek.com"):
        return False
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    return "/anthropic" in normalized.rstrip("/").lower()


def _requires_bearer_auth(base_url: str | None) -> bool:
    """Return True for Anthropic-compatible providers that require Bearer auth.

    Some third-party /anthropic endpoints implement Anthropic's Messages API but
    require Authorization: Bearer instead of Anthropic's native x-api-key header.
    MiniMax's global and China Anthropic-compatible endpoints, and Azure AI
    Foundry's Anthropic-style endpoint follow this pattern.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    normalized = normalized.rstrip("/").lower()
    return (
        normalized.startswith(("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic"))
        or "azure.com" in normalized
    )


def _base_url_needs_context_1m_beta(base_url: str | None) -> bool:
    """Return True for endpoints that still gate 1M context behind a beta."""
    normalized = _normalize_base_url_text(base_url).lower()
    if not normalized:
        return False
    return "azure.com" in normalized


def _is_minimax_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for MiniMax's Anthropic-compatible endpoints.

    MiniMax rejects the fine-grained-tool-streaming and context-1m betas;
    those need to be stripped even though MiniMax also uses Bearer auth.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    normalized = normalized.rstrip("/").lower()
    return normalized.startswith(
        ("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic")
    )


def _is_azure_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for Azure-hosted Anthropic Messages endpoints.

    Covers both the modern Foundry host family (``*.services.ai.azure.*``)
    and the legacy Azure OpenAI host family (``*.openai.azure.*``) when
    serving Anthropic's ``/anthropic`` route. Used to opt-in those hosts
    to the ``api-version`` query-param plumbing required by Azure.

    Intentionally avoids a finite allow-list of TLD suffixes so it works
    across sovereign / private Azure clouds.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower().rstrip(".")
    path = (parsed.path or "").lower()
    host_padded = f".{host}."
    is_foundry_host = ".services.ai.azure." in host_padded
    is_legacy_azoai_host = ".openai.azure." in host_padded
    return (is_foundry_host or is_legacy_azoai_host) and "/anthropic" in path


def _common_betas_for_base_url(
    base_url: str | None,
    *,
    drop_context_1m_beta: bool = False,
) -> list[str]:
    """Return the beta headers that are safe for the configured endpoint.

    MiniMax's Anthropic-compatible endpoints (Bearer-auth) reject requests
    that include Anthropic's ``fine-grained-tool-streaming`` beta — every
    tool-use message triggers a connection error. They also reject the
    1M-context beta. Azure AI Foundry's Anthropic endpoint also uses
    Bearer auth but keeps both betas (it needs the 1M beta for 1M context).

    The ``context-1m-2025-08-07`` beta is not sent to native Anthropic by
    default because some subscriptions reject it. Add it only for endpoint
    families that still require it for 1M context, currently Microsoft Foundry.
    Bedrock uses its own client helper below and opts in explicitly.

    ``drop_context_1m_beta=True`` strips the 1M-context beta from any path that
    would otherwise include it after a subscription/endpoint rejects the beta.
    """
    betas = list(_COMMON_BETAS)
    if _base_url_needs_context_1m_beta(base_url) and not drop_context_1m_beta:
        betas.append(_CONTEXT_1M_BETA)
    if _is_minimax_anthropic_endpoint(base_url):
        _stripped = {_TOOL_STREAMING_BETA, _CONTEXT_1M_BETA}
        return [b for b in betas if b not in _stripped]
    if drop_context_1m_beta:
        return [b for b in betas if b != _CONTEXT_1M_BETA]
    return betas


def _build_anthropic_client_with_bearer_hook(
    token_provider,
    base_url: str = None,
    timeout: float = None,
    *,
    drop_context_1m_beta: bool = False,
):
    """Anthropic-on-Foundry Entra ID variant of :func:`build_anthropic_client`.

    Anthropic SDK 0.86.0 stores ``api_key`` / ``auth_token`` as static
    strings; there is no callable-token contract. To get per-request
    bearer refresh (Microsoft's documented Foundry pattern), we hand
    the SDK a custom ``httpx.Client`` whose request event hook mints a
    fresh JWT from the Entra credential chain and rewrites
    ``Authorization: Bearer <jwt>`` on every outbound request. The SDK
    ignores its own auth logic when ``http_client`` is provided (the
    hook strips any pre-set Authorization).

    The placeholder ``auth_token`` is required because the SDK raises
    ``AnthropicError`` at construction if neither ``api_key`` nor
    ``auth_token`` is set — but the hook overrides it per-request so
    the placeholder value never reaches Azure.
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for Azure Foundry Anthropic-style "
            "endpoints with Entra ID auth. Install with: pip install 'anthropic>=0.39.0'"
        )

    normalize_proxy_env_vars()

    from httpx import Timeout
    from hermes_agent_azure import build_bearer_http_client

    _read_timeout = timeout if (isinstance(timeout, (int, float)) and timeout > 0) else 900.0
    timeout_obj = Timeout(timeout=float(_read_timeout), connect=10.0)

    # Strip any trailing /v1 — the Anthropic SDK appends /v1/messages.
    normalized_base_url = _normalize_base_url_text(base_url)
    if normalized_base_url:
        import re as _re
        normalized_base_url = _re.sub(r"/v1/?$", "", normalized_base_url.rstrip("/"))

    http_client = build_bearer_http_client(token_provider, timeout=timeout_obj)

    kwargs = {
        "timeout": timeout_obj,
        "http_client": http_client,
        # The SDK requires *something* for api_key/auth_token. Our
        # event hook overrides Authorization per request so this value
        # is never sent. The sentinel string makes accidental leaks
        # diagnosable in logs.
        "auth_token": "entra-id-bearer-via-http-hook",
    }

    if normalized_base_url:
        if _is_azure_anthropic_endpoint(normalized_base_url) and "api-version" not in normalized_base_url:
            kwargs["base_url"] = normalized_base_url
            kwargs["default_query"] = {"api-version": "2025-04-15"}
        else:
            kwargs["base_url"] = normalized_base_url

    common_betas = _common_betas_for_base_url(
        normalized_base_url,
        drop_context_1m_beta=drop_context_1m_beta,
    )
    if common_betas:
        kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}

    return _anthropic_sdk.Anthropic(**kwargs)


def build_anthropic_client(
    api_key,
    base_url: str = None,
    timeout: float = None,
    *,
    drop_context_1m_beta: bool = False,
):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys.

    ``api_key`` accepts either:

    * a static ``str`` — the historical contract for all key-based and
      OAuth flows.
    * a ``Callable[[], str]`` — an Entra ID bearer token provider from
      :mod:`agent.azure_identity_adapter`. The Anthropic SDK itself
      requires a static string, so when given a callable we construct
      a custom ``httpx.Client`` with a request event hook that mints a
      fresh JWT per outbound request and rewrites the ``Authorization``
      header. The SDK never sees the callable directly.

    If *timeout* is provided it overrides the default 900s read timeout.  The
    connect timeout stays at 10s.  Callers pass this from the per-provider /
    per-model ``request_timeout_seconds`` config so Anthropic-native and
    Anthropic-compatible providers respect the same knob as OpenAI-wire
    providers.

    ``drop_context_1m_beta=True`` strips ``context-1m-2025-08-07`` from the
    client-level ``anthropic-beta`` header. Used by the reactive OAuth retry
    path in ``run_agent.py`` when a subscription rejects the beta; leave at
    its default on fresh clients so 1M-capable subscriptions keep the
    capability.

    Returns an anthropic.Anthropic instance.
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )

    # Callable api_key → Entra ID bearer provider path. Delegated to a
    # helper so the existing static-key code below stays unchanged.
    if callable(api_key) and not isinstance(api_key, str):
        return _build_anthropic_client_with_bearer_hook(
            api_key, base_url, timeout,
            drop_context_1m_beta=drop_context_1m_beta,
        )

    normalize_proxy_env_vars()

    from httpx import Timeout

    normalized_base_url = _normalize_base_url_text(base_url)
    _read_timeout = timeout if (isinstance(timeout, (int, float)) and timeout > 0) else 900.0
    kwargs = {
        "timeout": Timeout(timeout=float(_read_timeout), connect=10.0),
    }
    if normalized_base_url:
        # Azure Anthropic endpoints require an ``api-version`` query parameter.
        # Pass it via default_query so the SDK appends it to every request URL
        # without corrupting the base_url (appending it directly produces
        # malformed paths like /anthropic?api-version=.../v1/messages).
        if _is_azure_anthropic_endpoint(normalized_base_url) and "api-version" not in normalized_base_url:
            kwargs["base_url"] = normalized_base_url.rstrip("/")
            kwargs["default_query"] = {"api-version": "2025-04-15"}
        else:
            kwargs["base_url"] = normalized_base_url
    common_betas = _common_betas_for_base_url(
        normalized_base_url,
        drop_context_1m_beta=drop_context_1m_beta,
    )

    if _is_kimi_coding_endpoint(base_url):
        # Kimi's /coding endpoint requires User-Agent: claude-code/0.1.0
        # to be recognized as a valid Coding Agent. Without it, returns 403.
        # Check this BEFORE _requires_bearer_auth since both match api.kimi.com/coding.
        kwargs["api_key"] = api_key
        kwargs["default_headers"] = {
            "User-Agent": "claude-code/0.1.0",
            **( {"anthropic-beta": ",".join(common_betas)} if common_betas else {} )
        }
    elif _requires_bearer_auth(normalized_base_url):
        # Some Anthropic-compatible providers (e.g. MiniMax) expect the API key in
        # Authorization: Bearer *** for regular API keys. Route those endpoints
        # through auth_token so the SDK sends Bearer auth instead of x-api-key.
        # Check this before OAuth token shape detection because MiniMax secrets do
        # not use Anthropic's sk-ant-api prefix and would otherwise be misread as
        # Anthropic OAuth/setup tokens.
        kwargs["auth_token"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _is_third_party_anthropic_endpoint(base_url):
        # Third-party proxies (Microsoft Foundry, AWS Bedrock, etc.) use their
        # own API keys with x-api-key auth. Skip OAuth detection — their keys
        # don't follow Anthropic's sk-ant-* prefix convention and would be
        # misclassified as OAuth tokens.
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
    elif _is_oauth_token(api_key):
        # OAuth access token / setup-token → Bearer auth + Claude Code identity.
        # Anthropic routes OAuth requests based on user-agent and headers;
        # without Claude Code's fingerprint, requests get intermittent 500s.
        all_betas = common_betas + _OAUTH_ONLY_BETAS
        kwargs["auth_token"] = api_key
        kwargs["default_headers"] = {
            "anthropic-beta": ",".join(all_betas),
            "user-agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            "x-app": "cli",
        }
    else:
        # Regular API key → x-api-key header + common betas
        kwargs["api_key"] = api_key
        if common_betas:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}

    return _anthropic_sdk.Anthropic(**kwargs)


def build_anthropic_bedrock_client(region: str):
    """Create an AnthropicBedrock client for Bedrock Claude models.

    Uses the Anthropic SDK's native Bedrock adapter, which provides full
    Claude feature parity: prompt caching, thinking budgets, adaptive
    thinking, fast mode — features not available via the Converse API.

    Attaches the common Anthropic beta headers as client-level defaults so
    that Bedrock-hosted Claude models get the same enhanced features as
    native Anthropic. The ``context-1m-2025-08-07`` beta in particular
    unlocks the 1M context window for Opus 4.6/4.7 on Bedrock — without
    it, Bedrock caps these models at 200K even though the Anthropic API
    serves them with 1M natively.

    Auth uses the boto3 default credential chain (IAM roles, SSO, env vars).
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Bedrock provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )
    if not hasattr(_anthropic_sdk, "AnthropicBedrock"):
        raise ImportError(
            "anthropic.AnthropicBedrock not available. "
            "Upgrade with: pip install 'anthropic>=0.39.0'"
        )
    from httpx import Timeout

    return _anthropic_sdk.AnthropicBedrock(
        aws_region=region,
        timeout=Timeout(timeout=900.0, connect=10.0),
        default_headers={"anthropic-beta": ",".join([*_COMMON_BETAS, _CONTEXT_1M_BETA])},
    )


def _read_claude_code_credentials_from_keychain() -> Optional[Dict[str, Any]]:
    """Read Claude Code OAuth credentials from the macOS Keychain.

    Claude Code >=2.1.114 stores credentials in the macOS Keychain under the
    service name "Claude Code-credentials" rather than (or in addition to)
    the JSON file at ~/.claude/.credentials.json.

    The password field contains a JSON string with the same claudeAiOauth
    structure as the JSON file.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    if platform.system() != "Darwin":
        return None

    try:
        # Read the "Claude Code-credentials" generic password entry
        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", "Claude Code-credentials",
             "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("Keychain: security command not available or timed out")
        return None

    if result.returncode != 0:
        logger.debug("Keychain: no entry found for 'Claude Code-credentials'")
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Keychain: credentials payload is not valid JSON")
        return None

    oauth_data = data.get("claudeAiOauth")
    if oauth_data and isinstance(oauth_data, dict):
        access_token = oauth_data.get("accessToken", "")
        if access_token:
            return {
                "accessToken": access_token,
                "refreshToken": oauth_data.get("refreshToken", ""),
                "expiresAt": oauth_data.get("expiresAt", 0),
                "source": "macos_keychain",
            }

    return None


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials.

    Checks two sources in order:
      1. macOS Keychain (Darwin only) — "Claude Code-credentials" entry
      2. ~/.claude/.credentials.json file

    This intentionally excludes ~/.claude.json primaryApiKey. Opencode's
    subscription flow is OAuth/setup-token based with refreshable credentials,
    and native direct Anthropic provider usage should follow that path rather
    than auto-detecting Claude's first-party managed key.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    # Try macOS Keychain first (covers Claude Code >=2.1.114)
    kc_creds = _read_claude_code_credentials_from_keychain()
    if kc_creds:
        return kc_creds

    # Fall back to JSON file
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if cred_path.exists():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            oauth_data = data.get("claudeAiOauth")
            if oauth_data and isinstance(oauth_data, dict):
                access_token = oauth_data.get("accessToken", "")
                if access_token:
                    return {
                        "accessToken": access_token,
                        "refreshToken": oauth_data.get("refreshToken", ""),
                        "expiresAt": oauth_data.get("expiresAt", 0),
                        "source": "claude_code_credentials_file",
                    }
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude/.credentials.json: %s", e)

    return None


def read_claude_managed_key() -> Optional[str]:
    """Read Claude's native managed key from ~/.claude.json for diagnostics only."""
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            data = json.loads(claude_json.read_text(encoding="utf-8"))
            primary_key = data.get("primaryApiKey", "")
            if isinstance(primary_key, str) and primary_key.strip():
                return primary_key.strip()
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude.json: %s", e)
    return None


def is_claude_code_token_valid(creds: Dict[str, Any]) -> bool:
    """Check if Claude Code credentials have a non-expired access token."""
    import time

    expires_at = creds.get("expiresAt", 0)
    if not expires_at:
        # No expiry set (managed keys) — valid if token is present
        return bool(creds.get("accessToken"))

    # expiresAt is in milliseconds since epoch
    now_ms = int(time.time() * 1000)
    # Allow 60 seconds of buffer
    return now_ms < (expires_at - 60_000)


def refresh_anthropic_oauth_pure(refresh_token: str, *, use_json: bool = False) -> Dict[str, Any]:
    """Refresh an Anthropic OAuth token without mutating local credential files."""
    import time
    import urllib.parse
    import urllib.request

    if not refresh_token:
        raise ValueError("refresh_token is required")

    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    if use_json:
        data = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/json"
    else:
        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }).encode()
        content_type = "application/x-www-form-urlencoded"

    token_endpoints = [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ]
    last_error = None
    for endpoint in token_endpoints:
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": content_type,
                "User-Agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception as exc:
            last_error = exc
            logger.debug("Anthropic token refresh failed at %s: %s", endpoint, exc)
            continue

        access_token = result.get("access_token", "")
        if not access_token:
            raise ValueError("Anthropic refresh response was missing access_token")
        next_refresh = result.get("refresh_token", refresh_token)
        expires_in = result.get("expires_in", 3600)
        return {
            "access_token": access_token,
            "refresh_token": next_refresh,
            "expires_at_ms": int(time.time() * 1000) + (expires_in * 1000),
        }

    if last_error is not None:
        raise last_error
    raise ValueError("Anthropic token refresh failed")


def _refresh_oauth_token(creds: Dict[str, Any]) -> Optional[str]:
    """Attempt to refresh an expired Claude Code OAuth token."""
    refresh_token = creds.get("refreshToken", "")
    if not refresh_token:
        logger.debug("No refresh token available — cannot refresh")
        return None

    try:
        refreshed = refresh_anthropic_oauth_pure(refresh_token, use_json=False)
        _write_claude_code_credentials(
            refreshed["access_token"],
            refreshed["refresh_token"],
            refreshed["expires_at_ms"],
        )
        logger.debug("Successfully refreshed Claude Code OAuth token")
        return refreshed["access_token"]
    except Exception as e:
        logger.debug("Failed to refresh Claude Code token: %s", e)
        return None


def _write_claude_code_credentials(
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
    *,
    scopes: Optional[list] = None,
) -> None:
    """Write refreshed credentials back to ~/.claude/.credentials.json.

    The optional *scopes* list (e.g. ``["user:inference", "user:profile", ...]``)
    is persisted so that Claude Code's own auth check recognises the credential
    as valid.  Claude Code >=2.1.81 gates on the presence of ``"user:inference"``
    in the stored scopes before it will use the token.
    """
    cred_path = Path.home() / ".claude" / ".credentials.json"
    try:
        # Read existing file to preserve other fields
        existing = {}
        if cred_path.exists():
            existing = json.loads(cred_path.read_text(encoding="utf-8"))

        oauth_data: Dict[str, Any] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }
        if scopes is not None:
            oauth_data["scopes"] = scopes
        elif "claudeAiOauth" in existing and "scopes" in existing["claudeAiOauth"]:
            # Preserve previously-stored scopes when the refresh response
            # does not include a scope field.
            oauth_data["scopes"] = existing["claudeAiOauth"]["scopes"]

        existing["claudeAiOauth"] = oauth_data

        cred_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process random suffix avoids collisions between concurrent
        # writers and stale leftovers from a prior crashed write.
        _tmp_cred = cred_path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
        try:
            # Create the temp file atomically at 0o600. The previous
            # write_text + post-replace chmod opened a TOCTOU window where
            # both the temp file and the destination briefly inherited the
            # process umask (commonly 0o644 = world-readable), exposing
            # Claude Code OAuth tokens to other local users between create
            # and chmod. Mirrors agent/google_oauth.py (#19673) and
            # tools/mcp_oauth.py (#21148). Parent dir (~/.claude/) is
            # owned by Claude Code itself, so we leave its mode alone.
            fd = os.open(
                str(_tmp_cred),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(_tmp_cred, cred_path)
        except OSError:
            try:
                _tmp_cred.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except (OSError, IOError) as e:
        logger.debug("Failed to write refreshed credentials: %s", e)


def _resolve_claude_code_token_from_credentials(creds: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve a token from Claude Code credential files, refreshing if needed."""
    creds = creds or read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        logger.debug("Using Claude Code credentials (auto-detected)")
        return creds["accessToken"]
    if creds:
        logger.debug("Claude Code credentials expired — attempting refresh")
        refreshed = _refresh_oauth_token(creds)
        if refreshed:
            return refreshed
        logger.debug("Token refresh failed — re-run 'claude setup-token' to reauthenticate")
    return None


def _prefer_refreshable_claude_code_token(env_token: str, creds: Optional[Dict[str, Any]]) -> Optional[str]:
    """Prefer Claude Code creds when a persisted env OAuth token would shadow refresh.

    Hermes historically persisted setup tokens into ANTHROPIC_TOKEN. That makes
    later refresh impossible because the static env token wins before we ever
    inspect Claude Code's refreshable credential file. If we have a refreshable
    Claude Code credential record, prefer it over the static env OAuth token.
    """
    if not env_token or not _is_oauth_token(env_token) or not isinstance(creds, dict):
        return None
    if not creds.get("refreshToken"):
        return None

    resolved = _resolve_claude_code_token_from_credentials(creds)
    if resolved and resolved != env_token:
        logger.debug(
            "Preferring Claude Code credential file over static env OAuth token so refresh can proceed"
        )
        return resolved
    return None


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all available sources.

    Priority:
      1. ANTHROPIC_TOKEN env var (OAuth/setup token saved by Hermes)
      2. CLAUDE_CODE_OAUTH_TOKEN env var
      3. Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json)
         — with automatic refresh if expired and a refresh token is available
      4. ANTHROPIC_API_KEY env var (regular API key, or legacy fallback)

    Returns the token string or None.
    """
    creds = read_claude_code_credentials()

    # 1. Hermes-managed OAuth/setup token env var
    token = os.getenv("ANTHROPIC_TOKEN", "").strip()
    if token:
        preferred = _prefer_refreshable_claude_code_token(token, creds)
        if preferred:
            return preferred
        return token

    # 2. CLAUDE_CODE_OAUTH_TOKEN (used by Claude Code for setup-tokens)
    cc_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if cc_token:
        preferred = _prefer_refreshable_claude_code_token(cc_token, creds)
        if preferred:
            return preferred
        return cc_token

    # 3. Claude Code credential file
    resolved_claude_token = _resolve_claude_code_token_from_credentials(creds)
    if resolved_claude_token:
        return resolved_claude_token

    # 4. Regular API key, or a legacy OAuth token saved in ANTHROPIC_API_KEY.
    # This remains as a compatibility fallback for pre-migration Hermes configs.
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key

    return None


def run_oauth_setup_token() -> Optional[str]:
    """Run 'claude setup-token' interactively and return the resulting token.

    Checks multiple sources after the subprocess completes:
      1. Claude Code credential files (may be written by the subprocess)
      2. CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_TOKEN env vars

    Returns the token string, or None if no credentials were obtained.
    Raises FileNotFoundError if the 'claude' CLI is not installed.
    """
    import shutil
    import subprocess

    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError(
            "The 'claude' CLI is not installed. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )

    # Run interactively — stdin/stdout/stderr inherited so user can interact
    try:
        subprocess.run([claude_path, "setup-token"])
    except (KeyboardInterrupt, EOFError):
        return None

    # Check if credentials were saved to Claude Code's config files
    creds = read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        return creds["accessToken"]

    # Check env vars that may have been set
    for env_var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"):
        val = os.getenv(env_var, "").strip()
        if val:
            return val

    return None


# ── Hermes-native PKCE OAuth flow ────────────────────────────────────────
# Mirrors the flow used by Claude Code, pi-ai, and OpenCode.
# Stores credentials in ~/.hermes/.anthropic_oauth.json (our own file).

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"
_HERMES_OAUTH_FILE = get_hermes_home() / ".anthropic_oauth.json"


def _generate_pkce() -> tuple:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    import base64
    import hashlib
    import secrets

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def run_hermes_oauth_login_pure() -> Optional[Dict[str, Any]]:
    """Run Hermes-native OAuth PKCE flow and return credential state."""
    import secrets
    import time
    import webbrowser

    verifier, challenge = _generate_pkce()
    oauth_state = secrets.token_urlsafe(32)

    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": oauth_state,
    }
    from urllib.parse import urlencode

    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    print()
    print("Authorize Hermes with your Claude Pro/Max subscription.")
    print()
    print("╭─ Claude Pro/Max Authorization ────────────────────╮")
    print("│                                                   │")
    print("│  Open this link in your browser:                  │")
    print("╰───────────────────────────────────────────────────╯")
    print()
    print(f"  {auth_url}")
    print()

    try:
        from hermes_cli.auth import _can_open_graphical_browser as _can_open_gui
    except Exception:
        _can_open_gui = lambda: True  # noqa: E731 — degrade to prior behavior

    if _can_open_gui():
        try:
            webbrowser.open(auth_url)
            print("  (Browser opened automatically)")
        except Exception:
            pass

    print()
    print("After authorizing, you'll see a code. Paste it below.")
    print()
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    if not auth_code:
        print("No code entered.")
        return None

    splits = auth_code.split("#")
    code = splits[0]
    received_state = splits[1] if len(splits) > 1 else ""

    # Validate state to prevent CSRF (RFC 6749 §10.12)
    if received_state != oauth_state:
        logger.warning("OAuth state mismatch — possible CSRF, aborting")
        return None

    try:
        import urllib.request

        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": received_state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=exchange_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        print("No access token in response.")
        return None

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at_ms": expires_at_ms,
    }


def read_hermes_oauth_credentials() -> Optional[Dict[str, Any]]:
    """Read Hermes-managed OAuth credentials from ~/.hermes/.anthropic_oauth.json."""
    if _HERMES_OAUTH_FILE.exists():
        try:
            data = json.loads(_HERMES_OAUTH_FILE.read_text(encoding="utf-8"))
            if data.get("accessToken"):
                return data
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read Hermes OAuth credentials: %s", e)
    return None


# ---------------------------------------------------------------------------
# Message / tool / response format conversion
# ---------------------------------------------------------------------------


def _is_bedrock_model_id(model: str) -> bool:
    """Detect AWS Bedrock model IDs that use dots as namespace separators.

    Bedrock model IDs come in two forms:
    - Bare:    ``anthropic.claude-opus-4-7``
    - Regional (inference profiles): ``us.anthropic.claude-sonnet-4-5-v1:0``

    In both cases the dots separate namespace components, not version
    numbers, and must be preserved verbatim for the Bedrock API.
    """
    lower = model.lower()
    # Regional inference-profile prefixes
    if any(lower.startswith(p) for p in ("global.", "us.", "eu.", "ap.", "jp.")):
        return True
    # Bare Bedrock model IDs: provider.model-family
    if lower.startswith("anthropic."):
        return True
    return False
