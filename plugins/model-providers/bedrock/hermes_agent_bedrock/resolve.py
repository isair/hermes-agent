"""Bedrock provider resolver for auxiliary client construction.

Handles ALL provider-specific logic for building auxiliary clients:
AWS credential detection, region resolution, and Bedrock client construction.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
    """Resolve an auxiliary client for the Bedrock provider.

    Returns ``(client, default_model)`` or ``(None, None)`` if unavailable.
    """
    from agent.plugin_registries import registries
    from agent.anthropic_aux import (
        AnthropicAuxiliaryClient,
        AsyncAnthropicAuxiliaryClient,
    )

    _bedrock = registries.get_provider_namespace("bedrock")
    _anthropic = registries.get_provider_namespace("anthropic")
    has_aws_credentials = _bedrock.get("has_aws_credentials")
    resolve_bedrock_region = _bedrock.get("resolve_bedrock_region")
    build_anthropic_bedrock_client = _anthropic.get("build_anthropic_bedrock_client")
    if has_aws_credentials is None or resolve_bedrock_region is None or build_anthropic_bedrock_client is None:
        return None, None

    if not has_aws_credentials():
        logger.debug("resolve_provider_client: bedrock requested but "
                     "no AWS credentials found")
        return None, None

    region = resolve_bedrock_region()
    default_model = "anthropic.claude-haiku-4-5-20251001-v1:0"
    final_model = model or default_model
    try:
        real_client = build_anthropic_bedrock_client(region)
    except ImportError as exc:
        logger.warning("resolve_provider_client: cannot create Bedrock "
                       "client: %s", exc)
        return None, None
    client = AnthropicAuxiliaryClient(
        real_client, final_model, api_key="aws-sdk",
        base_url=f"https://bedrock-runtime.{region}.amazonaws.com",
    )
    logger.debug("resolve_provider_client: bedrock (%s, %s)", final_model, region)

    if async_mode:
        client = AsyncAnthropicAuxiliaryClient(client)

    return client, final_model
