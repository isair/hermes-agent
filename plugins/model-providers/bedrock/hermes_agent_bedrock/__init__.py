"""hermes-agent-bedrock: AWS Bedrock Converse API adapter for Hermes Agent."""

from hermes_agent_bedrock.adapter import (  # noqa: F401
    BEDROCK_DEFAULT_CONTEXT_LENGTH,
    CONTEXT_OVERFLOW_PATTERNS,
    OVERLOAD_PATTERNS,
    THROTTLE_PATTERNS,
    _AWS_CREDENTIAL_ENV_VARS,
    _DISCOVERY_CACHE_TTL_SECONDS,
    _NON_TOOL_CALLING_PATTERNS,
    _STALE_LIB_MODULE_PREFIXES,
    _convert_content_to_converse,
    _converse_stop_reason_to_openai,
    _extract_provider_from_arn,
    _get_bedrock_control_client,
    _get_bedrock_runtime_client,
    _model_supports_tool_use,
    _require_boto3,
    _traceback_frames_modules,
    bedrock_model_ids_or_none,
    build_converse_kwargs,
    call_converse,
    call_converse_stream,
    classify_bedrock_error,
    convert_messages_to_converse,
    convert_tools_to_converse,
    discover_bedrock_models,
    get_bedrock_context_length,
    has_aws_credentials,
    invalidate_runtime_client,
    is_anthropic_bedrock_model,
    is_context_overflow_error,
    is_stale_connection_error,
    normalize_converse_response,
    normalize_converse_stream_events,
    reset_client_cache,
    reset_discovery_cache,
    resolve_aws_auth_env_var,
    resolve_bedrock_region,
    stream_converse_with_callbacks,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_bedrock import adapter

    ctx.register_provider_services("bedrock", {
        # Auth / credentials
        "has_aws_credentials": adapter.has_aws_credentials,
        "resolve_aws_auth_env_var": adapter.resolve_aws_auth_env_var,
        "resolve_bedrock_region": adapter.resolve_bedrock_region,
        "_AWS_CREDENTIAL_ENV_VARS": adapter._AWS_CREDENTIAL_ENV_VARS,
        # Transport
        "build_converse_kwargs": adapter.build_converse_kwargs,
        "convert_messages_to_converse": adapter.convert_messages_to_converse,
        "convert_tools_to_converse": adapter.convert_tools_to_converse,
        "normalize_converse_response": adapter.normalize_converse_response,
        "normalize_converse_stream_events": adapter.normalize_converse_stream_events,
        "call_converse": adapter.call_converse,
        "call_converse_stream": adapter.call_converse_stream,
        "stream_converse_with_callbacks": adapter.stream_converse_with_callbacks,
        # Model metadata
        "bedrock_model_ids_or_none": adapter.bedrock_model_ids_or_none,
        "discover_bedrock_models": adapter.discover_bedrock_models,
        "get_bedrock_context_length": adapter.get_bedrock_context_length,
        "BEDROCK_DEFAULT_CONTEXT_LENGTH": adapter.BEDROCK_DEFAULT_CONTEXT_LENGTH,
        # Client management
        "_get_bedrock_control_client": adapter._get_bedrock_control_client,
        "_get_bedrock_runtime_client": adapter._get_bedrock_runtime_client,
        "invalidate_runtime_client": adapter.invalidate_runtime_client,
        "reset_client_cache": adapter.reset_client_cache,
        "reset_discovery_cache": adapter.reset_discovery_cache,
        # Error handling
        "classify_bedrock_error": adapter.classify_bedrock_error,
        "is_context_overflow_error": adapter.is_context_overflow_error,
        "is_stale_connection_error": adapter.is_stale_connection_error,
        "CONTEXT_OVERFLOW_PATTERNS": adapter.CONTEXT_OVERFLOW_PATTERNS,
        "OVERLOAD_PATTERNS": adapter.OVERLOAD_PATTERNS,
        "THROTTLE_PATTERNS": adapter.THROTTLE_PATTERNS,
        "_NON_TOOL_CALLING_PATTERNS": adapter._NON_TOOL_CALLING_PATTERNS,
        "_STALE_LIB_MODULE_PREFIXES": adapter._STALE_LIB_MODULE_PREFIXES,
        "_DISCOVERY_CACHE_TTL_SECONDS": adapter._DISCOVERY_CACHE_TTL_SECONDS,
        # Internal helpers
        "_require_boto3": adapter._require_boto3,
        "_model_supports_tool_use": adapter._model_supports_tool_use,
        "is_anthropic_bedrock_model": adapter.is_anthropic_bedrock_model,
        "_convert_content_to_converse": adapter._convert_content_to_converse,
        "_converse_stop_reason_to_openai": adapter._converse_stop_reason_to_openai,
        "_extract_provider_from_arn": adapter._extract_provider_from_arn,
        "_traceback_frames_modules": adapter._traceback_frames_modules,
    })

    # Register the provider resolver — core dispatches to this instead of
    # having per-bedrock if/elif branches in resolve_provider_client().
    from hermes_agent_bedrock.resolve import resolve_auxiliary_client as _bedrock_resolver
    ctx.register_provider_resolver("bedrock", _bedrock_resolver)

    # Register the bedrock transport so core doesn't need to import it.
    from hermes_agent_bedrock.transport import BedrockTransport
    ctx.register_transport("bedrock_converse", BedrockTransport)

    # Register pricing entries — core looks these up via the registry
    # instead of hardcoding them in _OFFICIAL_DOCS_PRICING.
    from hermes_agent_bedrock.pricing import (
        get_bedrock_pricing_entries,
        BEDROCK_PRICING_KEYS,
    )
    _entries = get_bedrock_pricing_entries()
    _keyed = []
    for (prov, model), entry in zip(BEDROCK_PRICING_KEYS, _entries):
        _keyed.append((prov, model, entry))
    ctx.register_pricing_provider("bedrock", _keyed)

    # Register the provider overlay — core merges this into HERMES_OVERLAYS
    from agent.plugin_registries import ProviderOverlayEntry
    ctx.register_provider_overlay(ProviderOverlayEntry(
        provider_name="bedrock",
        transport="bedrock_converse",
        auth_type="aws_sdk",
        display_name="AWS Bedrock",
        aliases=["aws", "aws-bedrock", "amazon-bedrock", "amazon"],
    ))
