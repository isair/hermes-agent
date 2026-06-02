"""hermes-agent-anthropic: Anthropic Messages API adapter for Hermes Agent."""

# -----------------------------------------------------------------------
# Re-exports from adapter.py — SDK-dependent orchestration only.
# Wire-format code (message conversion, aux client wrappers, transport)
# has moved to core and is no longer re-exported here.
# -----------------------------------------------------------------------
from hermes_agent_anthropic.adapter import (  # noqa: F401
    _CLAUDE_CODE_VERSION_FALLBACK,
    _HERMES_OAUTH_FILE,
    _OAUTH_CLIENT_ID,
    _OAUTH_REDIRECT_URI,
    _OAUTH_SCOPES,
    _OAUTH_TOKEN_URL,
    _build_anthropic_client_with_bearer_hook,
    _detect_claude_code_version,
    _generate_pkce,
    _get_anthropic_sdk,
    _get_claude_code_version,
    _is_azure_anthropic_endpoint,
    _is_oauth_token,
    _prefer_refreshable_claude_code_token,
    _read_claude_code_credentials_from_keychain,
    _refresh_oauth_token,
    _requires_bearer_auth,
    _resolve_claude_code_token_from_credentials,
    _write_claude_code_credentials,
    build_anthropic_bedrock_client,
    build_anthropic_client,
    is_claude_code_token_valid,
    read_claude_code_credentials,
    read_claude_managed_key,
    read_hermes_oauth_credentials,
    refresh_anthropic_oauth_pure,
    resolve_anthropic_token,
    run_hermes_oauth_login_pure,
    run_oauth_setup_token,
)

# Re-exports from resolve.py — client resolution & endpoint detection
from hermes_agent_anthropic.resolve import (  # noqa: F401
    _ANTHROPIC_DEFAULT_BASE_URL as ANTHROPIC_DEFAULT_BASE_URL,
    convert_openai_images_to_anthropic,
    endpoint_speaks_anthropic_messages,
    is_anthropic_compat_endpoint,
    maybe_wrap_anthropic,
    resolve_auxiliary_client,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_anthropic import adapter

    # -----------------------------------------------------------------------
    # Plugin-only symbols — SDK-dependent orchestration that stays in the
    # plugin package.  Wire-format code (message conversion, aux client
    # wrappers, transport) has moved to core (agent.anthropic_format,
    # agent.anthropic_aux, agent.transports.anthropic) and is no longer
    # registered here.
    # -----------------------------------------------------------------------
    _symbols = [
        # OAuth / auth constants
        "_CLAUDE_CODE_VERSION_FALLBACK",
        "_HERMES_OAUTH_FILE",
        "_OAUTH_CLIENT_ID",
        "_OAUTH_REDIRECT_URI",
        "_OAUTH_SCOPES",
        "_OAUTH_TOKEN_URL",
        # SDK-dependent functions
        "_build_anthropic_client_with_bearer_hook",
        "_detect_claude_code_version",
        "_generate_pkce",
        "_get_anthropic_sdk",
        "_get_claude_code_version",
        "_is_azure_anthropic_endpoint",
        "_is_oauth_token",
        "_prefer_refreshable_claude_code_token",
        "_read_claude_code_credentials_from_keychain",
        "_refresh_oauth_token",
        "_requires_bearer_auth",
        "_resolve_claude_code_token_from_credentials",
        "_write_claude_code_credentials",
        "build_anthropic_bedrock_client",
        "build_anthropic_client",
        "is_claude_code_token_valid",
        "read_claude_code_credentials",
        "read_claude_managed_key",
        "read_hermes_oauth_credentials",
        "refresh_anthropic_oauth_pure",
        "resolve_anthropic_token",
        "run_hermes_oauth_login_pure",
        "run_oauth_setup_token",
    ]

    # resolve.py symbols — client resolution & endpoint detection
    _resolve_symbols = [
        "_ANTHROPIC_DEFAULT_BASE_URL",
        "_ANTHROPIC_COMPAT_PROVIDERS",
        "convert_openai_images_to_anthropic",
        "endpoint_speaks_anthropic_messages",
        "is_anthropic_compat_endpoint",
        "maybe_wrap_anthropic",
        "resolve_auxiliary_client",
    ]
    _all_symbols = _symbols + _resolve_symbols
    _services = {}
    for name in _symbols:
        _services[name] = getattr(adapter, name)
    for name in _resolve_symbols:
        from hermes_agent_anthropic import resolve as _resolve_mod
        _services[name] = getattr(_resolve_mod, name)
    # Also expose ANTHROPIC_DEFAULT_BASE_URL under the public (no-underscore) name
    _services["ANTHROPIC_DEFAULT_BASE_URL"] = _services.get("_ANTHROPIC_DEFAULT_BASE_URL", "")

    # Also expose the model name normalizer as a provider service
    from hermes_agent_anthropic.pricing import normalize_anthropic_model_name
    _services["normalize_model_name"] = normalize_anthropic_model_name

    ctx.register_provider_services("anthropic", _services)

    # Register the provider resolver — core dispatches to this instead of
    # having per-anthropic if/elif branches in resolve_provider_client().
    ctx.register_provider_resolver("anthropic", resolve_auxiliary_client)

    # Register the anthropic transport so core doesn't need to import it.
    from agent.transports.anthropic import AnthropicTransport
    ctx.register_transport("anthropic_messages", AnthropicTransport)

    # Register the credential pool hook — core dispatches to this instead of
    # having per-anthropic if/elif branches in credential_pool.py.
    from agent.plugin_registries import CredentialPoolHook
    from hermes_agent_anthropic.credential_pool_hook import (
        sync_from_credentials_file,
        refresh_oauth,
        needs_refresh,
        should_include_in_pool,
        source_priority,
        discover_credentials,
        ANTHROPIC_ENV_VAR_ORDER,
        detect_auth_type,
    )
    ctx.register_credential_pool_hook("anthropic", CredentialPoolHook(
        sync_from_credentials_file=sync_from_credentials_file,
        refresh_oauth=refresh_oauth,
        needs_refresh=needs_refresh,
        should_include_in_pool=should_include_in_pool,
        source_priority=source_priority,
        discover_credentials=discover_credentials,
        env_var_order=ANTHROPIC_ENV_VAR_ORDER,
        detect_auth_type=detect_auth_type,
    ))

    # Register pricing entries — core looks these up via the registry
    # instead of hardcoding them in _OFFICIAL_DOCS_PRICING.
    from hermes_agent_anthropic.pricing import (
        get_anthropic_pricing_entries,
        ANTHROPIC_PRICING_KEYS,
    )
    _entries = get_anthropic_pricing_entries()
    _keyed = []
    for (prov, model), entry in zip(ANTHROPIC_PRICING_KEYS, _entries):
        _keyed.append((prov, model, entry))
    ctx.register_pricing_provider("anthropic", _keyed)

    # Register the provider overlay — core merges this into HERMES_OVERLAYS
    from agent.plugin_registries import ProviderOverlayEntry
    ctx.register_provider_overlay(ProviderOverlayEntry(
        provider_name="anthropic",
        transport="anthropic_messages",
        extra_env_vars=("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
        display_name="Anthropic",
        aliases=[],
    ))
