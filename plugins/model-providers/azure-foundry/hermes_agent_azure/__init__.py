"""hermes-agent-azure: Microsoft Entra ID / Azure Identity adapter for Hermes Agent."""

from hermes_agent_azure.adapter import (  # noqa: F401
    SCOPE_AI_AZURE_DEFAULT,
    EntraIdentityConfig,
    _build_default_credential,
    _require_azure_identity,
    build_bearer_http_client,
    build_credential,
    build_token_provider,
    describe_active_credential,
    has_azure_identity_credentials,
    has_azure_identity_installed,
    is_token_provider,
    materialize_bearer_for_http,
    reset_credential_cache,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_azure import adapter

    ctx.register_provider_services("azure", {
        # Auth / credentials
        "is_token_provider": adapter.is_token_provider,
        "has_azure_identity_credentials": adapter.has_azure_identity_credentials,
        "has_azure_identity_installed": adapter.has_azure_identity_installed,
        # Client building
        "build_bearer_http_client": adapter.build_bearer_http_client,
        "build_credential": adapter.build_credential,
        "build_token_provider": adapter.build_token_provider,
        "materialize_bearer_for_http": adapter.materialize_bearer_for_http,
        "reset_credential_cache": adapter.reset_credential_cache,
        # Constants / config
        "SCOPE_AI_AZURE_DEFAULT": adapter.SCOPE_AI_AZURE_DEFAULT,
        "EntraIdentityConfig": adapter.EntraIdentityConfig,
        # Internal helpers
        "_build_default_credential": adapter._build_default_credential,
        "_require_azure_identity": adapter._require_azure_identity,
        "describe_active_credential": adapter.describe_active_credential,
    })

    # Register the provider resolver — core dispatches to this instead of
    # having a per-azure-foundry if/elif branch in resolve_provider_client().
    from hermes_agent_azure.resolve import resolve_auxiliary_client as _azure_resolver
    ctx.register_provider_resolver("azure-foundry", _azure_resolver)

    # Register the provider overlay — core merges this into HERMES_OVERLAYS
    from agent.plugin_registries import ProviderOverlayEntry
    ctx.register_provider_overlay(ProviderOverlayEntry(
        provider_name="azure-foundry",
        transport="openai_chat",  # default; overridden by api_mode in config
        base_url_env_var="AZURE_FOUNDRY_BASE_URL",
        display_name="Azure AI Foundry",
        aliases=[],
    ))
