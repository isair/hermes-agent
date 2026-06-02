"""hermes-agent-fal: FAL.ai SDK plumbing plugin for Hermes Agent."""

from hermes_agent_fal.fal_common import (  # noqa: F401
    import_fal_client,
    _ManagedFalSyncClient,
    _extract_http_status,
    _normalize_fal_queue_url_format,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group.

    Registers FAL SDK plumbing (import_fal_client, _ManagedFalSyncClient,
    etc.) in the plugin capability registry so core code can look them
    up without importing from ``hermes_agent_fal`` directly.
    """
    from hermes_agent_fal.fal_common import (
        import_fal_client,
        _ManagedFalSyncClient,
        _extract_http_status,
        _normalize_fal_queue_url_format,
    )
    ctx.register_tool_provider_entry(
        name="fal",
        tool_functions={
            "import_fal_client": import_fal_client,
        },
        constants={
            "_normalize_fal_queue_url_format": _normalize_fal_queue_url_format,
        },
        config_functions={
            "_ManagedFalSyncClient": _ManagedFalSyncClient,
            "_extract_http_status": _extract_http_status,
        },
    )
