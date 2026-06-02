"""Bridge module — delegates plugin registration to hermes_agent_dashboard."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_dashboard package."""
    from hermes_agent_dashboard import register as _inner_register
    _inner_register(ctx)
