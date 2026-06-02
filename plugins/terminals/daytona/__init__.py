"""Bridge module — delegates plugin registration to hermes_agent_daytona."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_daytona package."""
    from hermes_agent_daytona import register as _inner_register
    _inner_register(ctx)
