"""Bridge module — delegates plugin registration to hermes_agent_modal."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_modal package."""
    from hermes_agent_modal import register as _inner_register
    _inner_register(ctx)
