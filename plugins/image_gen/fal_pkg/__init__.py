"""Bridge module — delegates plugin registration to hermes_agent_fal."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_fal package."""
    from hermes_agent_fal import register as _inner_register
    _inner_register(ctx)
