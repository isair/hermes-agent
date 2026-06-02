"""Bridge module — delegates plugin registration to hermes_agent_telegram."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_telegram package."""
    from hermes_agent_telegram import register as _inner_register
    _inner_register(ctx)
