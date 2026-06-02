"""Bridge module — delegates plugin registration to hermes_agent_slack."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_slack package."""
    from hermes_agent_slack import register as _inner_register
    _inner_register(ctx)
