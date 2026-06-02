"""Bridge module — delegates plugin registration to hermes_agent_dingtalk."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_dingtalk package."""
    from hermes_agent_dingtalk import register as _inner_register
    _inner_register(ctx)
