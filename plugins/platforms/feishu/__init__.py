"""Bridge module — delegates plugin registration to hermes_agent_feishu."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_feishu package."""
    from hermes_agent_feishu import register as _inner_register
    _inner_register(ctx)
