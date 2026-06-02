"""Bridge module — delegates plugin registration to hermes_agent_matrix."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_matrix package."""
    from hermes_agent_matrix import register as _inner_register
    _inner_register(ctx)
