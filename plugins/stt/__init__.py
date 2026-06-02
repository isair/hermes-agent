"""Bridge module — delegates plugin registration to hermes_agent_stt."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_stt package."""
    from hermes_agent_stt import register as _inner_register
    _inner_register(ctx)
