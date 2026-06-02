"""Bridge module — delegates plugin registration to hermes_agent_tts."""


def register(ctx):
    """Plugin entry point — delegates to the inner hermes_agent_tts package."""
    from hermes_agent_tts import register as _inner_register
    _inner_register(ctx)
