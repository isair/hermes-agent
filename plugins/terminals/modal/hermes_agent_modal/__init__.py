"""Modal serverless terminal backend."""
from .modal import ModalEnvironment

def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group.

    Registers ModalEnvironment in the plugin capability registry so core
    code can look it up without importing from ``hermes_agent_modal``
    directly.
    """
    from .modal import ModalEnvironment
    ctx.register_tool_provider_entry(
        name="modal",
        environment_classes={
            "ModalEnvironment": ModalEnvironment,
        },
    )

__all__ = ["ModalEnvironment"]
