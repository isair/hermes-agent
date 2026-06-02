"""hermes-agent-daytona: Daytona cloud execution environment plugin for Hermes Agent."""

from hermes_agent_daytona.daytona import DaytonaEnvironment  # noqa: F401


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group.

    Registers DaytonaEnvironment in the plugin capability registry so
    core code can look it up without importing from
    ``hermes_agent_daytona`` directly.
    """
    from hermes_agent_daytona.daytona import DaytonaEnvironment
    ctx.register_tool_provider_entry(
        name="daytona",
        environment_classes={
            "DaytonaEnvironment": DaytonaEnvironment,
        },
    )
