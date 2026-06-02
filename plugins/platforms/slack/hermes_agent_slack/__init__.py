"""hermes-agent-slack: Slack platform adapter for Hermes Agent."""

from hermes_agent_slack.adapter import (  # noqa: F401
    SlackAdapter,
    check_slack_requirements,
    _slash_user_id,
    SLACK_AVAILABLE,
    AsyncApp,
    AsyncWebClient,
    AsyncSocketModeHandler,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_slack.adapter import (
        SlackAdapter,
        check_slack_requirements,
    )

    ctx.register_platform(
        name="slack",
        label="Slack",
        adapter_factory=lambda cfg: SlackAdapter(cfg),
        check_fn=check_slack_requirements,
        install_hint="pip install 'hermes-agent[slack]'",
        emoji="💬",
    )

    ctx.register_platform_entry(
        name="slack",
        adapter_class=SlackAdapter,
        check_requirements=check_slack_requirements,
    )
