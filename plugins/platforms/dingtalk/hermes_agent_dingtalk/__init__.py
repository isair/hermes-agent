"""hermes-agent-dingtalk: DingTalk platform adapter for Hermes Agent."""

from hermes_agent_dingtalk.adapter import (  # noqa: F401
    DingTalkAdapter,
    check_dingtalk_requirements,
    _DINGTALK_WEBHOOK_RE,
    _IncomingHandler,
    DINGTALK_TYPE_MAPPING,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_dingtalk.adapter import (
        DingTalkAdapter,
        check_dingtalk_requirements,
    )

    ctx.register_platform(
        name="dingtalk",
        label="DingTalk",
        adapter_factory=lambda cfg: DingTalkAdapter(cfg),
        check_fn=check_dingtalk_requirements,
        install_hint="pip install 'hermes-agent[dingtalk]'",
        emoji="📢",
    )

    ctx.register_platform_entry(
        name="dingtalk",
        adapter_class=DingTalkAdapter,
        check_requirements=check_dingtalk_requirements,
    )
