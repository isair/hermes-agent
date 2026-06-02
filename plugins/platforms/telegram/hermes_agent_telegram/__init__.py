"""hermes-agent-telegram: Telegram platform adapter for Hermes Agent."""

from hermes_agent_telegram.adapter import (  # noqa: F401
    TelegramAdapter,
    check_telegram_requirements,
    _strip_mdv2,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_telegram.adapter import (
        TelegramAdapter,
        check_telegram_requirements,
        _strip_mdv2,
    )

    ctx.register_platform(
        name="telegram",
        label="Telegram",
        adapter_factory=lambda cfg: TelegramAdapter(cfg),
        check_fn=check_telegram_requirements,
        emoji="✈️",
    )

    ctx.register_platform_entry(
        name="telegram",
        adapter_class=TelegramAdapter,
        check_requirements=check_telegram_requirements,
        helper_functions={"_strip_mdv2": _strip_mdv2},
    )
