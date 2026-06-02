"""hermes-agent-feishu: Feishu/Lark platform adapter for Hermes Agent."""

from hermes_agent_feishu.adapter import (  # noqa: F401
    FeishuAdapter,
    check_feishu_requirements,
    FEISHU_AVAILABLE,
    FEISHU_DOMAIN,
    LARK_DOMAIN,
    qr_register,
    probe_bot,
    normalize_feishu_message,
    _run_official_feishu_ws_client,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group."""
    from hermes_agent_feishu import (
        FeishuAdapter,
        check_feishu_requirements,
        FEISHU_AVAILABLE,
        FEISHU_DOMAIN,
        LARK_DOMAIN,
        qr_register,
        probe_bot,
    )

    ctx.register_platform(
        name="feishu",
        label="Feishu / Lark",
        adapter_factory=lambda cfg: FeishuAdapter(cfg),
        check_fn=check_feishu_requirements,
        install_hint="pip install 'hermes-agent[feishu]'",
        emoji="🐦",
    )

    ctx.register_platform_entry(
        name="feishu",
        adapter_class=FeishuAdapter,
        check_requirements=check_feishu_requirements,
        available_flag="FEISHU_AVAILABLE",
        constants={
            "FEISHU_AVAILABLE": FEISHU_AVAILABLE,
            "FEISHU_DOMAIN": FEISHU_DOMAIN,
            "LARK_DOMAIN": LARK_DOMAIN,
        },
        helper_functions={
            "qr_register": qr_register,
            "probe_bot": probe_bot,
        },
    )
