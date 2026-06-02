"""hermes-agent-tts: Text-to-speech tool plugin for Hermes Agent."""

from hermes_agent_tts.tts_tool import (  # noqa: F401
    BUILTIN_TTS_PROVIDERS,
    text_to_speech_tool,
    check_tts_requirements,
    _strip_markdown_for_tts,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group.

    Registers TTS tool functions in the plugin capability registry so
    core code (gateway, CLI) can look them up without importing from
    ``hermes_agent_tts`` directly.
    """
    from hermes_agent_tts.tts_tool import (
        text_to_speech_tool,
        check_tts_requirements,
        _strip_markdown_for_tts,
    )
    ctx.register_tool_provider_entry(
        name="tts",
        tool_functions={
            "text_to_speech_tool": text_to_speech_tool,
            "_strip_markdown_for_tts": _strip_markdown_for_tts,
        },
        check_fn=check_tts_requirements,
    )
