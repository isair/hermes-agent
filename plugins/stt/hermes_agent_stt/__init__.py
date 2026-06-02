"""hermes-agent-stt: Speech-to-text transcription plugin for Hermes Agent."""

from hermes_agent_stt.transcription_tools import (  # noqa: F401
    BUILTIN_STT_PROVIDERS,
    transcribe_audio,
    MAX_FILE_SIZE,
    SUPPORTED_FORMATS,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_GROQ_STT_MODEL,
    GROQ_BASE_URL,
    LOCAL_STT_COMMAND_ENV,
    OPENAI_BASE_URL,
    _get_local_command_template,
    _get_provider,
    _load_stt_config,
    _normalize_local_model,
    _transcribe_groq,
    _transcribe_local,
    _transcribe_local_command,
    _transcribe_mistral,
    _transcribe_openai,
    _transcribe_xai,
    _validate_audio_file,
    is_stt_enabled,
)


def register(ctx):
    """Entry point for the hermes_agent.plugins entry point group.

    Registers STT tool functions, constants, and config helpers in the
    plugin capability registry so core code (gateway, CLI) can look
    them up without importing from ``hermes_agent_stt`` directly.
    """
    from hermes_agent_stt.transcription_tools import (
        transcribe_audio,
        MAX_FILE_SIZE,
        _get_provider,
        _load_stt_config,
        is_stt_enabled,
    )
    ctx.register_tool_provider_entry(
        name="stt",
        tool_functions={
            "transcribe_audio": transcribe_audio,
        },
        constants={
            "MAX_FILE_SIZE": MAX_FILE_SIZE,
        },
        config_functions={
            "_get_provider": _get_provider,
            "_load_stt_config": _load_stt_config,
            "is_stt_enabled": is_stt_enabled,
        },
    )
