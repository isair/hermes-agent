"""Anthropic-specific fast mode tests moved from tests/cli/test_fast_command.py."""

import unittest
from types import SimpleNamespace


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestAnthropicFastMode(unittest.TestCase):
    """Verify Anthropic Fast Mode model support and override resolution."""

    def test_anthropic_opus_supported(self):
        from hermes_cli.models import model_supports_fast_mode

        # Native Anthropic format (hyphens)
        assert model_supports_fast_mode("claude-opus-4-6") is True
        # OpenRouter format (dots)
        assert model_supports_fast_mode("claude-opus-4.6") is True
        # With vendor prefix
        assert model_supports_fast_mode("anthropic/claude-opus-4-6") is True
        assert model_supports_fast_mode("anthropic/claude-opus-4.6") is True

    def test_anthropic_non_opus46_models_excluded(self):
        """Anthropic restricts fast mode to Opus 4.6 — others must be excluded.

        Per https://platform.claude.com/docs/en/build-with-claude/fast-mode,
        sending speed=fast to Opus 4.7, Sonnet, or Haiku returns HTTP 400.
        """
        from hermes_cli.models import model_supports_fast_mode

        assert model_supports_fast_mode("claude-sonnet-4-6") is False
        assert model_supports_fast_mode("claude-sonnet-4.6") is False
        assert model_supports_fast_mode("claude-haiku-4-5") is False
        assert model_supports_fast_mode("claude-opus-4-7") is False
        assert model_supports_fast_mode("anthropic/claude-sonnet-4.6") is False
        assert model_supports_fast_mode("anthropic/claude-opus-4-7") is False

    def test_non_claude_models_not_anthropic_fast(self):
        """Non-Claude models should not be treated as Anthropic fast-mode."""
        from hermes_cli.models import _is_anthropic_fast_model

        assert _is_anthropic_fast_model("gpt-5.4") is False
        assert _is_anthropic_fast_model("gemini-3-pro") is False
        assert _is_anthropic_fast_model("kimi-k2-thinking") is False

    def test_anthropic_variant_tags_stripped(self):
        from hermes_cli.models import model_supports_fast_mode

        # OpenRouter variant tags after colon should be stripped
        assert model_supports_fast_mode("claude-opus-4.6:fast") is True
        assert model_supports_fast_mode("claude-opus-4.6:beta") is True

    def test_resolve_overrides_returns_speed_for_anthropic(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        result = resolve_fast_mode_overrides("claude-opus-4-6")
        assert result == {"speed": "fast"}

        result = resolve_fast_mode_overrides("anthropic/claude-opus-4.6")
        assert result == {"speed": "fast"}

    def test_resolve_overrides_returns_none_for_unsupported_claude(self):
        """Opus 4.7 and other Claude models don't support fast mode (API 400s).

        Per Anthropic docs, fast mode is currently Opus 4.6 only.
        """
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("claude-opus-4-7") is None
        assert resolve_fast_mode_overrides("claude-sonnet-4-6") is None
        assert resolve_fast_mode_overrides("claude-haiku-4-5") is None

    def test_resolve_overrides_returns_service_tier_for_openai(self):
        """OpenAI models should still get service_tier, not speed."""
        from hermes_cli.models import resolve_fast_mode_overrides

        result = resolve_fast_mode_overrides("gpt-5.4")
        assert result == {"service_tier": "priority"}

    def test_is_anthropic_fast_model(self):
        """Fast mode is currently Opus 4.6 only — other Claude variants must be excluded."""
        from hermes_cli.models import _is_anthropic_fast_model

        # Supported: Opus 4.6 in any form
        assert _is_anthropic_fast_model("claude-opus-4-6") is True
        assert _is_anthropic_fast_model("claude-opus-4.6") is True
        assert _is_anthropic_fast_model("anthropic/claude-opus-4-6") is True
        assert _is_anthropic_fast_model("claude-opus-4.6:fast") is True

        # Unsupported per Anthropic API contract — would 400 if we sent speed=fast
        assert _is_anthropic_fast_model("claude-opus-4-7") is False
        assert _is_anthropic_fast_model("claude-sonnet-4-6") is False
        assert _is_anthropic_fast_model("claude-haiku-4-5") is False

        # Non-Claude
        assert _is_anthropic_fast_model("gpt-5.4") is False
        assert _is_anthropic_fast_model("") is False

    def test_fast_command_exposed_for_anthropic_model(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-opus-4-6", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is True

    def test_fast_command_hidden_for_anthropic_sonnet(self):
        """Sonnet doesn't support fast mode (Opus 4.6 only) — /fast must be hidden."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-sonnet-4-6", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is False

    def test_fast_command_hidden_for_anthropic_opus_47(self):
        """Opus 4.7 doesn't support fast mode — /fast must be hidden."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-opus-4-7", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is False

    def test_fast_command_hidden_for_non_claude_non_openai(self):
        """Non-Claude, non-OpenAI models should not expose /fast."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="gemini", requested_provider="gemini",
            model="gemini-3-pro-preview", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is False

    def test_turn_route_injects_speed_for_anthropic(self):
        """Anthropic models should get speed:'fast' override, not service_tier."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            api_mode="anthropic_messages",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            service_tier="priority",
        )

        route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        assert route["runtime"]["provider"] == "anthropic"
        assert route["request_overrides"] == {"speed": "fast"}


class TestAnthropicFastModeAdapter(unittest.TestCase):
    """Verify build_anthropic_kwargs handles fast_mode parameter."""

    def test_fast_mode_adds_speed_and_beta(self):
        from agent.anthropic_format import build_anthropic_kwargs, _FAST_MODE_BETA

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
        )
        assert kwargs.get("extra_body", {}).get("speed") == "fast"
        assert "speed" not in kwargs
        assert "extra_headers" in kwargs
        assert _FAST_MODE_BETA in kwargs["extra_headers"].get("anthropic-beta", "")

    def test_fast_mode_off_no_speed(self):
        from agent.anthropic_format import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=False,
        )
        assert kwargs.get("extra_body", {}).get("speed") is None
        assert "speed" not in kwargs
        assert "extra_headers" not in kwargs

    def test_fast_mode_skipped_for_third_party_endpoint(self):
        from agent.anthropic_format import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
            base_url="https://api.minimax.io/anthropic/v1",
        )
        # Third-party endpoints should NOT get speed or fast-mode beta
        assert kwargs.get("extra_body", {}).get("speed") is None
        assert "speed" not in kwargs
        assert "extra_headers" not in kwargs

    def test_fast_mode_kwargs_are_safe_for_sdk_unpacking(self):
        from agent.anthropic_format import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
        )
        assert "speed" not in kwargs
        assert kwargs.get("extra_body", {}).get("speed") == "fast"
