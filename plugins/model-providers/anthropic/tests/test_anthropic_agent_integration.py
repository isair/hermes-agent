"""Integration tests for Anthropic-specific AIAgent behaviour.

Tests that exercise the interaction between AIAgent and the Anthropic
provider plugin — covering max_tokens passthrough, image fallback,
provider fallback routing, base-url passthrough, credential refresh,
and OAuth flag setting.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent_anthropic.adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token

import run_agent
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


class TestBuildApiKwargsAnthropicMaxTokens:
    """Bug fix: max_tokens was always None for Anthropic mode, ignoring user config."""

    def test_max_tokens_passed_to_anthropic(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.max_tokens = 4096
        agent.reasoning_config = None

        with patch("agent.transports.anthropic.build_anthropic_kwargs") as mock_build:
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs([{"role": "user", "content": "test"}])
            _, kwargs = mock_build.call_args
            if not kwargs:
                kwargs = dict(zip(
                    ["model", "messages", "tools", "max_tokens", "reasoning_config"],
                    mock_build.call_args[0],
                ))
            assert kwargs.get("max_tokens") == 4096 or mock_build.call_args[1].get("max_tokens") == 4096

    def test_max_tokens_none_when_unset(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.max_tokens = None
        agent.reasoning_config = None

        with patch("agent.transports.anthropic.build_anthropic_kwargs") as mock_build:
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 16384}
            agent._build_api_kwargs([{"role": "user", "content": "test"}])
            call_args = mock_build.call_args
            # max_tokens should be None (let adapter use its default)
            if call_args[1]:
                assert call_args[1].get("max_tokens") is None
            else:
                assert call_args[0][3] is None


class TestAnthropicImageFallback:
    def test_build_api_kwargs_converts_multimodal_user_image_to_text(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.reasoning_config = None

        api_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you see this now?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }]

        with (
            patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=json.dumps({"success": True, "analysis": "A cat sitting on a chair."}))),
            patch("agent.transports.anthropic.build_anthropic_kwargs") as mock_build,
        ):
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs(api_messages)

        kwargs = mock_build.call_args.kwargs or dict(zip(
            ["model", "messages", "tools", "max_tokens", "reasoning_config"],
            mock_build.call_args.args,
        ))
        transformed = kwargs["messages"]
        assert isinstance(transformed[0]["content"], str)
        assert "A cat sitting on a chair." in transformed[0]["content"]
        assert "Can you see this now?" in transformed[0]["content"]
        assert "vision_analyze with image_url: https://example.com/cat.png" in transformed[0]["content"]

    def test_build_api_kwargs_reuses_cached_image_analysis_for_duplicate_images(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.reasoning_config = None
        data_url = "data:image/png;base64,QUFBQQ=="

        api_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ]

        mock_vision = AsyncMock(return_value=json.dumps({"success": True, "analysis": "A small test image."}))
        with (
            patch("tools.vision_tools.vision_analyze_tool", new=mock_vision),
            patch("agent.transports.anthropic.build_anthropic_kwargs") as mock_build,
        ):
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs(api_messages)

        assert mock_vision.await_count == 1


class TestFallbackAnthropicProvider:
    """Bug fix: _try_activate_fallback had no case for anthropic provider."""

    def test_fallback_to_anthropic_sets_api_mode(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "***"

        with (
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client") as mock_build,
            patch("hermes_agent_anthropic.adapter.resolve_anthropic_token", return_value=None),
        ):
            mock_build.return_value = MagicMock()
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "anthropic_messages"
        assert agent._anthropic_client is not None
        assert agent.client is None

    def test_fallback_to_anthropic_enables_prompt_caching(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "***"

        with (
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client", return_value=MagicMock()),
            patch("hermes_agent_anthropic.adapter.resolve_anthropic_token", return_value=None),
        ):
            agent._try_activate_fallback()

        assert agent._use_prompt_caching is True

    def test_fallback_to_openrouter_uses_openai_client(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://openrouter.ai/api/v1"
        mock_client.api_key = "sk-or-test"

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "chat_completions"
        assert agent.client is mock_client


class TestAnthropicBaseUrlPassthrough:
    """Bug fix: base_url was filtered with 'anthropic in base_url', blocking proxies."""

    def test_custom_proxy_base_url_passed_through(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            a = AIAgent(
                api_key="sk-ant...7890",
                base_url="https://llm-proxy.company.com/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            call_args = mock_build.call_args
            # base_url should be passed through, not filtered out
            assert call_args[0][1] == "https://llm-proxy.company.com/v1"

    def test_none_base_url_passed_as_none(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            a = AIAgent(
                api_key="sk-ant...7890",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            call_args = mock_build.call_args
            # No base_url provided, should be default empty string or None
            passed_url = call_args[0][1]
            assert not passed_url or passed_url is None


class TestAnthropicCredentialRefresh:
    def test_try_refresh_anthropic_client_credentials_rebuilds_client(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client") as mock_build,
        ):
            old_client = MagicMock()
            new_client = MagicMock()
            mock_build.side_effect = [old_client, new_client]
            agent = AIAgent(
                api_key="sk-ant...oken",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        agent._anthropic_client = old_client
        agent._anthropic_api_key = "sk-ant...old-token"   # differs from what resolve returns
        agent._anthropic_base_url = "https://api.anthropic.com"
        agent.provider = "anthropic"

        with (
            patch("hermes_agent_anthropic.adapter.resolve_anthropic_token", return_value="sk-ant...oken"),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client", return_value=new_client) as rebuild,
        ):
            assert agent._try_refresh_anthropic_client_credentials() is True

        old_client.close.assert_called_once()
        rebuild.assert_called_once_with(
            "sk-ant...oken", "https://api.anthropic.com", timeout=None,
        )
        assert agent._anthropic_client is new_client
        assert agent._anthropic_api_key == "sk-ant...oken"

    def test_try_refresh_anthropic_client_credentials_returns_false_when_token_unchanged(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant...oken",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        old_client = MagicMock()
        agent._anthropic_client = old_client
        agent._anthropic_api_key = "sk-ant...oken"

        with (
            patch("hermes_agent_anthropic.adapter.resolve_anthropic_token", return_value="sk-ant...oken"),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client") as rebuild,
        ):
            assert agent._try_refresh_anthropic_client_credentials() is False

        old_client.close.assert_not_called()
        rebuild.assert_not_called()

    def test_anthropic_messages_create_preflights_refresh(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant...oken",
                base_url="https://openrouter.ai/api/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        response = SimpleNamespace(content=[])
        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.create.return_value = response

        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=True) as refresh:
            result = agent._anthropic_messages_create({"model": "claude-sonnet-4-20250514"})

        refresh.assert_called_once_with()
        agent._anthropic_client.messages.create.assert_called_once_with(model="claude-sonnet-4-20250514")
        assert result is response


class TestFallbackSetsOAuthFlag:
    """_try_activate_fallback must set _is_anthropic_oauth for Anthropic fallbacks."""

    def test_fallback_to_anthropic_oauth_sets_flag(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-6"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-setup-oauth-token"

        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(mock_client, None)),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client",
                  return_value=MagicMock()),
            patch("hermes_agent_anthropic.adapter.resolve_anthropic_token",
                  return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent._is_anthropic_oauth is True

    def test_fallback_to_anthropic_api_key_clears_flag(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-6"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-api03-regular-key"

        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(mock_client, None)),
            patch("hermes_agent_anthropic.adapter.build_anthropic_client",
                  return_value=MagicMock()),
            patch("hermes_agent_anthropic.adapter.resolve_anthropic_token",
                  return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent._is_anthropic_oauth is False


class TestOAuthFlagAfterCredentialRefresh:
    """_is_anthropic_oauth must update when token type changes during refresh."""

    def test_oauth_flag_updates_api_key_to_oauth(self, agent):
        """Refreshing from API key to OAuth token must set flag to True."""
        from agent.plugin_registries import registries
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "***"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        with patch.dict(registries._provider_services, {"anthropic": {
            "resolve_anthropic_token": MagicMock(return_value="sk-ant...oken"),
            "build_anthropic_client": MagicMock(return_value=MagicMock()),
            "_is_oauth_token": MagicMock(return_value=True),
        }}):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._is_anthropic_oauth is True

    def test_oauth_flag_updates_oauth_to_api_key(self, agent):
        """Refreshing from OAuth to API key must set flag to False."""
        from agent.plugin_registries import registries
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "***"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = True

        with patch.dict(registries._provider_services, {"anthropic": {
            "resolve_anthropic_token": MagicMock(return_value="sk-ant...-key"),
            "build_anthropic_client": MagicMock(return_value=MagicMock()),
            "_is_oauth_token": MagicMock(return_value=False),
        }}):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._is_anthropic_oauth is False
