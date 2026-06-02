"""AWS Bedrock Converse API transport.

Delegates to the existing adapter functions in hermes_agent_bedrock.
Bedrock uses its own boto3 client (not the OpenAI SDK), so the transport
owns format conversion and normalization, while client construction and
boto3 calls stay on AIAgent.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage


class BedrockTransport(ProviderTransport):
    """Transport for api_mode='bedrock_converse'."""

    @property
    def api_mode(self) -> str:
        return "bedrock_converse"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI messages to Bedrock Converse format."""
        from agent.plugin_registries import registries
        _fn = registries.get_provider_service("bedrock", "convert_messages_to_converse")
        if _fn is None:
            raise ImportError("bedrock plugin not registered")
        return _fn(messages)

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Bedrock Converse toolConfig."""
        from agent.plugin_registries import registries
        _fn = registries.get_provider_service("bedrock", "convert_tools_to_converse")
        if _fn is None:
            raise ImportError("bedrock plugin not registered")
        return _fn(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Bedrock converse() kwargs."""
        from agent.plugin_registries import registries
        _fn = registries.get_provider_service("bedrock", "build_converse_kwargs")
        if _fn is None:
            raise ImportError("bedrock plugin not registered")

        region = params.get("region", "us-east-1")
        guardrail = params.get("guardrail_config")

        kwargs = _fn(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=params.get("max_tokens", 4096),
            temperature=params.get("temperature"),
            guardrail_config=guardrail,
        )
        # Sentinel keys for dispatch — agent pops these before the boto3 call
        kwargs["__bedrock_converse__"] = True
        kwargs["__bedrock_region__"] = region
        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Bedrock response to NormalizedResponse."""
        from agent.plugin_registries import registries
        normalize_converse_response = registries.get_provider_service("bedrock", "normalize_converse_response")
        if normalize_converse_response is None:
            raise ImportError("bedrock plugin not registered")

        if hasattr(response, "choices") and response.choices:
            ns = response
        else:
            ns = normalize_converse_response(response)

        choice = ns.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg.tool_calls
            ]

        usage = None
        if hasattr(ns, "usage") and ns.usage:
            u = ns.usage
            usage = Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

        return NormalizedResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning=reasoning,
            usage=usage,
        )

    def validate_response(self, response: Any) -> bool:
        if response is None:
            return False
        if isinstance(response, dict):
            return "output" in response
        if hasattr(response, "choices"):
            return bool(response.choices)
        return False

    def map_finish_reason(self, raw_reason: str) -> str:
        _MAP = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "guardrail_intervened": "content_filter",
            "content_filtered": "content_filter",
        }
        return _MAP.get(raw_reason, "stop")
