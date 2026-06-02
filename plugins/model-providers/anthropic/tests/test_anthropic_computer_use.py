"""Anthropic-specific computer use tests moved from tests/tools/test_computer_use.py."""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Anthropic adapter: multimodal tool-result conversion
# ---------------------------------------------------------------------------

class TestAnthropicAdapterMultimodal:
    def test_multimodal_envelope_becomes_tool_result_with_image_block(self):
        from agent.anthropic_format import convert_messages_to_anthropic

        fake_png = "iVBORw0KGgo="
        messages = [
            {"role": "user", "content": "take a screenshot"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "computer_use", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": {
                    "_multimodal": True,
                    "content": [
                        {"type": "text", "text": "1 element"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{fake_png}"}},
                    ],
                    "text_summary": "1 element",
                },
            },
        ]
        _, anthropic_msgs = convert_messages_to_anthropic(messages)
        tool_result_msgs = [m for m in anthropic_msgs if m["role"] == "user"
                            and isinstance(m["content"], list)
                            and any(b.get("type") == "tool_result" for b in m["content"])]
        assert tool_result_msgs, "expected a tool_result user message"
        tr = next(b for b in tool_result_msgs[-1]["content"] if b.get("type") == "tool_result")
        inner = tr["content"]
        assert any(b.get("type") == "image" for b in inner)
        assert any(b.get("type") == "text" for b in inner)

    def test_old_screenshots_are_evicted_beyond_max_keep(self):
        """Image blocks in old tool_results get replaced with placeholders."""
        from agent.anthropic_format import convert_messages_to_anthropic

        fake_png = "iVBORw0KGgo="

        def _mm_tool(call_id: str) -> Dict[str, Any]:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": {
                    "_multimodal": True,
                    "content": [
                        {"type": "text", "text": "cap"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{fake_png}"}},
                    ],
                    "text_summary": "cap",
                },
            }

        # Build 5 screenshots interleaved with assistant messages.
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "start"}]
        for i in range(5):
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "computer_use", "arguments": "{}"},
                }],
            })
            messages.append(_mm_tool(f"call_{i}"))
        messages.append({"role": "assistant", "content": "done"})

        _, anthropic_msgs = convert_messages_to_anthropic(messages)

        # Walk tool_result blocks in order; the OLDEST (5 - 3) = 2 should be
        # text-only placeholders, newest 3 should still carry image blocks.
        tool_results = []
        for m in anthropic_msgs:
            if m["role"] != "user" or not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    tool_results.append(b)

        assert len(tool_results) == 5
        with_images = [
            b for b in tool_results
            if isinstance(b.get("content"), list)
            and any(x.get("type") == "image" for x in b["content"])
        ]
        placeholders = [
            b for b in tool_results
            if isinstance(b.get("content"), list)
            and any(
                x.get("type") == "text"
                and "screenshot removed" in x.get("text", "")
                for x in b["content"]
            )
        ]
        assert len(with_images) == 3
        assert len(placeholders) == 2

    def test_content_parts_helper_filters_to_text_and_image(self):
        from agent.anthropic_format import _content_parts_to_anthropic_blocks

        fake_png = "iVBORw0KGgo="
        blocks = _content_parts_to_anthropic_blocks([
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fake_png}"}},
            {"type": "unsupported", "data": "ignored"},
        ])
        types = [b["type"] for b in blocks]
        assert "text" in types
        assert "image" in types
        assert len(blocks) == 2
