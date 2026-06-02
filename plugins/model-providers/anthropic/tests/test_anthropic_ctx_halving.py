"""Anthropic-specific ctx halving tests moved from tests/test_ctx_halving_fix.py."""


# ---------------------------------------------------------------------------
# build_anthropic_kwargs — output cap clamping
# ---------------------------------------------------------------------------

class TestBuildAnthropicKwargsClamping:
    """The context_length clamp only fires when output ceiling > window.
    For standard Anthropic models (output ceiling < window) it must not fire.
    """

    def _build(self, model, max_tokens=None, context_length=None):
        from agent.anthropic_format import build_anthropic_kwargs
        return build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=max_tokens,
            reasoning_config=None,
            context_length=context_length,
        )

    def test_no_clamping_when_output_ceiling_fits_in_window(self):
        """Opus 4.6 native output (128K) < context window (200K) — no clamping."""
        kwargs = self._build("claude-opus-4-6", context_length=200_000)
        assert kwargs["max_tokens"] == 128_000

    def test_clamping_fires_for_tiny_custom_window(self):
        """When context_length is 8K (local model), output cap is clamped to 7999."""
        kwargs = self._build("claude-opus-4-6", context_length=8_000)
        assert kwargs["max_tokens"] == 7_999

    def test_explicit_max_tokens_respected_when_within_window(self):
        """Explicit max_tokens smaller than window passes through unchanged."""
        kwargs = self._build("claude-opus-4-6", max_tokens=4096, context_length=200_000)
        assert kwargs["max_tokens"] == 4096

    def test_explicit_max_tokens_clamped_when_exceeds_window(self):
        """Explicit max_tokens larger than a small window is clamped."""
        kwargs = self._build("claude-opus-4-6", max_tokens=32_768, context_length=16_000)
        assert kwargs["max_tokens"] == 15_999

    def test_no_context_length_uses_native_ceiling(self):
        """Without context_length the native output ceiling is used directly."""
        kwargs = self._build("claude-sonnet-4-6")
        assert kwargs["max_tokens"] == 64_000
