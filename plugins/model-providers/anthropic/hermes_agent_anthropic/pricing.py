"""Anthropic model pricing data.

Official docs snapshot entries for Anthropic Claude models.
Source: https://platform.claude.com/docs/en/about-claude/pricing
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List


def get_anthropic_pricing_entries() -> list:
    """Return official docs pricing entries for Anthropic Claude models."""
    from agent.usage_pricing import PricingEntry

    _ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
    _ANTHROPIC_PRICING_VER = "anthropic-pricing-2026-05"

    return [
        PricingEntry(
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("25.00"),
            cache_read_cost_per_million=Decimal("0.50"),
            cache_write_cost_per_million=Decimal("6.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-opus-4-7")
        PricingEntry(
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("25.00"),
            cache_read_cost_per_million=Decimal("0.50"),
            cache_write_cost_per_million=Decimal("6.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-opus-4-6")
        PricingEntry(
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("25.00"),
            cache_read_cost_per_million=Decimal("0.50"),
            cache_write_cost_per_million=Decimal("6.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-opus-4-5")
        PricingEntry(
            input_cost_per_million=Decimal("3.00"),
            output_cost_per_million=Decimal("15.00"),
            cache_read_cost_per_million=Decimal("0.30"),
            cache_write_cost_per_million=Decimal("3.75"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-sonnet-4-7")
        PricingEntry(
            input_cost_per_million=Decimal("3.00"),
            output_cost_per_million=Decimal("15.00"),
            cache_read_cost_per_million=Decimal("0.30"),
            cache_write_cost_per_million=Decimal("3.75"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-sonnet-4-6")
        PricingEntry(
            input_cost_per_million=Decimal("3.00"),
            output_cost_per_million=Decimal("15.00"),
            cache_read_cost_per_million=Decimal("0.30"),
            cache_write_cost_per_million=Decimal("3.75"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-sonnet-4-5")
        PricingEntry(
            input_cost_per_million=Decimal("0.80"),
            output_cost_per_million=Decimal("4.00"),
            cache_read_cost_per_million=Decimal("0.08"),
            cache_write_cost_per_million=Decimal("1.00"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-haiku-4-5")
        PricingEntry(
            input_cost_per_million=Decimal("1.00"),
            output_cost_per_million=Decimal("5.00"),
            cache_read_cost_per_million=Decimal("0.10"),
            cache_write_cost_per_million=Decimal("1.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-4-7-sonnet")
        PricingEntry(
            input_cost_per_million=Decimal("1.00"),
            output_cost_per_million=Decimal("5.00"),
            cache_read_cost_per_million=Decimal("0.10"),
            cache_write_cost_per_million=Decimal("1.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-4-6-sonnet")
        PricingEntry(
            input_cost_per_million=Decimal("3.00"),
            output_cost_per_million=Decimal("15.00"),
            cache_read_cost_per_million=Decimal("0.30"),
            cache_write_cost_per_million=Decimal("3.75"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-4-5-sonnet")
        PricingEntry(
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("25.00"),
            cache_read_cost_per_million=Decimal("0.50"),
            cache_write_cost_per_million=Decimal("6.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-4-7-opus")
        PricingEntry(
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("25.00"),
            cache_read_cost_per_million=Decimal("0.50"),
            cache_write_cost_per_million=Decimal("6.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-4-6-opus")
        PricingEntry(
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("25.00"),
            cache_read_cost_per_million=Decimal("0.50"),
            cache_write_cost_per_million=Decimal("6.25"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-4-5-opus")
        PricingEntry(
            input_cost_per_million=Decimal("0.80"),
            output_cost_per_million=Decimal("4.00"),
            cache_read_cost_per_million=Decimal("0.08"),
            cache_write_cost_per_million=Decimal("1.00"),
            source="official_docs_snapshot",
            source_url=_ANTHROPIC_PRICING_URL,
            pricing_version=_ANTHROPIC_PRICING_VER,
        ),  # key: ("anthropic", "claude-4-5-haiku")
    ]


# Model name keys for the pricing entries — must match the order above
ANTHROPIC_PRICING_KEYS = [
    ("anthropic", "claude-opus-4-7"),
    ("anthropic", "claude-opus-4-6"),
    ("anthropic", "claude-opus-4-5"),
    ("anthropic", "claude-sonnet-4-7"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-sonnet-4-5"),
    ("anthropic", "claude-haiku-4-5"),
    ("anthropic", "claude-4-7-sonnet"),
    ("anthropic", "claude-4-6-sonnet"),
    ("anthropic", "claude-4-5-sonnet"),
    ("anthropic", "claude-4-7-opus"),
    ("anthropic", "claude-4-6-opus"),
    ("anthropic", "claude-4-5-opus"),
    ("anthropic", "claude-4-5-haiku"),
]


def normalize_anthropic_model_name(model: str) -> str:
    """Normalize Anthropic model name variants to canonical form.

    Handles:
      - Dot notation: claude-opus-4.7 → claude-opus-4-7
      - Short aliases: claude-opus-4.7 → claude-opus-4-7
      - Strips anthropic/ prefix if present
    """
    import re
    name = model.lower().strip()
    if name.startswith("anthropic/"):
        name = name[len("anthropic/"):]
    # Normalize dots to dashes in version numbers
    name = re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)
    return name
