"""Bedrock model pricing data.

Official docs snapshot entries for AWS Bedrock models.
Source: https://aws.amazon.com/bedrock/pricing/
"""

from __future__ import annotations

from decimal import Decimal


def get_bedrock_pricing_entries() -> list:
    """Return official docs pricing entries for Bedrock models."""
    from agent.usage_pricing import PricingEntry

    _BEDROCK_PRICING_URL = "https://aws.amazon.com/bedrock/pricing/"
    _BEDROCK_PRICING_VER = "bedrock-pricing-2026-04"

    return [
        PricingEntry(
            input_cost_per_million=Decimal("15.00"),
            output_cost_per_million=Decimal("75.00"),
            source="official_docs_snapshot",
            source_url=_BEDROCK_PRICING_URL,
            pricing_version=_BEDROCK_PRICING_VER,
        ),  # ("bedrock", "anthropic.claude-opus-4-6")
        PricingEntry(
            input_cost_per_million=Decimal("3.00"),
            output_cost_per_million=Decimal("15.00"),
            source="official_docs_snapshot",
            source_url=_BEDROCK_PRICING_URL,
            pricing_version=_BEDROCK_PRICING_VER,
        ),  # ("bedrock", "anthropic.claude-sonnet-4-6")
        PricingEntry(
            input_cost_per_million=Decimal("3.00"),
            output_cost_per_million=Decimal("15.00"),
            source="official_docs_snapshot",
            source_url=_BEDROCK_PRICING_URL,
            pricing_version=_BEDROCK_PRICING_VER,
        ),  # ("bedrock", "anthropic.claude-sonnet-4-5")
        PricingEntry(
            input_cost_per_million=Decimal("0.80"),
            output_cost_per_million=Decimal("4.00"),
            source="official_docs_snapshot",
            source_url=_BEDROCK_PRICING_URL,
            pricing_version=_BEDROCK_PRICING_VER,
        ),  # ("bedrock", "anthropic.claude-haiku-4-5")
        PricingEntry(
            input_cost_per_million=Decimal("0.80"),
            output_cost_per_million=Decimal("3.20"),
            source="official_docs_snapshot",
            source_url=_BEDROCK_PRICING_URL,
            pricing_version=_BEDROCK_PRICING_VER,
        ),  # ("bedrock", "amazon.nova-pro")
        PricingEntry(
            input_cost_per_million=Decimal("0.06"),
            output_cost_per_million=Decimal("0.24"),
            source="official_docs_snapshot",
            source_url=_BEDROCK_PRICING_URL,
            pricing_version=_BEDROCK_PRICING_VER,
        ),  # ("bedrock", "amazon.nova-lite")
        PricingEntry(
            input_cost_per_million=Decimal("0.035"),
            output_cost_per_million=Decimal("0.14"),
            source="official_docs_snapshot",
            source_url=_BEDROCK_PRICING_URL,
            pricing_version=_BEDROCK_PRICING_VER,
        ),  # ("bedrock", "amazon.nova-micro")
    ]


BEDROCK_PRICING_KEYS = [
    ("bedrock", "anthropic.claude-opus-4-6"),
    ("bedrock", "anthropic.claude-sonnet-4-6"),
    ("bedrock", "anthropic.claude-sonnet-4-5"),
    ("bedrock", "anthropic.claude-haiku-4-5"),
    ("bedrock", "amazon.nova-pro"),
    ("bedrock", "amazon.nova-lite"),
    ("bedrock", "amazon.nova-micro"),
]
