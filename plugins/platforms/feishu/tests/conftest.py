"""Shared fixtures for feishu plugin tests."""

import sys
from pathlib import Path

# Make feishu_helpers importable from this test directory
sys.path.insert(0, str(Path(__file__).parent))
