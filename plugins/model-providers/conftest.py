"""Ensure the real `anthropic` SDK package is importable from plugin tests.

pytest adds parent directories containing ``__init__.py`` files to ``sys.path``.
``plugins/model-providers/anthropic/__init__.py`` (the provider profile) makes
``plugins/model-providers/`` appear in ``sys.path``, which means ``import anthropic``
resolves to ``plugins/model-providers/anthropic/`` rather than the installed
``anthropic`` SDK package.  This conftest removes that shadowing entry before
any tests run.
"""

import sys
from pathlib import Path

# Remove any sys.path entry that would shadow the real anthropic SDK with the
# provider-profile __init__.py living at plugins/model-providers/anthropic/.
_repo_root = Path(__file__).resolve().parent.parent.parent  # main/
_bad_entry = str(_repo_root / "plugins" / "model-providers")
if _bad_entry in sys.path:
    sys.path.remove(_bad_entry)
