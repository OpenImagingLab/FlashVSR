"""Low-overhead runtime route and fallback telemetry for FlashVSR.

Enable with ``FLASHVSR_TELEMETRY=1``. The profiling target enables it
automatically when ``FLASHVSR_REQUIRE_FASTPATHS=1`` is requested.
"""

import os
from collections import Counter


_ENABLED = (
    os.environ.get("FLASHVSR_TELEMETRY", "0") != "0"
    or os.environ.get("FLASHVSR_REQUIRE_FASTPATHS", "0") != "0"
)
_COUNTS = Counter()
_ERRORS = {}


def enabled():
    return _ENABLED


def record(name, count=1):
    if _ENABLED:
        _COUNTS[name] += count


def record_error(name, error):
    if not _ENABLED:
        return
    _COUNTS[f"{name}_error"] += 1
    _ERRORS.setdefault(name, f"{type(error).__name__}: {error}")


def reset(preserve_errors=False):
    _COUNTS.clear()
    if not preserve_errors:
        _ERRORS.clear()


def snapshot():
    return {
        "counts": dict(sorted(_COUNTS.items())),
        "errors": dict(sorted(_ERRORS.items())),
    }
