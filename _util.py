"""Small helpers for the lecture. lectrace does not trace files starting with _."""

import sys


def untraced(fn, *args, **kwargs):
    """Call fn with lectrace's tracer switched off, and switch it back on after.

    `# @stepover` hides a call from the viewer, but the tracer still runs its
    hook on every line, and training an MLP is tens of thousands of lines.
    This records nothing for the call either, it just skips the hook.
    """
    hook = sys.gettrace()
    sys.settrace(None)
    try:
        return fn(*args, **kwargs)
    finally:
        sys.settrace(hook)
