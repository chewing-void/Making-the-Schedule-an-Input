"""Protocol errors — spec §10.

Every safety assertion aborts; none of them repairs state in the running
process.  The exception carries the structured context that the abort dump
needs (actor, tag, slot, state, generation, reader count).
"""

from __future__ import annotations

from typing import Any


class ProtocolError(RuntimeError):
    """A safety assertion from spec §10 fired, or a watchdog expired."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        self.code = code
        self.context = context
        rendered = " ".join("%s=%r" % kv for kv in sorted(context.items()))
        super().__init__("[%s] %s | %s" % (code, message, rendered))


def require(condition: Any, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise ProtocolError(code, message, **context)
