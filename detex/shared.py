"""Shared-memory plumbing common to channels and boards.

Two things live here.  ``Abort`` is the group-terminate flag of spec §10: any
safety assertion sets it, and every blocking wait in the package polls it so a
peer never hangs behind a dead partner.  ``CondObject`` is the base for every
shared object: one ``multiprocessing.Condition`` guarding one shared int64
tensor, plus the watchdog discipline of spec §5 — wall clock may abort a wait
but may never enter a semantic branch.

``BLOCKED`` accumulates how long this process sat in those waits, by reason.
It exists for the Path-M/Path-E cost question: the producer sync is only worth
removing if the actor paying it has something else to do, and an actor that is
already gate-bound behind its peer would just convert saved sync time into more
gate wait.  The counters are per process (spawn re-imports the module) and
nothing reads them back into a decision — they are dumped and that is all.
"""

from __future__ import annotations

import time
from typing import Any, List

import torch

from .errors import ProtocolError

WAIT_TICK_S = 0.02
DEFAULT_WATCHDOG_S = 180.0

BLOCKED: dict = {}


def blocked_report() -> dict:
    """Per-reason blocked time for this process, in seconds."""
    return {code: {"waits": waits, "seconds": round(seconds, 6)}
            for code, (waits, seconds) in sorted(BLOCKED.items())}


class Abort:
    """Group terminate flag, shared by every channel and board in a run."""

    def __init__(self, ctx: Any) -> None:
        self.flag = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._conds: List[Any] = []

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_conds"] = []
        return state

    def register(self, cond: Any) -> None:
        """Called per process, after spawn, by every object that can block."""
        self._conds.append(cond)

    def is_set(self) -> bool:
        return bool(self.flag[0].item())

    def set(self) -> None:
        self.flag[0] = 1
        for cond in self._conds:
            with cond:
                cond.notify_all()


class CondObject:
    """One condition variable over one shared control tensor."""

    def __init__(self, ctx: Any, abort: Abort, watchdog_s: float = DEFAULT_WATCHDOG_S):
        self.cond = ctx.Condition()
        self.abort = abort
        self.watchdog_s = float(watchdog_s)
        self.actor = None
        self.recorder = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["recorder"] = None
        state["actor"] = None
        return state

    def attach(self, actor: int, recorder: Any) -> None:
        """Bind this process's role and trace sink.  Call once after spawn."""
        self.actor = actor
        self.recorder = recorder
        self.abort.register(self.cond)

    def _deadline(self) -> float:
        return time.monotonic() + self.watchdog_s

    def _tick(self, deadline: float, code: str, **context: Any) -> None:
        """Block until notified, the watchdog fires, or the group aborts.

        Wall clock is read here and only here.  Its only effect on the run is
        to raise; the second read feeds ``BLOCKED``, which no caller may branch
        on.  How long the wait took never enters a semantic decision
        (spec §5, C8).
        """
        if self.abort.is_set():
            raise ProtocolError("PEER_ABORT", "group terminated by peer", **context)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise ProtocolError(code, "watchdog expired while blocked", **context)
        started = time.perf_counter()
        self.cond.wait(min(WAIT_TICK_S, remaining))
        entry = BLOCKED.get(code)
        if entry is None:
            BLOCKED[code] = [1, time.perf_counter() - started]
        else:
            entry[0] += 1
            entry[1] += time.perf_counter() - started

    def _record(self, tag, event: str, generation=None, k=None, digest=None) -> None:
        if self.recorder is not None:
            self.recorder.record(tag, event, generation=generation, k=k, digest=digest)
