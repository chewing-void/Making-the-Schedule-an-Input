"""Control channels — spec §1 (C1/C3/C4) and §7.

Three small shared objects replace the three busy-wait flags of the current
code:

``LtcBoard``   the mapper's logical-tag-complete frontier (replaces polling
              ``target_gaussians_ready``).  A gate waits for a *frontier*, then
              reads an outcome; the two steps are separate on purpose, so
              "nothing was published for this tag" is a value, not a timeout.
``CreditBoard`` the mapper's per-block work credit record, ``MAP_BLOCK_DONE``
              (replaces the free-running ``train_iter`` counter).
``StopBoard`` the three-phase end-of-stream handshake of spec §7, with the
              tracker fixed as the coordinator.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .errors import require
from .shared import Abort, CondObject
from .tags import Tag, make_tag

L_KNOWN, L_T, L_B, L_PHASE = range(4)
LTC_FIELDS = 4

CR_KNOWN, CR_K, CR_KEY, CR_SPARE = range(4)
CREDIT_FIELDS = 4

ST_REQ, ST_REQ_T, ST_REQ_B, ST_REQ_PHASE, ST_GRANT, ST_T, ST_B, ST_PHASE = range(8)
STOP_FIELDS = 8


class LtcBoard(CondObject):
    def __init__(self, ctx: Any, abort: Abort, watchdog_s: float = 180.0) -> None:
        super().__init__(ctx, abort, watchdog_s=watchdog_s)
        self.state = torch.zeros(LTC_FIELDS, dtype=torch.int64).share_memory_()

    def advance(self, value: Tag, at: Tag) -> None:
        """Publish frontier ``value``; the event itself is emitted at ``at``.

        The two differ by one phase: spec §2 has the mapper publish
        ``LTC_m((t,b,P4))`` during ``(t,b,P5_M_LTC)``.
        """
        with self.cond:
            if int(self.state[L_KNOWN].item()):
                current = (
                    int(self.state[L_T].item()),
                    int(self.state[L_B].item()),
                    int(self.state[L_PHASE].item()),
                )
                require(
                    tuple(value) > current, "LTC_NOT_MONOTONE",
                    "LTC moved backwards or repeated", current=current, proposed=tuple(value))
            self.state[L_KNOWN] = 1
            self.state[L_T] = value.t
            self.state[L_B] = value.b
            self.state[L_PHASE] = value.phase
            self.cond.notify_all()
        self._record(at, "LTC", generation=value.t)

    def current(self) -> Optional[Tag]:
        with self.cond:
            if not int(self.state[L_KNOWN].item()):
                return None
            return make_tag(
                int(self.state[L_T].item()),
                int(self.state[L_B].item()),
                int(self.state[L_PHASE].item()))

    def wait_ge(self, tag: Tag) -> None:
        deadline = self._deadline()
        with self.cond:
            while True:
                if int(self.state[L_KNOWN].item()):
                    current = (
                        int(self.state[L_T].item()),
                        int(self.state[L_B].item()),
                        int(self.state[L_PHASE].item()),
                    )
                    if current >= tuple(tag):
                        return
                self._tick(deadline, "LTC_BLOCKED", waiting_for=tuple(tag))


class CreditBoard(CondObject):
    def __init__(
        self,
        ctx: Any,
        abort: Abort,
        blocks_per_frame: int = 1,
        capacity: int = 4096,
        watchdog_s: float = 180.0,
    ) -> None:
        super().__init__(ctx, abort, watchdog_s=watchdog_s)
        self.blocks_per_frame = int(blocks_per_frame)
        self.capacity = int(capacity)
        self.records = torch.zeros((self.capacity, CREDIT_FIELDS), dtype=torch.int64).share_memory_()

    def _key(self, t: int, b: int) -> int:
        return t * self.blocks_per_frame + b

    def done(self, tag: Tag, k: int) -> None:
        key = self._key(tag.t, tag.b)
        index = key % self.capacity
        with self.cond:
            require(
                not (int(self.records[index, CR_KNOWN].item())
                     and int(self.records[index, CR_KEY].item()) == key),
                "CREDIT_REPUBLISH", "credit record already written for this block",
                t=tag.t, b=tag.b)
            self.records[index, CR_KNOWN] = 1
            self.records[index, CR_K] = int(k)
            self.records[index, CR_KEY] = key
            self.cond.notify_all()
        self._record(tag, "MAP_BLOCK_DONE", k=k)

    def get(self, t: int, b: int) -> Optional[int]:
        key = self._key(t, b)
        index = key % self.capacity
        with self.cond:
            if int(self.records[index, CR_KNOWN].item()) and int(
                    self.records[index, CR_KEY].item()) == key:
                return int(self.records[index, CR_K].item())
            return None


class StopBoard(CondObject):
    """Three-phase EOS of spec §7; the tracker is the fixed coordinator."""

    def __init__(self, ctx: Any, abort: Abort, watchdog_s: float = 180.0) -> None:
        super().__init__(ctx, abort, watchdog_s=watchdog_s)
        self.state = torch.zeros(STOP_FIELDS, dtype=torch.int64).share_memory_()

    def request(self, proposed: Tag, tag: Tag) -> None:
        with self.cond:
            self.state[ST_REQ] = 1
            self.state[ST_REQ_T] = proposed.t
            self.state[ST_REQ_B] = proposed.b
            self.state[ST_REQ_PHASE] = proposed.phase
            self.cond.notify_all()
        self._record(tag, "STOP_REQUEST", generation=proposed.t)

    def requested(self) -> Optional[Tag]:
        with self.cond:
            if not int(self.state[ST_REQ].item()):
                return None
            return make_tag(
                int(self.state[ST_REQ_T].item()),
                int(self.state[ST_REQ_B].item()),
                int(self.state[ST_REQ_PHASE].item()))

    def wait_request(self) -> Tag:
        deadline = self._deadline()
        with self.cond:
            while True:
                if int(self.state[ST_REQ].item()):
                    return make_tag(
                        int(self.state[ST_REQ_T].item()),
                        int(self.state[ST_REQ_B].item()),
                        int(self.state[ST_REQ_PHASE].item()))
                self._tick(deadline, "STOP_REQUEST_BLOCKED")

    def grant(self, stop_tag: Tag, tag: Tag) -> None:
        with self.cond:
            require(
                int(self.state[ST_REQ].item()) == 1, "GRANT_WITHOUT_REQUEST",
                "STOP_GRANT issued before any STOP_REQUEST")
            self.state[ST_GRANT] = 1
            self.state[ST_T] = stop_tag.t
            self.state[ST_B] = stop_tag.b
            self.state[ST_PHASE] = stop_tag.phase
            self.cond.notify_all()
        self._record(tag, "STOP_GRANT", generation=stop_tag.t)

    def wait_grant(self) -> Tag:
        deadline = self._deadline()
        with self.cond:
            while True:
                if int(self.state[ST_GRANT].item()):
                    return make_tag(
                        int(self.state[ST_T].item()),
                        int(self.state[ST_B].item()),
                        int(self.state[ST_PHASE].item()))
                self._tick(deadline, "STOP_GRANT_BLOCKED")

    def granted(self) -> Optional[Tag]:
        with self.cond:
            if not int(self.state[ST_GRANT].item()):
                return None
            return make_tag(
                int(self.state[ST_T].item()),
                int(self.state[ST_B].item()),
                int(self.state[ST_PHASE].item()))
