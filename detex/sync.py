"""Synchronisation paths — spec §4.

One run picks exactly one path and records the choice in the environment
fingerprint; switching mid-run is not allowed.

``Path-M`` (the MVP): the producer finishes every payload write, calls
``torch.cuda.synchronize()``, and only then release-stores the commit word.  A
consumer that acquire-loads the commit word may read the payload immediately,
because completion is already guaranteed.

``Path-E`` (the optimisation, spec §13 item 1): the producer records a
cross-process CUDA IPC event, publishes the commit word carrying a reference to
it, and the consumer stream-waits that event before reading.  Deferred until
Path-M's cost is measured, so it raises rather than pretending to work.

The ACK direction is shared by both paths and is enforced in
``Channel.last_ack``: the consumer-done event must actually be complete before
the release-ACK, or the producer may overwrite a read that is still in flight.

Both blocking calls are timed.  Spec §5 bans wall clock from *gate predicates*,
and these counters are outside every predicate: they wrap the two blocking calls
and feed a side file that nothing reads back, so no control flow can depend on
them.  They are always on because a cost that is only measured when someone
remembers to switch it on is a cost nobody has.  The claim that the timing is
inert is not an argument — the refactor-equivalence gate re-runs frozen
configurations and requires the recorded digests to be unchanged.

The split matters for what Path-E could buy.  ``producer_publish`` is the block
Path-E removes: the producer would record an event and carry on instead of
waiting.  ``wait_done`` is the ACK direction, shared by both paths, so it is
*not* recoverable — counting them together would overstate the prize.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch


class SyncPath:
    def __init__(self, kind: str = "M") -> None:
        if kind not in ("M", "E"):
            raise ValueError("sync path must be 'M' or 'E', got %r" % (kind,))
        if kind == "E":
            raise NotImplementedError(
                "Path-E (CUDA IPC event) is deferred; see spec §4 and §13 item 1")
        self.kind = kind
        # Cost accounting; see the module docstring for why this is not a
        # wall-clock dependency.  One SyncPath is pickled into each child, so
        # each actor accumulates only its own blocking.
        self.publish_calls = 0        # every commit, CUDA-domain or not
        self.publish_sync_calls = 0   # the subset that actually blocked
        self.publish_s = 0.0
        self.publish_max_s = 0.0
        self.wait_calls = 0
        self.wait_s = 0.0
        self.wait_max_s = 0.0

    def producer_publish(self, has_cuda: bool) -> None:
        """Called after the last payload write and before the commit word."""
        self.publish_calls += 1
        if not has_cuda:
            return
        started = time.perf_counter()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.publish_sync_calls += 1
        self.publish_s += elapsed
        if elapsed > self.publish_max_s:
            self.publish_max_s = elapsed

    def consumer_done_event(self, has_cuda: bool) -> Optional[Any]:
        """Record the completion of this consumer's reads of the payload."""
        if not has_cuda:
            return None
        event = torch.cuda.Event()
        event.record()
        return event

    def wait_done(self, event: Optional[Any]) -> None:
        if event is None:
            return
        started = time.perf_counter()
        event.synchronize()
        elapsed = time.perf_counter() - started
        self.wait_calls += 1
        self.wait_s += elapsed
        if elapsed > self.wait_max_s:
            self.wait_max_s = elapsed

    def to_dict(self):
        return {"sync_path": self.kind}

    def cost(self):
        """What this actor blocked on, in seconds.

        ``publish_*`` is Path-M's own cost and the ceiling on what Path-E could
        return; ``wait_*`` belongs to the ACK direction and would survive the
        switch.  ``publish_calls`` minus ``publish_sync_calls`` is the CPU-domain
        commits, which never block on either path.
        """
        return {
            "sync_path": self.kind,
            "publish_calls": self.publish_calls,
            "publish_sync_calls": self.publish_sync_calls,
            "publish_s": round(self.publish_s, 6),
            "publish_max_s": round(self.publish_max_s, 6),
            "wait_calls": self.wait_calls,
            "wait_s": round(self.wait_s, 6),
            "wait_max_s": round(self.wait_max_s, 6),
        }
