"""Seeded perturbation injection — spec §12, gate G-PERT.

A contract that only holds when the two processes happen to run at their usual
speeds has not removed the schedule from the result, it has only hidden it.
The perturber inserts seeded sleeps at gate boundaries so that the interleaving
is genuinely different run to run; L1, L2 and the end state must not notice.

Conclusions from a perturbation campaign are limited to the declared support
set — the perturbations actually applied — and the report says which those
were.  Sleep injection and start-order swap do not, for instance, cover a card
whose kernels reorder under contention.
"""

from __future__ import annotations

import random
import time
from typing import Optional


class Perturber:
    def __init__(self, seed: Optional[int], max_ms: float, actor: int) -> None:
        self.seed = seed
        self.max_ms = float(max_ms)
        self.actor = int(actor)
        self.enabled = seed is not None and self.max_ms > 0.0
        self._rng = random.Random((seed or 0) * 1000 + self.actor)
        self.hits = 0
        self.slept_s = 0.0

    def hit(self) -> None:
        if not self.enabled:
            return
        delay = self._rng.uniform(0.0, self.max_ms) / 1000.0
        self.hits += 1
        self.slept_s += delay
        time.sleep(delay)

    def to_dict(self):
        """Evidence, not configuration.

        A perturbation gate that only records "the switch was on" cannot tell a
        real perturbation campaign from one whose injection points never fired.
        ``hits`` and ``slept_s`` are what the gate asserts on.
        """
        return {"seed": self.seed, "max_ms": self.max_ms, "enabled": self.enabled,
                "actor": self.actor, "hits": self.hits,
                "slept_s": round(self.slept_s, 4)}
