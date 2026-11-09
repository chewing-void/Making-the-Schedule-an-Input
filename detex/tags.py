"""Event addressing — spec §2.

``EventId = tag = (t, b, phase)`` with a lexicographic total order.  The phase
table is static: which actor is allowed to emit in which phase is fixed at
import time, so a misplaced event is a protocol error rather than a race.
"""

from __future__ import annotations

from typing import NamedTuple, Union

TRACKER = 0
MAPPER = 1
ACTORS = ("tracker", "mapper")

PHASES = (
    "P0_T_CONSUME",   # tracker resolves u(t), DROP_ACKs skipped gens, CLAIMs u(t)
    "P1_T_PUBLISH",   # tracker publishes the keyframe/camera outcome
    "P2_M_INGEST",    # mapper claims and ingests the keyframe/camera
    "P3_M_WORK",      # mapper burns exactly m(t,b) credits
    "P4_M_PUBLISH",   # mapper publishes the target outcome (for t+1)
    "P5_M_LTC",       # mapper advances the logical tag complete frontier
    "P6_T_COMMIT",    # tracker commits frame t
)
PHASE_INDEX = {name: i for i, name in enumerate(PHASES)}
PHASE_ACTOR = {
    "P0_T_CONSUME": TRACKER,
    "P1_T_PUBLISH": TRACKER,
    "P2_M_INGEST": MAPPER,
    "P3_M_WORK": MAPPER,
    "P4_M_PUBLISH": MAPPER,
    "P5_M_LTC": MAPPER,
    "P6_T_COMMIT": TRACKER,
}


class Tag(NamedTuple):
    """A fully qualified event address.  Ordering is the tuple ordering."""

    t: int
    b: int
    phase: int

    def __str__(self) -> str:
        return "(%d,%d,%s)" % (self.t, self.b, PHASES[self.phase])

    @property
    def phase_name(self) -> str:
        return PHASES[self.phase]

    @property
    def actor(self) -> int:
        return PHASE_ACTOR[PHASES[self.phase]]


def make_tag(t: int, b: int, phase: Union[int, str]) -> Tag:
    if isinstance(phase, str):
        phase = PHASE_INDEX[phase]
    return Tag(int(t), int(b), int(phase))


def parse_tag(text: str) -> Tag:
    """Inverse of the ``t,b,phase`` field written into the L1 trace."""
    t, b, phase = text.split(",")
    return make_tag(int(t), int(b), phase)
