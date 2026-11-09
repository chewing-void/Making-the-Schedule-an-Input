#!/usr/bin/env python3
"""The detex exchange fabric for monobeast — spec §1/§3, phase table §4.

ADDITIVE HOST FILE.  Nothing under ``torchbeast/`` and nothing under ``detex/``
is edited: this module declares a channel topology over payloads shaped by
upstream's own ``create_buffers`` and ``Net.state_dict``, and it subclasses two
detex objects host-side (the ``ConsumerAckChannel`` precedent from MonoGS-det)
rather than teaching the library about a new host.

Replaced by these objects (T01 §1 coupling map):

    C1  policy parameters, single in-place shared copy (:295 -> :166/:147)
                                              -> ``weights`` channel, 2 slots,
                                                 generation + CLAIM/ACK
    C2  rollout buffers                       -> ``rollout`` channel, 4 slots
    C4  free_queue (buffer credits)           -> LAST_ACK at P2 (:216)
    C5  full_queue (rollout hand-off)         -> PUBLISH/outcome at P1 (:180)
    C6  step counter / stop                   -> StopBoard three-phase EOS

The two module-level digest functions are module-level on purpose: the fabric is
pickled into the spawned learner, and a closure dies there as
``UNPICKLABLE_DIGEST`` (HARNESS_PORT_READOUT_20260824.md §6.1).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from detex.boards import LtcBoard                                   # noqa: E402
from detex.channel import Channel, Outcome, PayloadSpec             # noqa: E402
from detex.fabric import TO_MAPPER, TO_TRACKER, FabricBase          # noqa: E402
from detex.schedule import Schedule                                 # noqa: E402
from detex.trace import digest_tensors                              # noqa: E402

# Outcome kinds.  One shape per direction, so one kind each.
KIND_NONE = 0
KIND_ROLLOUT = 1
KIND_WEIGHTS = 1

CHANNEL_ROLLOUT = "rollout"
CHANNEL_WEIGHTS = "weights"


# ---------------------------------------------------------------------------
# payload digests (spec §10 / §14.3: the snapshot must not change between READY
# and the last ACK).  Both payloads are fixed-shape -- there is no valid prefix
# to take, unlike GS-ICP's point buffers -- so the whole slot is hashed.
# ---------------------------------------------------------------------------

def digest_rollout(fields: Dict[str, Any]) -> str:
    return digest_tensors([fields[key] for key in sorted(fields)])


def digest_weights(fields: Dict[str, Any]) -> str:
    return digest_tensors([fields[key] for key in sorted(fields)])


# ---------------------------------------------------------------------------
# host-side absorptions for the negative control (phase table §7, S_free)
# ---------------------------------------------------------------------------

class RunLocalLtcBoard(LtcBoard):
    """LTC board whose *in-run* frontier wait is skipped below ``frames``.

    ``TrackerRole`` waits ``LTC >= (t-1,0,P4)`` at every P0
    (``detex/role.py:370``).  In the free-running negative control the actor
    must not wait for the learner at all -- that is what "upstream hogwild"
    means -- but the EOS drain wait at ``t = frames`` has to stay real, or the
    tracker would assert quiescence while the learner is still mid-frame and the
    control would fail for the wrong reason.  The split is exactly on that
    boundary and nowhere else.

    Host-side subclass, zero ``detex/`` edits (precedent: MonoGS's
    ``ConsumerAckChannel``).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.skip_below = int(kwargs.pop("skip_below"))
        super().__init__(*args, **kwargs)

    def wait_ge(self, tag: Any) -> None:
        if int(tag.t) < self.skip_below:
            return
        return super().wait_ge(tag)


class RunLocalOutcomeChannel(Channel):
    """Channel whose *in-run* outcome read does not block (negative control).

    Same boundary as ``RunLocalLtcBoard``: under ``S_free`` the weights channel
    publishes ``ABSENT`` on every frame, so the value the tracker reads is the
    same either way -- only the blocking differs, and blocking is precisely the
    coupling the control has to remove.  At ``t >= frames`` the real blocking
    read is used so the EOS handshake is unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.skip_below = int(kwargs.pop("skip_below"))
        super().__init__(*args, **kwargs)

    def read_outcome(self, t: int, b: int) -> Outcome:
        if int(t) < self.skip_below:
            return Outcome(False, None, KIND_NONE)
        return super().read_outcome(t, b)


# ---------------------------------------------------------------------------
# fabric
# ---------------------------------------------------------------------------

class MonobeastFabric(FabricBase):
    """Phase table §4: two channels, declared in the order the hand-off uses.

    Registration order is part of the run's identity (``detex/fabric.py``), so
    ``rollout`` (A->L, published first at P1) precedes ``weights`` (L->A,
    published at P4).
    """

    def __init__(
        self,
        ctx: Any,
        rollout_specs: List[PayloadSpec],
        weight_specs: List[PayloadSpec],
        rollout_slots: int = 4,
        weight_slots: int = 2,
        watchdog_s: float = 300.0,
        verify: bool = True,
        outcome_capacity: int = 8192,
        leak_frames: Optional[int] = None,
    ) -> None:
        super().__init__(ctx, num_slots=weight_slots, watchdog_s=watchdog_s,
                         verify=verify, sync_path="M")
        self.rollout_slots = int(rollout_slots)
        self.weight_slots = int(weight_slots)
        self.leak_frames = None if leak_frames is None else int(leak_frames)

        common = dict(ctx=ctx, abort=self.abort, sync=self.sync,
                      watchdog_s=watchdog_s, outcome_capacity=outcome_capacity,
                      domain="cpu")
        self.register(CHANNEL_ROLLOUT, Channel(
            CHANNEL_ROLLOUT, rollout_specs, num_slots=self.rollout_slots,
            **common), digest_rollout, TO_MAPPER)

        if self.leak_frames is None:
            weights = Channel(CHANNEL_WEIGHTS, weight_specs,
                              num_slots=self.weight_slots, **common)
        else:
            weights = RunLocalOutcomeChannel(
                CHANNEL_WEIGHTS, weight_specs, num_slots=self.weight_slots,
                skip_below=self.leak_frames, **common)
            # replace the board built by FabricBase; nothing has attached yet
            self.ltc = RunLocalLtcBoard(ctx, self.abort, watchdog_s=watchdog_s,
                                        skip_below=self.leak_frames)
        self.register(CHANNEL_WEIGHTS, weights, digest_weights, TO_TRACKER)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels": self.channel_order,
            "rollout_slots": self.rollout_slots,
            "weight_slots": self.weight_slots,
            "verify": self.verify,
            "sync_path": self.sync.kind,
            "leak_frames": self.leak_frames,
        }


# ---------------------------------------------------------------------------
# S (spec §6, phase table §7)
# ---------------------------------------------------------------------------

def build_schedule(frames: int, batch_size: int, mode: str = "closure",
                   num_slots: int = 2, slack: int = 1) -> Schedule:
    """Materialise S as a table and validate it offline before any run starts.

    closure-S rule (phase table §7): generation ``g`` is published at P4 of
    frame ``f(g) = (g+1)B - 1``; the P0 frontier wait guarantees every P4 at a
    frame ``<= t-1`` has completed; hence ``u(t) = floor(t/B) - 1``, the latest
    generation whose P4 precedes unroll ``t``'s P0.  No arrival time enters it.

    ``slack`` generalises that last step to ``u(t) = floor(t/B) - slack``
    (phase table §9, W4).  ``slack = 1`` is the W2/W3-relaxed schedule and is
    left byte-identical, name included, so every recorded digest stays
    reachable.  ``slack = 2`` names a generation published a whole block
    earlier, which is the schedule change registered as W4: it is still a pure
    function of the credit structure -- no arrival time, no poll, no wall clock
    -- and it is still rejected offline by ``Schedule.validate`` if the extra
    reuse window outruns the double buffer.  It is a *different S*, so it has
    its own reference digests; it is not a relaxation of the same run.
    """
    frames = int(frames)
    B = int(batch_size)
    slack = int(slack)
    if frames % B:
        raise ValueError("frames must be a whole number of learner steps")
    if slack < 1:
        raise ValueError("slack must be >= 1; slack 0 would name a generation "
                         "published at the consuming frame (S_CAUSALITY)")
    fires = [((t + 1) % B) == 0 for t in range(frames)]
    m_table = [[1 if fires[t] else 0] for t in range(frames)]
    kf_table = [True] * frames
    if mode == "closure":
        pub_table = list(fires)
        u_table = [None if (t // B) - slack < 0 else (t // B) - slack
                   for t in range(frames)]
        name = ("closure-S-B%d" % B if slack == 1
                else "monobeast-closure-S-slack%d-B%d" % (slack, B))
    elif mode == "free_running":
        # S_free: no versioned publication at all.  The learner instead writes
        # the single un-versioned shared copy in place at :295 (phase table §7).
        pub_table = [False] * frames
        u_table = [None] * frames
        name = "free-running-S-B%d" % B
    else:
        raise ValueError("unknown schedule mode %r" % (mode,))
    schedule = Schedule(frames, m_table, kf_table, pub_table, u_table,
                        blocks_per_frame=1, tail_credits=[], name=name)
    schedule.validate(num_slots=num_slots)
    return schedule


def specs_from_buffers(buffers: Dict[str, List[torch.Tensor]]) -> List[PayloadSpec]:
    """Payload specs taken from upstream ``create_buffers`` output.

    Shapes and dtypes are read off the tensors upstream itself allocates rather
    than restated here, so the channel cannot drift from ``monobeast.py:299``.
    """
    return [PayloadSpec(key, tuple(value[0].shape), value[0].dtype, "cpu")
            for key, value in buffers.items()]


def specs_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[PayloadSpec]:
    return [PayloadSpec(key, tuple(value.shape), value.dtype, "cpu")
            for key, value in state_dict.items()]


CORE_STATE_PREFIX = "core_state_"


def core_state_keys(fields: Any) -> List[str]:
    """The LSTM initial-state fields of a rollout payload, in tuple order.

    ``()`` when ``--use_lstm`` is off, which is why the non-LSTM payload -- and
    therefore every W2/W3 digest -- is untouched by this addition.
    """
    return sorted((key for key in fields if key.startswith(CORE_STATE_PREFIX)),
                  key=lambda key: int(key[len(CORE_STATE_PREFIX):]))


def specs_from_core_state(state: Any) -> List[PayloadSpec]:
    """Payload specs for upstream's ``initial_agent_state_buffers`` entry.

    T01 §6 / phase table §8: upstream keeps the per-unroll LSTM initial state in
    a *separate* set of buffers whose lifetime is governed by ``free_queue``
    alone, and reads them through a lazy generator forced after the buffer has
    been handed back.  Under this phase table the initial state is part of the
    rollout payload instead, so it is covered by the same slot state machine,
    the same commit-word digest and the same CLAIM/LAST_ACK window as the rest
    of the unroll -- the metering the registered prediction is about.
    """
    return [PayloadSpec("%s%d" % (CORE_STATE_PREFIX, i), tuple(t.shape),
                        t.dtype, "cpu")
            for i, t in enumerate(state)]
