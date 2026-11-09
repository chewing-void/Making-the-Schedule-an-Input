"""Versioned data channel and its slot state machine — spec §3.

One channel carries one logical stream (keyframe gaussians, camera, target
points) between the two processes.  It owns N payload slots plus a control
block, and it is the only place where a payload becomes visible to the peer.

Slot assignment is a pure function of the generation, ``slot = generation %
num_slots``.  The spec says the producer takes a FREE slot; picking "the lowest
FREE slot" would make the choice depend on *when* an ACK landed, which is
exactly the timing dependence the contract removes.  Deriving the slot from the
generation keeps double buffering while making the choice schedule-independent.
(Implementation refinement to spec §3 — written back into the spec.)

The (state, generation) CAS of spec §3 is realised as a compare-and-set under
the channel's condition variable.  The contract needs the pair to be inspected
and transitioned atomically; it does not need lock freedom, and a mutex makes
the "READY published, not yet claimed" window impossible to lose by accident.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import torch

from .errors import ProtocolError, require
from .shared import Abort, CondObject
from .sync import SyncPath
from .tags import Tag

FREE, WRITING, READY, READING = 0, 1, 2, 3
STATE_NAMES = {FREE: "FREE", WRITING: "WRITING", READY: "READY", READING: "READING"}

S_STATE, S_GEN, S_READERS, S_T, S_B, S_PHASE, S_LEN, S_DIGEST = range(8)
SLOT_FIELDS = 8

C_NEXT_GEN, C_LAST_PUB = 0, 1
CH_FIELDS = 4

O_KNOWN, O_PRESENT, O_GEN, O_KEY, O_KIND = range(5)
OUTCOME_FIELDS = 5


class Outcome(NamedTuple):
    """What a gate reads for one tag: presence, version, and payload kind.

    ``kind`` carries the producer's classification of the publication — for
    GS-ICP, which of the two keyframe kinds this is.  It is part of the
    outcome, not of the payload, because the consumer branches on it before it
    claims anything.
    """

    present: bool
    generation: Optional[int]
    kind: int = 0


def _digest_word(digest: Optional[str]) -> int:
    """Pack a 16-hex-char payload digest into the commit word.

    Signed int64 storage, so the top bit is folded away rather than wrapping
    into a negative that would compare unequal to itself on read-back.
    """
    if digest is None:
        return 0
    return int(digest, 16) & 0x7FFFFFFFFFFFFFFF


class PayloadSpec(NamedTuple):
    name: str
    shape: tuple
    dtype: Any
    device: str = "cpu"


class SlotView(NamedTuple):
    """A borrowed view of one slot.  Valid only inside its protocol window."""

    slot_id: int
    generation: int
    fields: Dict[str, Any]
    valid_length: int
    digest_word: int = 0


class Channel(CondObject):
    def __init__(
        self,
        name: str,
        specs: Sequence[PayloadSpec],
        ctx: Any,
        abort: Abort,
        sync: SyncPath,
        num_slots: int = 2,
        blocks_per_frame: int = 1,
        outcome_capacity: int = 4096,
        watchdog_s: float = 180.0,
        external_slots: Optional[Sequence[Any]] = None,
        domain: Optional[str] = None,
    ) -> None:
        """``external_slots`` hands the channel payload objects the caller
        already owns — the existing ``SharedGaussians`` / ``SharedCam`` /
        ``SharedTargetPoints`` modules — instead of tensors built from
        ``specs``.  Integrating into GS-ICP that way leaves every numerical
        path untouched and replaces only the exchange, which is what makes a
        bitwise A/B result attributable to the contract.  ``domain`` then says
        which completion chain applies ('cuda' or 'cpu').
        """
        super().__init__(ctx, abort, watchdog_s=watchdog_s)
        self.name = name
        self.specs = list(specs)
        self.blocks_per_frame = int(blocks_per_frame)
        self.outcome_capacity = int(outcome_capacity)
        self.sync = sync

        if external_slots is not None:
            self.slots = list(external_slots)
            self.num_slots = len(self.slots)
            require(
                domain in ("cpu", "cuda"), "CHANNEL_DOMAIN",
                "external_slots requires an explicit domain", channel=name, domain=domain)
            self.has_cuda = domain == "cuda"
        else:
            self.num_slots = int(num_slots)
            self.has_cuda = (
                domain == "cuda" if domain is not None
                else any(str(s.device).startswith("cuda") for s in self.specs))
            self.slots = []
            for _ in range(self.num_slots):
                fields: Dict[str, Any] = {}
                for spec in self.specs:
                    tensor = torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
                    if not str(spec.device).startswith("cuda"):
                        tensor.share_memory_()
                    fields[spec.name] = tensor
                self.slots.append(fields)

        self.ctrl = torch.zeros((self.num_slots, SLOT_FIELDS), dtype=torch.int64).share_memory_()
        self.ctrl[:, S_GEN] = -1          # -1: this slot has never held a generation
        self.chctrl = torch.zeros(CH_FIELDS, dtype=torch.int64).share_memory_()
        self.outcomes = torch.zeros(
            (self.outcome_capacity, OUTCOME_FIELDS), dtype=torch.int64).share_memory_()
        self._held: set = set()

    # ---- process-local state -------------------------------------------------

    def __getstate__(self):
        state = super().__getstate__()
        state["_held"] = set()
        return state

    def attach(self, actor: int, recorder: Any) -> None:
        super().attach(actor, recorder)
        self._held = set()

    def holds(self, generation: int) -> bool:
        return generation in self._held

    # ---- producer side -------------------------------------------------------

    def begin_write(self, tag: Tag) -> SlotView:
        generation = int(self.chctrl[C_NEXT_GEN].item())
        slot = generation % self.num_slots
        deadline = self._deadline()
        with self.cond:
            while True:
                state = int(self.ctrl[slot, S_STATE].item())
                readers = int(self.ctrl[slot, S_READERS].item())
                if state == FREE and readers == 0:
                    break
                self._tick(
                    deadline, "WRITE_BLOCKED",
                    channel=self.name, slot=slot, generation=generation,
                    state=STATE_NAMES[state], readers=readers)
            self.chctrl[C_NEXT_GEN] = generation + 1
            self.ctrl[slot, S_STATE] = WRITING
            self.ctrl[slot, S_GEN] = generation
            self.ctrl[slot, S_T] = tag.t
            self.ctrl[slot, S_B] = tag.b
            self.ctrl[slot, S_PHASE] = tag.phase
            self.ctrl[slot, S_LEN] = 0
            self.ctrl[slot, S_DIGEST] = 0
            self.cond.notify_all()
        self._record(tag, "WRITE_BEGIN", generation)
        return SlotView(slot, generation, self.slots[slot], 0)

    def commit(
        self,
        view: SlotView,
        tag: Tag,
        valid_length: int = 0,
        digest: Optional[str] = None,
        _unsafe_skip_producer_sync: bool = False,
    ) -> None:
        """Publish the slot.  ``_unsafe_*`` exists only for the G-EX litmus."""
        if not _unsafe_skip_producer_sync:
            self.sync.producer_publish(self.has_cuda)
        with self.cond:
            state = int(self.ctrl[view.slot_id, S_STATE].item())
            generation = int(self.ctrl[view.slot_id, S_GEN].item())
            require(
                state == WRITING and generation == view.generation,
                "COMMIT_STATE", "commit on a slot that is not the one being written",
                channel=self.name, slot=view.slot_id, state=STATE_NAMES[state],
                slot_generation=generation, view_generation=view.generation)
            self.ctrl[view.slot_id, S_STATE] = READY
            self.ctrl[view.slot_id, S_READERS] = 1     # channel pin, spec §3
            self.ctrl[view.slot_id, S_LEN] = int(valid_length)
            self.ctrl[view.slot_id, S_DIGEST] = _digest_word(digest)
            self.chctrl[C_LAST_PUB] = view.generation
            self.cond.notify_all()
        self._record(tag, "PUBLISH", view.generation, digest=digest)

    # ---- consumer side -------------------------------------------------------

    def claim(self, generation: int, tag: Tag) -> SlotView:
        require(
            generation not in self._held, "DOUBLE_CLAIM",
            "generation already held by this reader; reuse must not re-claim",
            channel=self.name, generation=generation)
        slot = generation % self.num_slots
        deadline = self._deadline()
        with self.cond:
            while True:
                state = int(self.ctrl[slot, S_STATE].item())
                slot_generation = int(self.ctrl[slot, S_GEN].item())
                if slot_generation == generation and state == READY:
                    self.ctrl[slot, S_STATE] = READING          # CAS(state, generation)
                    self.cond.notify_all()
                    break
                require(
                    slot_generation <= generation, "MISSED_GENERATION",
                    "slot already advanced past the requested generation",
                    channel=self.name, slot=slot, requested=generation,
                    slot_generation=slot_generation, state=STATE_NAMES[state])
                require(
                    not (slot_generation == generation and state in (FREE, READING)),
                    "GENERATION_GONE",
                    "requested generation is no longer claimable",
                    channel=self.name, slot=slot, generation=generation,
                    state=STATE_NAMES[state])
                self._tick(
                    deadline, "CLAIM_BLOCKED",
                    channel=self.name, slot=slot, requested=generation,
                    slot_generation=slot_generation, state=STATE_NAMES[state])
            valid_length = int(self.ctrl[slot, S_LEN].item())
            digest_word = int(self.ctrl[slot, S_DIGEST].item())
        self._held.add(generation)
        self._record(tag, "CLAIM", generation)
        return SlotView(slot, generation, self.slots[slot], valid_length, digest_word)

    def view(self, generation: int) -> SlotView:
        """View of a generation this reader already holds (multi-frame reuse)."""
        require(
            generation in self._held, "NOT_HELD",
            "view requested for a generation this reader does not hold",
            channel=self.name, generation=generation)
        slot = generation % self.num_slots
        with self.cond:
            state = int(self.ctrl[slot, S_STATE].item())
            slot_generation = int(self.ctrl[slot, S_GEN].item())
            valid_length = int(self.ctrl[slot, S_LEN].item())
            digest_word = int(self.ctrl[slot, S_DIGEST].item())
        require(
            state == READING and slot_generation == generation, "VIEW_STATE",
            "held generation is not in READING", channel=self.name,
            slot=slot, generation=generation, state=STATE_NAMES[state],
            slot_generation=slot_generation)
        return SlotView(slot, generation, self.slots[slot], valid_length, digest_word)

    def verify_payload(self, view: SlotView, computed_digest: str, tag: Tag) -> None:
        """Spec §10: the snapshot must not change between READY and LAST_ACK.

        The consumer hashes what it actually read and compares against the
        digest carried in the commit word.  This is what turns a torn or stale
        read — the failure mode both GPU litmus tests provoke — into a protocol
        abort instead of a silently different trajectory.
        """
        expected = int(self.ctrl[view.slot_id, S_DIGEST].item())
        computed = _digest_word(computed_digest)
        require(
            expected == computed, "PAYLOAD_MISMATCH",
            "payload read does not match the digest published in the commit word",
            channel=self.name, tag=str(tag), slot=view.slot_id,
            generation=view.generation, expected=expected, computed=computed)

    def last_ack(
        self,
        generation: int,
        tag: Tag,
        done_event: Optional[Any] = None,
        _unsafe_skip_completion: bool = False,
    ) -> None:
        """Release after the last use (spec §3, LastUse_S(g)).

        For a CUDA-domain channel the consumer must hand in the event that
        marks its reads complete; releasing on Python return alone lets the
        producer overwrite an in-flight read.  ``_unsafe_skip_completion``
        exists only so the G-EX litmus can show the negative control failing.
        """
        if not _unsafe_skip_completion:
            if self.has_cuda:
                require(
                    done_event is not None, "ACK_WITHOUT_COMPLETION",
                    "CUDA-domain LAST_ACK requires a consumer-done event",
                    channel=self.name, generation=generation)
            self.sync.wait_done(done_event)
        slot = generation % self.num_slots
        with self.cond:
            state = int(self.ctrl[slot, S_STATE].item())
            slot_generation = int(self.ctrl[slot, S_GEN].item())
            require(
                state == READING and slot_generation == generation, "ACK_STATE",
                "LAST_ACK on a slot that is not being read by this generation",
                channel=self.name, slot=slot, generation=generation,
                state=STATE_NAMES[state], slot_generation=slot_generation)
            self.ctrl[slot, S_STATE] = FREE
            self.ctrl[slot, S_READERS] = 0
            self.cond.notify_all()
        self._held.discard(generation)
        self._record(tag, "LAST_ACK", generation)

    def drop_ack(self, generation: int, tag: Tag) -> None:
        """Release a generation the schedule plans to skip (spec §3).

        Without this the channel pin taken at PUBLISH is never handed back and
        two consecutive planned skips wedge a double-buffered channel.
        """
        slot = generation % self.num_slots
        with self.cond:
            state = int(self.ctrl[slot, S_STATE].item())
            slot_generation = int(self.ctrl[slot, S_GEN].item())
            require(
                state == READY and slot_generation == generation, "DROP_STATE",
                "DROP_ACK on a slot that is not the READY skipped generation",
                channel=self.name, slot=slot, generation=generation,
                state=STATE_NAMES[state], slot_generation=slot_generation)
            self.ctrl[slot, S_STATE] = FREE
            self.ctrl[slot, S_READERS] = 0
            self.cond.notify_all()
        self._record(tag, "DROP_ACK", generation)

    # ---- outcomes ------------------------------------------------------------

    def _outcome_key(self, t: int, b: int) -> int:
        return t * self.blocks_per_frame + b

    def publish_outcome(
        self, t: int, b: int, present: bool, generation: Optional[int], tag: Tag,
        kind: int = 0,
    ) -> None:
        key = self._outcome_key(t, b)
        index = key % self.outcome_capacity
        with self.cond:
            known = int(self.outcomes[index, O_KNOWN].item())
            stored_key = int(self.outcomes[index, O_KEY].item())
            require(
                not (known and stored_key == key), "OUTCOME_REPUBLISH",
                "outcome for this tag was already published",
                channel=self.name, t=t, b=b)
            require(
                not (known and stored_key != key), "OUTCOME_RING_WRAP",
                "outcome ring wrapped onto an unconsumed entry",
                channel=self.name, t=t, b=b, stored_key=stored_key,
                capacity=self.outcome_capacity)
            self.outcomes[index, O_KNOWN] = 1
            self.outcomes[index, O_PRESENT] = 1 if present else 0
            self.outcomes[index, O_GEN] = -1 if generation is None else int(generation)
            self.outcomes[index, O_KEY] = key
            self.outcomes[index, O_KIND] = int(kind)
            self.cond.notify_all()
        self._record(
            tag, "OUTCOME_PRESENT" if present else "OUTCOME_ABSENT", generation)

    def read_outcome(self, t: int, b: int) -> Outcome:
        """Blocking read of a published outcome (spec §5: wait, then read)."""
        key = self._outcome_key(t, b)
        index = key % self.outcome_capacity
        deadline = self._deadline()
        with self.cond:
            while True:
                known = int(self.outcomes[index, O_KNOWN].item())
                stored_key = int(self.outcomes[index, O_KEY].item())
                if known and stored_key == key:
                    present = bool(self.outcomes[index, O_PRESENT].item())
                    generation = int(self.outcomes[index, O_GEN].item())
                    kind = int(self.outcomes[index, O_KIND].item())
                    return Outcome(present, None if generation < 0 else generation, kind)
                require(
                    not (known and stored_key > key), "OUTCOME_LOST",
                    "outcome ring already advanced past this tag",
                    channel=self.name, t=t, b=b, stored_key=stored_key)
                self._tick(
                    deadline, "OUTCOME_BLOCKED", channel=self.name, t=t, b=b)

    # ---- diagnostics ---------------------------------------------------------

    def dump(self) -> Dict[str, Any]:
        """Abort dump of spec §10: every slot's (state, generation, readers)."""
        slots = []
        for slot in range(self.num_slots):
            slots.append({
                "slot": slot,
                "state": STATE_NAMES[int(self.ctrl[slot, S_STATE].item())],
                "generation": int(self.ctrl[slot, S_GEN].item()),
                "readers": int(self.ctrl[slot, S_READERS].item()),
                "t": int(self.ctrl[slot, S_T].item()),
                "b": int(self.ctrl[slot, S_B].item()),
                "valid_length": int(self.ctrl[slot, S_LEN].item()),
            })
        return {
            "channel": self.name,
            "domain": "cuda" if self.has_cuda else "cpu",
            "next_generation": int(self.chctrl[C_NEXT_GEN].item()),
            "last_published": int(self.chctrl[C_LAST_PUB].item()),
            "slots": slots,
            "held": sorted(self._held),
        }

    def assert_quiescent(self, where: str) -> None:
        """Spec §8/§10: at a Q cut no slot may be WRITING or READING."""
        for slot in range(self.num_slots):
            state = int(self.ctrl[slot, S_STATE].item())
            require(
                state in (FREE, READY), "QUIESCENCE",
                "slot is mid-transfer at a cut", where=where,
                channel=self.name, slot=slot, state=STATE_NAMES[state])

    def assert_no_orphans(self, where: str) -> None:
        """Spec §10: no pinned slot may survive EOS."""
        for slot in range(self.num_slots):
            state = int(self.ctrl[slot, S_STATE].item())
            readers = int(self.ctrl[slot, S_READERS].item())
            require(
                state == FREE and readers == 0, "ORPHAN_PIN",
                "slot still pinned at end of stream", where=where,
                channel=self.name, slot=slot, state=STATE_NAMES[state],
                readers=readers)
