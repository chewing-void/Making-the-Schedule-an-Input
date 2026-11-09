"""The schedule S — spec §6.

S is an explicit input, not an emergent property of the OS scheduler:
``Y = F(X, seed, S)``.  It fixes, per frame,

    m(t, b)   how many optimisation credits the mapper burns in that block,
    kf(t)     whether the tracker publishes a keyframe/camera outcome,
    pub(t)    whether the mapper publishes a target outcome (for t+1),
    u(t)      which published generation the tracker consumes,

plus the stop tag and the tail credits burned during EOS drain.

``validate`` enforces ``Valid(X, seed, S)``.  One of its conditions was found
while implementing rather than while writing the contract, and is written back
into the spec: with ``num_slots`` buffers per channel, a generation's last use
must not fall after the frame in which generation ``g + num_slots`` is
published, or the producer blocks on a slot the consumer will not release until
a frame the consumer cannot reach.  It is a liveness condition on S, and it is
checkable offline, which is the whole point of S being an input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .errors import ProtocolError, require


class Schedule:
    def __init__(
        self,
        frames: int,
        m_table: Sequence[Sequence[int]],
        kf_table: Sequence[bool],
        pub_table: Sequence[bool],
        u_table: Sequence[Optional[int]],
        blocks_per_frame: int = 1,
        tail_credits: Sequence[int] = (),
        name: str = "S",
    ) -> None:
        self.frames = int(frames)
        self.blocks_per_frame = int(blocks_per_frame)
        self.m_table = [list(row) for row in m_table]
        self.kf_table = [bool(x) for x in kf_table]
        self.pub_table = [bool(x) for x in pub_table]
        self.u_table = [None if x is None else int(x) for x in u_table]
        self.tail_credits = list(tail_credits)
        self.name = name

        # generation g is the g-th publication on the target channel
        self.publish_frame: List[int] = [t for t in range(self.frames) if self.pub_table[t]]
        self.generation_at: Dict[int, int] = {
            t: g for g, t in enumerate(self.publish_frame)}

    # ---- accessors -----------------------------------------------------------

    def m(self, t: int, b: int) -> int:
        return self.m_table[t][b]

    def kf(self, t: int) -> bool:
        return self.kf_table[t]

    def pub(self, t: int) -> bool:
        return self.pub_table[t]

    def generation_published_at(self, t: int) -> Optional[int]:
        return self.generation_at.get(t)

    def u(self, t: int) -> Optional[int]:
        return self.u_table[t]

    def published_generations(self) -> List[int]:
        return list(range(len(self.publish_frame)))

    def publish_frame_of(self, generation: int) -> int:
        return self.publish_frame[generation]

    def used_generations(self) -> List[int]:
        return sorted({g for g in self.u_table if g is not None})

    def skipped_generations(self) -> List[int]:
        used = set(self.used_generations())
        return [g for g in self.published_generations() if g not in used]

    def skips(self, generation: int) -> bool:
        return generation in set(self.skipped_generations())

    def last_use(self, generation: int) -> Optional[int]:
        frames = [t for t, g in enumerate(self.u_table) if g == generation]
        return max(frames) if frames else None

    def total_credits(self) -> int:
        return sum(sum(row) for row in self.m_table) + sum(self.tail_credits)

    # ---- Valid(X, seed, S) ---------------------------------------------------

    def validate(self, num_slots: int = 2) -> None:
        require(
            len(self.m_table) == self.frames
            and len(self.kf_table) == self.frames
            and len(self.pub_table) == self.frames
            and len(self.u_table) == self.frames,
            "S_SHAPE", "schedule tables disagree with the frame count",
            frames=self.frames)
        for t, row in enumerate(self.m_table):
            require(
                len(row) == self.blocks_per_frame, "S_BLOCKS",
                "credit row has the wrong block count", t=t, row=row)
            for b, credits in enumerate(row):
                require(credits >= 0, "S_CREDIT", "negative credit", t=t, b=b)

        published = set(self.published_generations())
        previous = None
        for t, generation in enumerate(self.u_table):
            if generation is None:
                continue
            require(
                generation in published, "S_UNKNOWN_GENERATION",
                "u(t) names a generation S never publishes", t=t, generation=generation)
            require(
                self.publish_frame_of(generation) < t, "S_CAUSALITY",
                "u(t) consumes a generation published at or after t; the target of "
                "frame p is first usable at p+1 (spec §2, C6)",
                t=t, generation=generation, published_at=self.publish_frame_of(generation))
            require(
                previous is None or generation >= previous, "S_NON_MONOTONE",
                "u(t) goes backwards", t=t, generation=generation, previous=previous)
            previous = generation

        for generation in self.published_generations():
            successor = generation + num_slots
            if successor >= len(self.publish_frame):
                continue
            last = self.last_use(generation)
            if last is None:
                continue
            require(
                last <= self.publish_frame_of(successor), "S_SLOT_FEASIBILITY",
                "generation is still in use when its slot is needed again; with "
                "num_slots buffers the producer would block on a release the "
                "consumer cannot reach",
                generation=generation, last_use=last, num_slots=num_slots,
                successor=successor, successor_frame=self.publish_frame_of(successor))

    # ---- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frames": self.frames,
            "blocks_per_frame": self.blocks_per_frame,
            "m": self.m_table,
            "kf": self.kf_table,
            "pub": self.pub_table,
            "u": self.u_table,
            "tail_credits": self.tail_credits,
            "published_generations": self.published_generations(),
            "skipped_generations": self.skipped_generations(),
            "total_credits": self.total_credits(),
        }

    # ---- constructors --------------------------------------------------------

    @classmethod
    def constant(cls, frames: int, k: int, blocks_per_frame: int = 1,
                 tail_credits: Sequence[int] = ()) -> "Schedule":
        """closure-S of the MVP: every frame publishes, every block burns k."""
        m_table = [[k] * blocks_per_frame for _ in range(frames)]
        kf_table = [True] * frames
        pub_table = [True] * frames
        u_table: List[Optional[int]] = [None if t == 0 else t - 1 for t in range(frames)]
        return cls(frames, m_table, kf_table, pub_table, u_table,
                   blocks_per_frame=blocks_per_frame, tail_credits=tail_credits,
                   name="constant-k%d" % k)

    @classmethod
    def gex(cls, frames: int, k: int = 3, blocks_per_frame: int = 1,
            skip_every: int = 4) -> "Schedule":
        """The G-EX exercise schedule.

        Deliberately not the easy case: keyframes are absent on some frames,
        targets are published every other frame, generations are reused across
        several frames, and every ``skip_every``-th generation is never
        consumed so DROP_ACK is on the critical path rather than a corner.
        """
        m_table = [[k + (t % 2) for _ in range(blocks_per_frame)] for t in range(frames)]
        kf_table = [(t % 3) != 2 for t in range(frames)]
        pub_table = [(t % 2) == 0 for t in range(frames)]

        publish_frame = [t for t in range(frames) if pub_table[t]]
        skipped = {g for g in range(len(publish_frame))
                   if skip_every > 0 and g > 0 and g % skip_every == 0}

        u_table: List[Optional[int]] = []
        for t in range(frames):
            usable = [g for g, p in enumerate(publish_frame) if p < t and g not in skipped]
            u_table.append(max(usable) if usable else None)

        schedule = cls(frames, m_table, kf_table, pub_table, u_table,
                       blocks_per_frame=blocks_per_frame,
                       tail_credits=[k], name="gex-k%d-skip%d" % (k, skip_every))
        return schedule


class ConstantCreditSchedule:
    """closure-S for the SLAM MVP: S is a *rule*, not a table (spec §6).

    In the real system the keyframe decision is data-dependent — it reads the
    correspondence ratio of the current alignment — so ``kf(t)``, ``pub(t)``
    and ``u(t)`` cannot be tabulated ahead of the run.  They do not need to be:
    the contract only requires that they be functions of (X, seed, exchange
    history) rather than of arrival times.  What S fixes here is the one thing
    the OS scheduler used to decide, the mapper's work credit per block; every
    realised value is written into the trace, which is what a later replay-S
    campaign reads back.
    """

    def __init__(self, k: int, blocks_per_frame: int = 1,
                 tail_credits: Sequence[int] = (), name: str = "closure-S") -> None:
        self.k = int(k)
        self.blocks_per_frame = int(blocks_per_frame)
        self.tail_credits = list(tail_credits)
        self.name = "%s-k%d" % (name, k)

    def m(self, t: int, b: int) -> int:
        return self.k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": "rule",
            "m": "constant %d per block" % self.k,
            "blocks_per_frame": self.blocks_per_frame,
            "tail_credits": self.tail_credits,
            "kf": "data-dependent, recorded in trace",
            "pub": "data-dependent (tracking keyframes), recorded in trace",
            "u": "latest generation with for <= t, recorded in trace",
        }


def relax_for_slots(schedule: Schedule, num_slots: int = 2) -> Schedule:
    """Return a schedule whose skips respect double-buffer feasibility.

    A skipped generation extends the previous generation's reuse window; skip
    too many in a row and ``S_SLOT_FEASIBILITY`` fires.  Rather than silently
    repairing S inside the run — S is an input, and quietly editing an input is
    how a study stops being reproducible — this helper produces an explicitly
    different S, which the caller records as such.
    """
    u_table = list(schedule.u_table)
    for generation in schedule.published_generations():
        successor = generation + num_slots
        if successor >= len(schedule.publish_frame):
            continue
        limit = schedule.publish_frame_of(successor)
        for t in range(limit + 1, schedule.frames):
            if u_table[t] == generation:
                later = [g for g, p in enumerate(schedule.publish_frame) if p < t and g > generation]
                u_table[t] = min(later) if later else None
    relaxed = Schedule(
        schedule.frames, schedule.m_table, schedule.kf_table, schedule.pub_table,
        u_table, blocks_per_frame=schedule.blocks_per_frame,
        tail_credits=schedule.tail_credits, name=schedule.name + "-relaxed")
    relaxed.validate(num_slots=num_slots)
    return relaxed
