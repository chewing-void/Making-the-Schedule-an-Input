"""Phase-table drivers — the part of the contract a host system should not rewrite.

Until now the contract's loop lived in the SLAM fork: ``mp_Tracker_exchange.py``
and ``mp_Mapper_exchange.py`` each carried ~100 lines of phase-table walk,
CLAIM/ACK sequencing, EOS drain and abort handling alongside the host's own
arithmetic.  That made the integration a *fork* rather than a module — adopting
the contract meant transcribing the skeleton correctly, and transcribing it
correctly is exactly the thing the contract exists to stop people having to do.

Everything here is host-independent.  What a host still supplies is only what a
host actually knows:

    what a payload contains          -> ``WriteSpec.fill`` / ``digest``
    what to do with a claimed payload -> ``consume`` (tracker), ``extract`` (mapper)
    what one credit does              -> ``train_once``
    what the end state digests to     -> ``exit_digest``

The event sequence emitted here is byte-for-byte the sequence the GS-ICP fork
emitted before this module existed; the refactor is verified by re-running a
frozen configuration and requiring the recorded L1/L2/trajectory/map digests to
be unchanged.  A driver that "cleans up" the ordering would silently invalidate
every closure result already published, so the ordering below is deliberate and
is commented where it looks arbitrary.

Scope, stated rather than implied: this drives the two-actor tracker/mapper
coupling the spec's phase table describes.  It is not a general dataflow
runtime, and a system whose actors do not decompose that way needs a different
phase table, not a different host class.

Two claim regimes, and the second one exists because of what a second host
exposed.  The skeleton above was extracted from the GS-ICP fork, the only
system that had adopted the contract, and it silently inherited that system's
schedule shape: keyframe publication is data-dependent there, so the schedule
cannot name the generation a frame will use, and the tracker instead reads the
outcome board and claims whatever was published.  Every generation is then used
exactly once, by the very next frame.

``detex.harness`` — the contract's own exercise scenario, written earlier and
deliberately built to be *not* the easy case: absent keyframes, targets
published every other frame, generations reused across several frames, and
every fourth generation never consumed so DROP_ACK sits on the critical path —
cannot run on that.  Its schedule *does* name the generation in advance, which
is the ``closure-S`` half of the spec §5 gate predicate
(``g_use := u(t)``), and none of ``u`` / ``skips`` / ``last_use`` / channel
reuse came along with the extraction because the host it was extracted from
never needed them.

So there are two regimes and a host declares which it is in:

    CLAIM_BY_OUTCOME    presence is discovered at run time from the outcome
                        board; the schedule fixes only the credits m(t,b).
                        GS-ICP, and the default.
    CLAIM_BY_SCHEDULE   the schedule names u(t), which generations are skipped,
                        and when each one is last used.  The harness.

The default path below is byte-for-byte the code that produced every closure
digest on record; the second regime is additive.  That is deliberate — a
refactor that "unified" the two would put every published result at risk to buy
tidiness, and the equivalence gate would have to re-certify all of it.

And a second axis, exposed the same way by a third host.  *Which* generation a
frame uses is one question; *how long the frame has to wait before it may start*
is another, and the skeleton answered the second one the way the first adopter
needed: P0 waits ``LTC >= (t-1,0,P4)`` unconditionally, i.e. for the mapper to
have finished the whole previous frame.  For GS-ICP that is exactly right —
``u(t)`` is always the generation published at ``t-1``, so the frontier wait and
the data dependency coincide.  For a host whose ``u(t)`` names an older
generation the two come apart, and the wait serialises two processes that the
schedule says may overlap: the monobeast port (``u(t) = floor(t/B) - 1``) runs
in lock-step, the actor unable to begin frame ``t+1`` until the learner has
finished frame ``t``, for a generation published ``B`` frames ago.

So the frontier is a second declared thing, on the same additive terms:

    FRONTIER_BY_OUTCOME   wait for ``LTC >= (t-1,0,P4)``, then read
                          ``outcome(t-1)``.  GS-ICP, the harness, and the
                          default.
    FRONTIER_BY_SCHEDULE  wait only until ``u(t)``'s generation is published,
                          i.e. for ``LTC >= (f(u(t)),0,P4)`` where ``f`` is
                          ``Schedule.publish_frame_of``.  The wait becomes a
                          function of S, which is what the contract says the
                          run is a function of.

The relaxation is a physical one only: the same generation is claimed at the
same frame, the same generations are dropped at the same frames, and no
recorded event moves — the outcome reads it defers are silent (``read_outcome``
and ``LtcBoard.wait_ge`` record nothing), so the L1 sequence, the per-actor
sequence numbers and therefore L2 and L3 are unchanged.  That is a claim with a
gate behind it, not an argument: the monobeast matrix is re-run under
FRONTIER_BY_SCHEDULE and every digest must equal the lock-step values bit for
bit.  Requires ``CLAIM_BY_SCHEDULE`` — without a declared ``u(t)`` there is no
frontier to compute.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from .errors import ProtocolError, require
from .shared import blocked_report
from .tags import MAPPER, TRACKER, make_tag
from .trace import EventRecorder

KIND_NONE = 0

# Which half of the spec §5 gate predicate a host uses to pick its generation.
CLAIM_BY_OUTCOME = "outcome"
CLAIM_BY_SCHEDULE = "schedule"

# How far the tracker's P0 waits before it may claim: the previous frame's
# publish phase (what the first adopter needed) or the publish phase of the
# generation S actually names.  See the module docstring.
FRONTIER_BY_OUTCOME = "outcome"
FRONTIER_BY_SCHEDULE = "schedule"


class WriteSpec(NamedTuple):
    """How to fill one outbound slot, and how to hash what was filled.

    ``fill`` receives the slot's payload object and writes into it in place;
    ``valid_length`` is read *after* filling, because for several payload types
    the used length is a property of the data the host just wrote (the target
    channel reports it through ``using_idx``).  ``digest`` is called only when
    verification is on.
    """

    fill: Callable[[Any], None]
    valid_length: Callable[[Any], int]
    digest: Optional[Callable[[Any], str]] = None


class Publication(NamedTuple):
    """What an actor puts on the wire at its publish phase.

    ``kind`` is the host's own classification of the outcome (GS-ICP uses
    tracking/mapping keyframe); the contract only requires that it be an int and
    that ``KIND_NONE`` mean "nothing published".  ``writes`` maps channel name to
    ``WriteSpec``; a publication with ``kind != KIND_NONE`` must write every
    channel in its direction, because a consumer keys presence off the first one
    and a partial publication would desynchronise them.
    """

    kind: int
    writes: Dict[str, WriteSpec]

    @classmethod
    def nothing(cls) -> "Publication":
        return cls(KIND_NONE, {})


class _RoleBase:
    """Recorder lifecycle, abort propagation and perturbation evidence.

    Every actor needs these three and none of them is host-specific.  The abort
    path matters more than it looks: without it a crashed actor leaves its peer
    blocked on a condition variable until the watchdog fires, and the watchdog
    deadline is long on purpose (it must not fire during a legitimate slow
    phase), so the peer would look hung rather than failed.
    """

    actor: int = -1
    abort_phase: str = "P6_T_COMMIT"

    def __init__(self, fabric: Any, schedule: Any, host: Any,
                 recorder_path: Any, perturb: Any) -> None:
        self.fabric = fabric
        self.schedule = schedule
        self.host = host
        self.recorder_path = Path(recorder_path)
        self.perturb = perturb
        self._t = 0

    def run(self) -> None:
        recorder = EventRecorder(self.recorder_path, self.actor)
        self.fabric.attach(self.actor, recorder)
        started = time.perf_counter()
        try:
            self._body(recorder)
        except BaseException:
            self._wall_s = time.perf_counter() - started
            self.fabric.abort.set()
            recorder.record(make_tag(self._t, 0, self.abort_phase), "ABORT")
            self._dump_evidence()
            recorder.close()
            raise
        self._wall_s = time.perf_counter() - started
        self._dump_evidence()
        recorder.close()

    @property
    def _name(self) -> str:
        return "tracker" if self.actor == TRACKER else "mapper"

    def _dump_evidence(self) -> None:
        """Side files: what was perturbed, and what the actor waited on.

        Both are written on the abort path too.  A run that died is exactly the
        run whose blocking profile someone will want, and writing it only on
        success would make the interesting case the unavailable one.
        """
        directory = self.recorder_path.parent
        (directory / ("%s.perturb.json" % self._name)).write_text(
            json.dumps(self.perturb.to_dict(), indent=2))
        (directory / ("%s.cost.json" % self._name)).write_text(
            json.dumps(self._cost(), indent=2))

    def _cost(self) -> Dict[str, Any]:
        """This actor's wall time and where it went.

        ``wall_s`` covers the actor loop only — process start-up, CUDA context
        creation and dataset construction happen before ``run`` and are not the
        contract's to account for.  ``sync`` is the two blocking calls of the
        sync path; ``blocked`` is the gate waits.  Together they answer the
        question Path-E turns on: is this actor paying for the producer sync out
        of time it could have used, or out of time it would have spent waiting
        for its peer anyway?
        """
        sync = self.fabric.sync.cost()
        blocked = blocked_report()
        return {
            "actor": self._name,
            "wall_s": round(getattr(self, "_wall_s", 0.0), 6),
            "sync": sync,
            "blocked": blocked,
            "blocked_total_s": round(
                sum(entry["seconds"] for entry in blocked.values()), 6),
        }

    def _body(self, recorder: EventRecorder) -> None:  # pragma: no cover
        raise NotImplementedError

    # ---- shared helpers ------------------------------------------------------

    def _fill_and_hold(self, names: List[str], writes: Dict[str, WriteSpec],
                       tag) -> Dict[str, Any]:
        """Acquire and fill every outbound slot, without committing.

        Splitting acquisition from commit is what lets the mapper compute its
        target snapshot at ingest — where the original computed it — while the
        commit stays at the publish phase the phase table assigns it.
        """
        pending: Dict[str, Any] = {}
        for name in names:
            spec = writes.get(name)
            if spec is None:
                continue
            channel = self.fabric.channels[name]
            write = channel.begin_write(tag)
            spec.fill(write.fields)
            pending[name] = (write, spec)
        return pending

    def _commit_and_publish(self, pending: Dict[str, Any], names: List[str],
                            t: int, b: int, kind: int, tag) -> None:
        """Commit every held slot, then publish every outcome.

        All commits precede all publishes.  A consumer may act the instant an
        outcome appears, so publishing channel A before channel B is committed
        would expose a window in which A is readable and B is not yet valid.
        """
        verify = self.fabric.verify
        for name in names:
            held = pending.get(name)
            if held is None:
                continue
            write, spec = held
            channel = self.fabric.channels[name]
            channel.commit(
                write, tag,
                valid_length=int(spec.valid_length(write.fields)),
                digest=spec.digest(write.fields) if (verify and spec.digest) else None)
        for name in names:
            held = pending.get(name)
            channel = self.fabric.channels[name]
            if held is None:
                channel.publish_outcome(t, b, False, None, tag, kind=KIND_NONE)
            else:
                channel.publish_outcome(
                    t, b, True, held[0].generation, tag, kind=kind)


class TrackerRole(_RoleBase):
    """Walks P0 / P1 / P6 per frame, then the three-phase stop of spec §7.

    Host protocol:

    ``frame_count() -> int``
    ``consume(name, view, tag) -> None``   install a claimed inbound payload
    ``produce(t) -> Publication``          the host's per-frame work
    ``frame_digest(t) -> str | None``      what P6 records for this frame
    ``on_stop() -> None``                  host work that must precede STOP_GRANT
    ``exit_digest() -> str | None``
    """

    actor = TRACKER
    abort_phase = "P6_T_COMMIT"

    def __init__(self, *args: Any, regime: str = CLAIM_BY_OUTCOME,
                 frontier: str = FRONTIER_BY_OUTCOME, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        require(
            regime in (CLAIM_BY_OUTCOME, CLAIM_BY_SCHEDULE), "CLAIM_REGIME",
            "regime must be CLAIM_BY_OUTCOME or CLAIM_BY_SCHEDULE",
            regime=regime)
        require(
            frontier in (FRONTIER_BY_OUTCOME, FRONTIER_BY_SCHEDULE),
            "FRONTIER_MODE",
            "frontier must be FRONTIER_BY_OUTCOME or FRONTIER_BY_SCHEDULE",
            frontier=frontier)
        require(
            frontier == FRONTIER_BY_OUTCOME or regime == CLAIM_BY_SCHEDULE,
            "FRONTIER_MODE",
            "FRONTIER_BY_SCHEDULE computes the wait from u(t), which only the "
            "declared-schedule regime has",
            frontier=frontier, regime=regime)
        self.regime = regime
        self.frontier = frontier
        # Only the schedule-driven regime carries state between frames: a
        # generation can be reused for several frames and dropped for several
        # more, so what this reader has seen and released is not recoverable
        # from the board alone.
        self._known: Dict[str, List[int]] = {}
        self._dropped: Dict[str, set] = {}
        self._held_generation: Optional[int] = None
        # Highest frame whose outcomes have been read.  Under
        # FRONTIER_BY_OUTCOME this is always ``t-1`` and nothing needs to
        # remember it; under FRONTIER_BY_SCHEDULE the reader lags the frame
        # loop, so the drain has to know where the unread tail starts.
        self._read_through = -1

    def _body(self, recorder: EventRecorder) -> None:
        if self.regime == CLAIM_BY_SCHEDULE:
            return self._body_by_schedule(recorder)
        return self._body_by_outcome(recorder)

    def _body_by_outcome(self, recorder: EventRecorder) -> None:
        fabric = self.fabric
        verify = fabric.verify
        inbound = fabric.to_tracker
        outbound = fabric.to_mapper
        frames = int(self.host.frame_count())

        for t in range(frames):
            self._t = t

            # ---- P0: install the generation this frame is due ----------------
            tag0 = make_tag(t, 0, "P0_T_CONSUME")
            self.perturb.hit()
            if t > 0:
                # one frontier wait covers every inbound channel: the producer
                # advances LTC once, after all of them are published
                fabric.ltc.wait_ge(make_tag(t - 1, 0, "P4_M_PUBLISH"))
                for name in inbound:
                    channel = fabric.channels[name]
                    outcome = channel.read_outcome(t - 1, 0)
                    if not outcome.present:
                        continue
                    view = channel.claim(outcome.generation, tag0)
                    if verify:
                        channel.verify_payload(
                            view, fabric.digests[name](view.fields), tag0)
                    self.host.consume(name, view, tag0)
                    channel.last_ack(outcome.generation, tag0)

            # ---- host arithmetic --------------------------------------------
            publication = self.host.produce(t)

            # ---- P1: publish ---------------------------------------------------
            tag1 = make_tag(t, 0, "P1_T_PUBLISH")
            self.perturb.hit()
            if publication.kind != KIND_NONE:
                self._require_complete(publication, outbound, t)
                pending = self._fill_and_hold(outbound, publication.writes, tag1)
            else:
                pending = {}
            self._commit_and_publish(pending, outbound, t, 0, publication.kind, tag1)

            # ---- P6: commit the frame ------------------------------------------
            tag6 = make_tag(t, 0, "P6_T_COMMIT")
            self.perturb.hit()
            recorder.record(tag6, "COMMIT", digest=self.host.frame_digest(t))

        self._stop(recorder, frames)

    def _body_by_schedule(self, recorder: EventRecorder) -> None:
        """P0 / P1 / P6 with the generation named by the schedule — spec §5.

        The differences from the outcome-driven body are all in P0 and P6, and
        each one is a contract feature the other regime never reaches:

        * ``u(t)`` may name a generation published several frames ago, so a
          reader that already holds it must ``view`` rather than ``claim``
          (claiming twice is DOUBLE_CLAIM);
        * a generation the schedule skips is never consumed, so its channel pin
          has to be handed back with DROP_ACK or two consecutive skips wedge a
          double-buffered channel;
        * the release therefore happens at P6, when ``last_use`` says so, not
          at P0 immediately after reading.
        """
        fabric = self.fabric
        verify = fabric.verify
        inbound = fabric.to_tracker
        outbound = fabric.to_mapper
        frames = int(self.host.frame_count())
        # Both bodies index every tag at b=0.  A schedule with several blocks per
        # frame is legal in the spec and the boards support it, but this driver
        # does not walk it, and silently collapsing the blocks would produce a
        # perfectly reproducible run of the wrong schedule.
        require(
            int(getattr(self.schedule, "blocks_per_frame", 1)) == 1,
            "BLOCKS_UNSUPPORTED",
            "the phase-table driver walks one block per frame; a multi-block "
            "schedule needs the b index threaded through P3/P4/P5",
            blocks_per_frame=getattr(self.schedule, "blocks_per_frame", 1))

        for t in range(frames):
            self._t = t

            # ---- P0: release what the schedule skips, then take u(t) --------
            tag0 = make_tag(t, 0, "P0_T_CONSUME")
            self.perturb.hit()
            if self.frontier == FRONTIER_BY_SCHEDULE:
                self._wait_declared_frontier(t, inbound)
            elif t > 0:
                fabric.ltc.wait_ge(make_tag(t - 1, 0, "P4_M_PUBLISH"))
                for name in inbound:
                    outcome = fabric.channels[name].read_outcome(t - 1, 0)
                    if outcome.present and outcome.generation is not None:
                        self._known.setdefault(name, []).append(outcome.generation)

            g_use = self.schedule.u(t)
            for name in inbound:
                channel = fabric.channels[name]
                dropped = self._dropped.setdefault(name, set())
                for generation in self._known.setdefault(name, []):
                    if (generation not in dropped and self.schedule.skips(generation)
                            and g_use is not None and generation < g_use):
                        channel.drop_ack(generation, tag0)
                        dropped.add(generation)

            # a second hit here because the claim is the interesting boundary:
            # the perturber has to be able to land between deciding what to take
            # and taking it, or the window it is meant to widen stays closed
            self.perturb.hit()
            if g_use is not None:
                for name in inbound:
                    channel = fabric.channels[name]
                    view = (channel.view(g_use) if channel.holds(g_use)
                            else channel.claim(g_use, tag0))
                    if verify:
                        channel.verify_payload(
                            view, fabric.digests[name](view.fields), tag0)
                    self.host.consume(name, view, tag0)
            self._held_generation = g_use

            # ---- host arithmetic --------------------------------------------
            publication = self.host.produce(t)

            # ---- P1: publish -------------------------------------------------
            tag1 = make_tag(t, 0, "P1_T_PUBLISH")
            self.perturb.hit()
            if publication.kind != KIND_NONE:
                self._require_complete(publication, outbound, t)
                pending = self._fill_and_hold(outbound, publication.writes, tag1)
            else:
                pending = {}
            self._commit_and_publish(pending, outbound, t, 0, publication.kind, tag1)

            # ---- P6: last use release, then commit ---------------------------
            tag6 = make_tag(t, 0, "P6_T_COMMIT")
            self.perturb.hit()
            if g_use is not None and self.schedule.last_use(g_use) == t:
                for name in inbound:
                    fabric.channels[name].last_ack(g_use, tag6)
            recorder.record(tag6, "COMMIT", digest=self.host.frame_digest(t))

        self._stop_by_schedule(recorder, frames)

    def _wait_declared_frontier(self, t: int, inbound: List[str]) -> None:
        """Wait for u(t)'s generation only, then catch the outcome reads up.

        The frontier the contract actually requires at ``(t,0,P0)`` is "the
        generation S names for this frame is published", not "the previous
        frame is finished".  ``f(g) = publish_frame_of(g)`` and S_CAUSALITY
        already guarantees ``f(u(t)) < t``, so this wait is never later than
        the FRONTIER_BY_OUTCOME one and is usually much earlier.

        The outcome reads are what makes this bookkeeping rather than a plain
        deletion.  ``_known`` has to end up holding every published generation
        in publish order — the drop rule at P0 and the orphan sweep at EOS both
        walk it — so the reads are not skipped, only deferred: everything from
        ``_read_through + 1`` up to the frontier frame is read here, in frame
        order, which is the same list the frame-by-frame reader builds.  A
        generation the schedule skips is published before ``u(t)`` is, hence at
        a frame ``<= f(u(t))``, so it is in ``_known`` in time to be dropped at
        exactly the frame the other mode drops it.

        Reading is safe as soon as the wait returns: the mapper publishes its
        outcome at P4 and only then advances LTC at P5, so LTC past
        ``(f,0,P4)`` implies frames ``<= f`` are on the outcome ring.
        """
        g_use = self.schedule.u(t)
        if g_use is None:
            # nothing named, nothing to wait for.  The frames skipped here are
            # not lost: the next frame with a u(t), or the drain, reads them.
            return
        frontier_t = int(self.schedule.publish_frame_of(g_use))
        self.fabric.ltc.wait_ge(make_tag(frontier_t, 0, "P4_M_PUBLISH"))
        for read_t in range(self._read_through + 1, frontier_t + 1):
            for name in inbound:
                outcome = self.fabric.channels[name].read_outcome(read_t, 0)
                if outcome.present and outcome.generation is not None:
                    self._known.setdefault(name, []).append(outcome.generation)
        if frontier_t > self._read_through:
            self._read_through = frontier_t

    def _stop_by_schedule(self, recorder: EventRecorder, frames: int) -> None:
        """EOS for the schedule-driven regime — spec §7 and §10.

        Unlike the outcome-driven drain, which only has to release the final
        frame's generation, this one has to account for every generation the
        schedule ever published: some were skipped and never claimed, some are
        still held, and the last frames' outcomes have not been read at all
        because the frame loop only ever reads ``outcome(t-1)``.  Anything left
        pinned here fails ``assert_no_orphans``, which is the point of checking.
        """
        fabric = self.fabric
        eos = frames
        self._t = eos
        stop_tag = make_tag(eos, 0, "P6_T_COMMIT")
        fabric.stop.request(stop_tag, make_tag(eos - 1, 0, "P6_T_COMMIT"))

        tag_drain = make_tag(eos, 0, "P0_T_CONSUME")
        fabric.ltc.wait_ge(make_tag(eos, 0, "P4_M_PUBLISH"))
        # Where the unread tail starts.  The frame loop under
        # FRONTIER_BY_OUTCOME has read everything up to eos-2, so the tail is
        # the last two frames and this is the literal ``eos - 1`` it always
        # was; under FRONTIER_BY_SCHEDULE the reader may be several frames
        # behind, and every one of those frames can hold a generation nobody
        # will ever claim.  Missing one shows up as a failed assert_no_orphans,
        # which is the check earning its keep.
        first_tail = eos - 1
        if self.frontier == FRONTIER_BY_SCHEDULE:
            first_tail = min(eos - 1, self._read_through + 1)
        for name in fabric.to_tracker:
            channel = fabric.channels[name]
            known = self._known.setdefault(name, [])
            dropped = self._dropped.setdefault(name, set())
            for tail_t in range(first_tail, eos + 1):
                outcome = channel.read_outcome(tail_t, 0)
                if (outcome.present and outcome.generation is not None
                        and outcome.generation not in known):
                    known.append(outcome.generation)
            for generation in known:
                if generation in dropped or channel.holds(generation):
                    continue
                # published and never taken: either the schedule skipped it or
                # it has no last use at all.  Both are pins nobody will release.
                if (self.schedule.skips(generation)
                        or self.schedule.last_use(generation) is None):
                    channel.drop_ack(generation, tag_drain)
                    dropped.add(generation)
            for generation in sorted({g for g in known if channel.holds(g)}):
                channel.last_ack(generation, tag_drain)

        for name in fabric.channel_order:
            fabric.channels[name].assert_quiescent("Q(stop)")
        for name in fabric.channel_order:
            fabric.channels[name].assert_no_orphans("Q(stop)")

        self.host.on_stop()
        fabric.stop.grant(stop_tag, stop_tag)
        recorder.record(stop_tag, "EXIT", digest=self.host.exit_digest())
        recorder.flush()

    def _require_complete(self, publication: Publication, outbound: List[str],
                          t: int) -> None:
        missing = [name for name in outbound if name not in publication.writes]
        if missing:
            raise ProtocolError(
                "PARTIAL_PUBLICATION",
                "a publication with a kind must write every outbound channel; "
                "consumers key presence off the first one",
                t=t, kind=publication.kind, missing=missing)

    def _stop(self, recorder: EventRecorder, frames: int) -> None:
        """Three-phase EOS — spec §7."""
        fabric = self.fabric
        eos = frames
        self._t = eos
        stop_tag = make_tag(eos, 0, "P6_T_COMMIT")
        fabric.stop.request(stop_tag, make_tag(eos - 1, 0, "P6_T_COMMIT"))

        tag_drain = make_tag(eos, 0, "P0_T_CONSUME")
        fabric.ltc.wait_ge(make_tag(eos, 0, "P4_M_PUBLISH"))
        # the last frame is forced to be a keyframe, so the generation published
        # for it is never consumed by a following frame; releasing it here is
        # what keeps Q(stop) free of orphan pins
        for name in fabric.to_tracker:
            channel = fabric.channels[name]
            for tail_t in range(eos - 1, eos + 1):
                outcome = channel.read_outcome(tail_t, 0)
                if not (outcome.present and outcome.generation is not None):
                    continue
                if channel.holds(outcome.generation):
                    channel.last_ack(outcome.generation, tag_drain)
                else:
                    channel.drop_ack(outcome.generation, tag_drain)

        for name in fabric.channel_order:
            fabric.channels[name].assert_quiescent("Q(stop)")
        for name in fabric.channel_order:
            fabric.channels[name].assert_no_orphans("Q(stop)")

        self.host.on_stop()
        fabric.stop.grant(stop_tag, stop_tag)
        recorder.record(stop_tag, "EXIT", digest=self.host.exit_digest())
        recorder.flush()


class MapperRole(_RoleBase):
    """Walks P2 / P3 / P4 / P5 per frame, then drains under STOP_REQUEST.

    Host protocol:

    ``extract(name, view, tag) -> Any``    read a claimed payload out, between
                                           CLAIM and ACK and nowhere else
    ``ingest(t, extracted, kind, tag) -> Publication | None``
    ``train_once(t, b, j) -> None``        exactly one credit's worth of work
    ``exit_digest() -> str | None``
    ``exit_k() -> int | None``
    """

    actor = MAPPER
    abort_phase = "P3_M_WORK"

    def __init__(self, *args: Any, leak_mode: str = "none",
                 regime: str = CLAIM_BY_OUTCOME, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        require(
            regime in (CLAIM_BY_OUTCOME, CLAIM_BY_SCHEDULE), "CLAIM_REGIME",
            "regime must be CLAIM_BY_OUTCOME or CLAIM_BY_SCHEDULE",
            regime=regime)
        # On this actor the regime governs the *publish* decision, not the
        # claim.  Under CLAIM_BY_OUTCOME the mapper publishes only when it had
        # something to ingest, which is GS-ICP: a target exists because a
        # keyframe arrived.  Under CLAIM_BY_SCHEDULE the two are independent —
        # the G-EX schedule has frames with no keyframe that must still publish
        # a target (kf is false at t%3==2, pub is true at t%2==0, and t=2 is
        # both) — so the host is asked what to publish on every frame.
        #
        # It also governs WHEN the outbound snapshot is taken.  GS-ICP fills the
        # target at ingest, before this frame's credits are burned, because that
        # is where the original code computed it; the harness fills it at P4,
        # after, so the published payload reflects this frame's training.  Both
        # are legal — the phase table fixes when a payload becomes *visible*, not
        # when it is computed — but they are different states, and a host must
        # not get the other one by accident.
        self.regime = regime
        self._deferred: Optional[Publication] = None
        # G-F5's negative control: the mapper back on the clock with the rest of
        # the contract intact, so a divergence is attributable to the credit
        # discipline alone.  It lives here rather than in a host because a new
        # host must be able to run the negative control without reimplementing
        # it — a gate nobody can reproduce is not a gate.
        self.leak_mode = leak_mode

    def _body(self, recorder: EventRecorder) -> None:
        fabric = self.fabric
        verify = fabric.verify
        inbound = fabric.to_mapper
        outbound = fabric.to_tracker
        frames = int(self.host.frame_count())
        primary = inbound[0]

        for t in range(frames):
            self._t = t
            pending: Dict[str, Any] = {}

            # ---- P2: ingest ---------------------------------------------------
            tag2 = make_tag(t, 0, "P2_M_INGEST")
            self.perturb.hit()
            outcomes = {name: fabric.channels[name].read_outcome(t, 0)
                        for name in inbound}
            present = outcomes[primary].present
            disagree = [name for name in inbound
                        if outcomes[name].present != present]
            if disagree:
                raise ProtocolError(
                    "SPLIT_PUBLICATION",
                    "inbound channels disagree on presence for this tag; the "
                    "producer must publish them together",
                    t=t, primary=primary, disagree=disagree)

            if present:
                kind = outcomes[primary].kind
                extracted: Dict[str, Any] = {}
                for name in inbound:
                    channel = fabric.channels[name]
                    outcome = outcomes[name]
                    view = channel.claim(outcome.generation, tag2)
                    if verify:
                        channel.verify_payload(
                            view, fabric.digests[name](view.fields), tag2)
                    extracted[name] = self.host.extract(name, view, tag2)
                    # a CUDA payload needs a recorded completion, not just a
                    # returned Python object, before the slot may be reused
                    channel.last_ack(
                        outcome.generation, tag2,
                        done_event=fabric.sync.consumer_done_event(channel.has_cuda))

                publication = self.host.ingest(t, extracted, kind, tag2)
                if self.regime == CLAIM_BY_SCHEDULE:
                    self._deferred = publication
                elif publication is not None and publication.kind != KIND_NONE:
                    pending = self._fill_and_hold(
                        outbound, publication.writes, make_tag(t, 0, "P4_M_PUBLISH"))
                    self._pending_kind = publication.kind
                else:
                    self._pending_kind = KIND_NONE
            elif self.regime == CLAIM_BY_SCHEDULE:
                # nothing arrived, but the schedule may still call for a
                # publication this frame; only the host knows, so ask it
                self._deferred = self.host.ingest(t, {}, KIND_NONE, tag2)

            # ---- P3: burn exactly m(t,b) credits -------------------------------
            tag3 = make_tag(t, 0, "P3_M_WORK")
            self.perturb.hit()
            credits = self.schedule.m(t, 0)
            if self.leak_mode == "free_running":
                import time as _time
                credits += int((_time.time() * 1000.0) % 3)
            for j in range(credits):
                self.host.train_once(t, 0, j)
            fabric.credit.done(tag3, credits)

            # ---- P4 / P5 --------------------------------------------------------
            tag4 = make_tag(t, 0, "P4_M_PUBLISH")
            self.perturb.hit()
            if self.regime == CLAIM_BY_SCHEDULE:
                # acquire and fill here, not at ingest: the snapshot is meant to
                # include the credits just burned.  The slot is therefore held
                # for a shorter window too, which is a side effect and not the
                # reason.
                publication = self._deferred
                self._deferred = None
                if publication is not None and publication.kind != KIND_NONE:
                    pending = self._fill_and_hold(outbound, publication.writes, tag4)
                    self._pending_kind = publication.kind
                else:
                    pending = {}
                    self._pending_kind = KIND_NONE
            self._commit_and_publish(
                pending, outbound, t, 0,
                getattr(self, "_pending_kind", KIND_NONE) if pending else KIND_NONE,
                tag4)
            fabric.ltc.advance(tag4, make_tag(t, 0, "P5_M_LTC"))

        self._drain(recorder, frames)

    def _drain(self, recorder: EventRecorder, frames: int) -> None:
        """Drain under STOP_REQUEST — spec §7."""
        fabric = self.fabric
        eos = frames
        self._t = eos
        fabric.stop.wait_request()
        for b, credits in enumerate(self.schedule.tail_credits):
            tag_tail = make_tag(eos, b, "P3_M_WORK")
            for j in range(credits):
                self.host.train_once(eos, b, j)
            fabric.credit.done(tag_tail, credits)

        tag4 = make_tag(eos, 0, "P4_M_PUBLISH")
        self._commit_and_publish({}, fabric.to_tracker, eos, 0, KIND_NONE, tag4)
        fabric.ltc.advance(tag4, make_tag(eos, 0, "P5_M_LTC"))

        stop_tag = fabric.stop.wait_grant()
        recorder.record(
            stop_tag, "EXIT",
            k=self.host.exit_k(), digest=self.host.exit_digest())
        recorder.flush()
