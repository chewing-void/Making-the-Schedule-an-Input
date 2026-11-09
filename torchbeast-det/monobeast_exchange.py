#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for this project: deterministic exchange driver, role hosts,
# reporting, startup and stop coordination. Derived portions remain Apache-2.0.
# Upstream: facebookresearch/torchbeast @ 0af07b051a2176a8f9fd20c36891ba2bba6bae68
"""detex-hosted monobeast — the exchange driver and the two role hosts.

HOST ADAPTER WITH UPSTREAM-ADAPTED LOGIC. The upstream checkout and detex
are separate dependencies. Upstream
``act``, ``learn`` and ``Net`` are imported and interposed, and the phase-table
walk comes from ``detex.role``.  What this file supplies is only what a host
actually knows -- what a payload contains, what to do with a claimed one, what
one credit does, and what the end state digests to.

Roles: **actor = tracker**, **learner = mapper**; logical tick
``t := unroll``; ``u`` is pinned per unroll.

The one piece of upstream control logic this file replaces is ``train``
(``monobeast.py:319``) -- process start-up, the stop protocol and the learner
thread pool.

Usage
-----
  python monobeast_exchange.py --run_id C1 --seed 20260825 --outdir DIR \
      --mode closure -- --env PongNoFrameskip-v4 --num_actors 1 \
      --num_learner_threads 1 --batch_size 4 --unroll_length 20 \
      --total_steps 4000 --disable_cuda --disable_checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _path in (_HERE, _REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from torchbeast import monobeast as mb  # noqa: E402

from detex.errors import ProtocolError, require  # noqa: E402
from detex.perturb import Perturber  # noqa: E402
from detex.role import (  # noqa: E402
    CLAIM_BY_SCHEDULE, FRONTIER_BY_OUTCOME, FRONTIER_BY_SCHEDULE, MapperRole,
    Publication, TrackerRole, WriteSpec,
)
from detex.tags import MAPPER, TRACKER, make_tag  # noqa: E402
from detex.trace import merge_traces, write_layers  # noqa: E402

from m6_detex_fabric import (  # noqa: E402
    CHANNEL_ROLLOUT, CHANNEL_WEIGHTS, KIND_ROLLOUT, KIND_WEIGHTS,
    MonobeastFabric, build_schedule, core_state_keys, digest_rollout,
    digest_weights, specs_from_buffers, specs_from_core_state,
    specs_from_state_dict,
)


# ---------------------------------------------------------------- utilities --

def sd_sha256(state_dict: Dict[str, Any]) -> str:
    """L3 end-state digest, same recipe as the W1 baseline's ``sd_digest``."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().contiguous().view(-1).numpy().tobytes())
    return h.hexdigest()


class JsonlSink:
    def __init__(self, path: str) -> None:
        self.f = open(path, "w", buffering=1)

    def __call__(self, **record: Any) -> None:
        self.f.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


class _OsShim:
    """Only monobeast's own ``os`` lookups are affected (W1 pattern)."""

    def __init__(self, real: Any, urandom: Any) -> None:
        self._real = real
        self.urandom = urandom

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def det_urandom_factory(seed: int, actor_index: int):
    def det_urandom(n: int) -> bytes:
        out, i = b"", 0
        while len(out) < n:
            out += hashlib.sha256(
                ("m6det:%d:%d:%d" % (seed, actor_index, i)).encode()).digest()
            i += 1
        return out[:n]
    return det_urandom


class _CapturePublish:
    """Contract mode: ``:295`` must not reach any shared memory.

    Upstream's last statement in ``learn`` is
    ``actor_model.load_state_dict(model.state_dict())`` -- the hogwild publish.
    Under the contract the publication is an event at P4 on the weights channel,
    not a memcpy at an arbitrary point of P3, so the object handed to ``learn``
    as ``actor_model`` swallows it and records that one was due.  The learner
    model itself is untouched, so the state P4 publishes is bit-identical to what
    ``:295`` would have copied.
    """

    def __init__(self) -> None:
        self.n = 0

    def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self.n += 1
        return None


# ------------------------------------------------------- actor rendezvous ----

class _UnrollGate:
    """Strict one-unroll-at-a-time rendezvous with upstream ``act``.

    ``act`` (``monobeast.py:128``) is a ``while True`` loop whose two coupling
    points are ``free_queue.get()`` (unroll start) and ``full_queue.put()``
    (unroll publish).  Those are exactly the phase-table boundaries, so the loop
    runs verbatim on a thread and the host drives it one unroll per grant.

    Only one of the two threads is ever runnable: ``grant`` blocks until the act
    thread reports done, and the act thread blocks until the next grant.  This
    is a coroutine, not a race -- there is one possible interleaving.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._grant: Optional[int] = None
        self._granted = False
        self._done = True
        self._error: Optional[BaseException] = None

    # -- role side ----------------------------------------------------------
    def grant(self, index: Optional[int]) -> None:
        with self._cv:
            if self._error is not None:
                raise self._error
            self._grant = index
            self._granted = True
            self._done = False
            self._cv.notify_all()
            while not self._done:
                self._cv.wait()
            if self._error is not None:
                raise self._error

    def release(self) -> None:
        """Post upstream's loop-exit sentinel without waiting for a completion.

        ``act`` answers ``None`` by breaking out of its ``while`` (``:150``), so
        there is no ``full_queue.put`` to wait for -- ``grant`` would deadlock.
        """
        with self._cv:
            self._grant = None
            self._granted = True
            self._cv.notify_all()

    # -- act side -----------------------------------------------------------
    def take(self) -> Optional[int]:
        with self._cv:
            while not self._granted:
                self._cv.wait()
            self._granted = False
            return self._grant

    def finish(self) -> None:
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self._cv:
            self._error = exc
            self._done = True
            self._cv.notify_all()


class _ActFreeQueue:
    """Stands in for ``free_queue``: one credit per grant (``:149``)."""

    def __init__(self, gate: _UnrollGate) -> None:
        self._gate = gate

    def get(self) -> Optional[int]:
        return self._gate.take()


class _ActFullQueue:
    """Stands in for ``full_queue``: the unroll boundary (``:180``)."""

    def __init__(self, gate: _UnrollGate, counter: Dict[str, int]) -> None:
        self._gate = gate
        self._counter = counter

    def put(self, index: int) -> None:
        self._counter["unrolls"] += 1
        self._gate.finish()


# ------------------------------------------------------------- actor host ----

class ActorHost:
    """The tracker-role host: phase table §4, rows P0 / P1 / P6."""

    def __init__(self, flags: Any, fabric: Any, schedule: Any, net: Any,
                 staging: Dict[str, List[Any]], seed: int, sink: Any,
                 leak_params: Optional[Dict[str, Any]] = None,
                 pin_assert: bool = True,
                 staging_core: Optional[List[Any]] = None) -> None:
        self.flags = flags
        self.fabric = fabric
        self.schedule = schedule
        self.frames = int(schedule.frames)
        self.sink = sink
        self.seed = int(seed)
        self.pin_assert = bool(pin_assert)

        self._net = net
        self._staging = staging
        self._keys = list(staging.keys())
        # Upstream's ``initial_agent_state_buffers``: one entry, because the
        # host grants exactly one staging index.  ``()`` without --use_lstm, in
        # which case nothing below this line does anything at all.
        self._staging_core = staging_core if staging_core is not None else [tuple()]
        self._core_keys = ["core_state_%d" % i
                           for i in range(len(self._staging_core[0]))]
        self._gate = _UnrollGate()
        self._counter = {"unrolls": 0}
        self._thread: Optional[threading.Thread] = None
        self._view = None
        self._claimed_generation: Optional[int] = None
        self._rollout_digest: Optional[str] = None
        self._traj = hashlib.blake2b(b"m6:traj:v1", digest_size=16)
        self._consumes = 0
        self._stopped = False

        if leak_params is not None:
            # S_free: the actor reads the single un-versioned shared copy for the
            # whole run -- upstream C1 (``monobeast.py:352`` + ``:295``).
            for key, param in self._net.named_parameters():
                param.data = leak_params[key]
        self._leaked = leak_params is not None

    # -- upstream act on a thread -------------------------------------------
    def start(self) -> None:
        # daemon=True is an ABORT-PATH fix and nothing else.  On the success
        # path ``on_stop`` posts the sentinel, joins, and ``require``s the
        # thread dead, so daemon-ness is never reached.  On a protocol abort the
        # tracker raises before ``on_stop``, leaving upstream ``act`` parked in
        # ``free_queue.get()`` forever; with a non-daemon thread the interpreter
        # then refuses to exit and a fail-stop looks like a hang, which is the
        # worst possible reading of a gate whose whole point is fail-stopping.
        # Found by the W3 use_lstm cell, which is the first run to abort.
        self._thread = threading.Thread(
            target=self._act_main, name="upstream-act", daemon=True)
        self._thread.start()

    def _act_main(self) -> None:
        try:
            mb.os = _OsShim(os, det_urandom_factory(self.seed, 0))
            torch.set_num_threads(1)
            mb.act(self.flags, 0,
                   _ActFreeQueue(self._gate),
                   _ActFullQueue(self._gate, self._counter),
                   self._net, self._staging, self._staging_core)
        except BaseException as exc:  # noqa: BLE001 - propagated to the role
            self._gate.fail(exc)

    # -- host protocol ------------------------------------------------------
    def frame_count(self) -> int:
        return self.frames

    def consume(self, name: str, view: Any, tag: Any) -> None:
        """P0: bind the actor Net's parameter storage to the claimed slot.

        Zero copy and structurally torn-read-free: the slot is READING for the
        whole unroll, and ``Channel.begin_write`` blocks on FREE, so the producer
        cannot touch it while a forward is reading it.
        """
        require(name == CHANNEL_WEIGHTS, "UNEXPECTED_INBOUND",
                "the actor has exactly one inbound channel", channel=name)
        for key, param in self._net.named_parameters():
            param.data = view.fields[key]
        self._view = view
        self._claimed_generation = view.generation
        self._consumes += 1

    def produce(self, t: int) -> Publication:
        """One unroll of upstream ``act`` under the pinned generation."""
        self._gate.grant(0)
        if self.pin_assert and self._view is not None:
            # Phase table §6: what was read is still what was published.
            self.fabric.channels[CHANNEL_WEIGHTS].verify_payload(
                self._view, digest_weights(self._view.fields),
                make_tag(t, 0, "P1_T_PUBLISH"))
        return Publication(KIND_ROLLOUT, {
            CHANNEL_ROLLOUT: WriteSpec(
                fill=self._fill_rollout,
                valid_length=lambda fields: 0,
                digest=self._stashed_digest)})

    def _fill_rollout(self, fields: Dict[str, Any]) -> None:
        for key in self._keys:
            fields[key].copy_(self._staging[key][0])
        # The LSTM initial state upstream writes at ``:158``-``:159`` is part of
        # this unroll's payload, so it is copied into the slot here, inside
        # ``WriteSpec.fill`` -- i.e. while the slot is FREE and the act thread
        # is parked in ``free_queue.get()``.  It is hashed into the commit word
        # by ``digest_rollout`` with everything else.
        for i, key in enumerate(self._core_keys):
            fields[key].copy_(self._staging_core[0][i])
        self._rollout_digest = digest_rollout(fields)

    def _stashed_digest(self, fields: Dict[str, Any]) -> str:
        return self._rollout_digest

    def frame_digest(self, t: int) -> str:
        generation = self.schedule.u(t)
        self._traj.update(("%d|%s|%s\n" % (
            t, "-" if generation is None else generation,
            self._rollout_digest)).encode())
        digest = self._traj.hexdigest()
        self.sink(s="unroll", t=t, u=generation, rollout=self._rollout_digest,
                  traj=digest)
        return digest

    def on_stop(self) -> None:
        """EOS: release upstream's loop with its own sentinel, then join."""
        self._stopped = True
        self._gate.release()
        if self._thread is not None:
            self._thread.join(timeout=60.0)
            require(not self._thread.is_alive(), "ACTOR_THREAD_ALIVE",
                    "upstream act did not exit after the stop sentinel")
        require(self._counter["unrolls"] == self.frames, "UNROLL_COUNT",
                "actor published a different number of unrolls than S declares",
                produced=self._counter["unrolls"], declared=self.frames)

    def exit_digest(self) -> str:
        return self._traj.hexdigest()


# ----------------------------------------------------------- learner host ----

class LearnerHost:
    """The mapper-role host: phase table §4, rows P2 / P3 / P4 / P5."""

    def __init__(self, flags: Any, schedule: Any, model: Any, optimizer: Any,
                 scheduler: Any, publish_target: Any, sink: Any,
                 fabric: Any = None, lazy_core: bool = False,
                 lazy_verify: bool = False) -> None:
        self.flags = flags
        self.schedule = schedule
        self.frames = int(schedule.frames)
        self.B = int(flags.batch_size)
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._publish_target = publish_target
        self.sink = sink
        self._fabric = fabric
        self.lazy_core = bool(lazy_core)
        self.lazy_verify = bool(lazy_verify)
        self._lazy: Dict[int, Any] = {}
        self.lazy_forced = 0
        self.lazy_recycled = 0
        self.lazy_view_after_ack_code: Optional[str] = None
        self._cols: List[Optional[Dict[str, Any]]] = [None] * self.B
        self._order: List[int] = []
        self.credits = 0
        self.publications = 0
        self._last_stats: Dict[str, Any] = {}

    def frame_count(self) -> int:
        return self.frames

    def extract(self, name: str, view: Any, tag: Any) -> Any:
        """P2: copy the payload out between CLAIM and ACK, and nowhere else.

        ``lazy_core`` is the deliberately NON-CONFORMING diagnostic arm.  It
        reproduces upstream ``get_batch``'s shape inside the contract: the
        rollout tensors are copied here (upstream's eager ``torch.stack`` at
        ``:207``) but the LSTM initial state is left as a live reference into
        the slot (upstream's generator at ``:210``-``:213``) and is only read in
        ``train_once``, i.e. after LAST_ACK -- which is upstream's
        ``free_queue.put`` at ``:216``.  It exists to answer "what does the
        contract do when a host writes the racy pattern", and it is never the
        gate configuration.
        """
        require(name == CHANNEL_ROLLOUT, "UNEXPECTED_INBOUND",
                "the learner has exactly one inbound channel", channel=name)
        if not self.lazy_core:
            return {key: value.clone() for key, value in view.fields.items()}
        core = core_state_keys(view.fields)
        require(bool(core), "LAZY_WITHOUT_LSTM",
                "the lazy-core diagnostic needs --use_lstm; without it there is "
                "no lazily-read field and nothing to demonstrate")
        out = {key: (value if key in core else value.clone())
               for key, value in view.fields.items()}
        self._lazy[int(view.generation)] = {
            "view": view,
            # what the payload hashed to while the claim was still held; the
            # commit word itself cannot be used after release because it moves
            # with the slot's next generation (see the report)
            "claim_digest": digest_rollout(view.fields),
            "shadow": {key: view.fields[key].clone() for key in core},
        }
        return out

    def ingest(self, t: int, extracted: Dict[str, Any], kind: int,
               tag: Any) -> Publication:
        require(CHANNEL_ROLLOUT in extracted, "MISSING_ROLLOUT",
                "S declares a rollout on every unroll", t=t)
        self._cols[t % self.B] = extracted[CHANNEL_ROLLOUT]
        self._order.append(t)
        if self.schedule.pub(t):
            return Publication(KIND_WEIGHTS, {
                CHANNEL_WEIGHTS: WriteSpec(
                    fill=self._fill_weights,
                    valid_length=lambda fields: 0,
                    digest=digest_weights)})
        return Publication.nothing()

    def train_once(self, t: int, b: int, j: int) -> None:
        """P3: exactly one credit = one call of upstream ``learn``.

        The batch expression is ``get_batch``'s own (``monobeast.py:207-209``)
        with ``buffers[key][m]`` replaced by the copy claimed at P2; sigma is the
        ascending unroll order, which S declares and the trace records.
        """
        require(all(col is not None for col in self._cols), "BATCH_INCOMPLETE",
                "a credit fired before B unrolls were ingested", t=t)
        order = list(self._order)
        self._order = []
        core = core_state_keys(self._cols[0])
        batch = {key: torch.stack([self._cols[i][key] for i in range(self.B)],
                                  dim=1)
                 for key in self._cols[0] if key not in core}
        # ``get_batch``'s own expression for the initial state (:210-:213):
        # concatenate the per-index states along the batch dim, in sigma order.
        # Under the contract these are the copies claimed at P2, not live
        # buffer memory -- that substitution is the registered prediction.
        if self.lazy_core and core:
            self._force_lazy(order, t)
        initial_agent_state = tuple(
            torch.cat([self._cols[i][key] for i in range(self.B)], dim=1)
            for key in core)
        stats = mb.learn(self.flags, self._publish_target, self._model, batch,
                         initial_agent_state, self._optimizer, self._scheduler)
        self.credits += 1
        self._last_stats = stats
        self.sink(s="credit", t=t, j=j, k=self.credits, order=order,
                  lr=self._optimizer.param_groups[0]["lr"],
                  learner_w=sd_sha256(self._model.state_dict()),
                  total_loss=repr(stats["total_loss"]),
                  pg_loss=repr(stats["pg_loss"]),
                  baseline_loss=repr(stats["baseline_loss"]),
                  entropy_loss=repr(stats["entropy_loss"]))

    def _force_lazy(self, order: List[int], t: int) -> None:
        """Diagnostic: what does the slot hold at the moment upstream forces?

        Called immediately before the ``torch.cat`` that reads the LSTM initial
        state out of slots this reader has already LAST_ACKed.  Three things are
        recorded, and they are what separate "the metering catches it" from
        "the metering was never reached":

        * ``lazy_recycled`` -- the read differs from the value that was in the
          slot while the claim was held.  That is the read-after-recycle of
          T01 §6, reproduced inside the contract.
        * ``lazy_view_after_ack_code`` -- what ``Channel.view`` says when a
          released generation is asked for.  A conforming host cannot get a
          post-release view at all; this arm only has one because it stashed
          the raw field dict behind the channel's back.
        * with ``lazy_verify``, a consumer-side comparison against the digest
          latched at CLAIM.  Note ``Channel.verify_payload`` cannot do this job
          after release: it reads the slot's *current* commit word, which moves
          with the content, so it is blind by construction outside the window.
        """
        channel = self._fabric.channels[CHANNEL_ROLLOUT] if self._fabric else None
        for generation in order:
            entry = self._lazy.pop(int(generation), None)
            if entry is None:
                continue
            view = entry["view"]
            self.lazy_forced += 1
            recycled = any(
                not torch.equal(view.fields[key], entry["shadow"][key])
                for key in entry["shadow"])
            if channel is not None and self.lazy_view_after_ack_code is None:
                try:
                    channel.view(int(generation))
                    self.lazy_view_after_ack_code = "NO_ERROR"
                except ProtocolError as exc:
                    self.lazy_view_after_ack_code = getattr(exc, "code", repr(exc))
            now = digest_rollout(view.fields)
            if recycled:
                self.lazy_recycled += 1
                self.sink(s="lazy_recycled", t=t, generation=int(generation),
                          slot=int(view.slot_id),
                          claim_digest=entry["claim_digest"], force_digest=now,
                          view_after_ack=self.lazy_view_after_ack_code)
            if self.lazy_verify and now != entry["claim_digest"]:
                raise ProtocolError(
                    "PAYLOAD_MISMATCH_AFTER_ACK",
                    "the payload changed between CLAIM and the deferred read; "
                    "the slot was recycled while a lazy reference into it was "
                    "still live (upstream monobeast :216 before :219)",
                    channel=CHANNEL_ROLLOUT, t=t, generation=int(generation),
                    slot=int(view.slot_id), claimed=entry["claim_digest"],
                    read=now)

    def _fill_weights(self, fields: Dict[str, Any]) -> None:
        """P4: publish exactly the state ``:295`` would have copied."""
        state = self._model.state_dict()
        for key, value in fields.items():
            value.copy_(state[key])
        self.publications += 1

    def exit_k(self) -> int:
        return self.credits

    def exit_digest(self) -> str:
        return sd_sha256(self._model.state_dict())


# ------------------------------------------------------------ learner main ---

def learner_main(flags: Any, fabric: Any, schedule: Any, obs_shape: Any,
                 num_actions: int, init_state: Dict[str, Any],
                 leak_params: Optional[Dict[str, Any]], trace_path: str,
                 sink_path: str, queue: Any, no_det_algos: bool,
                 perturb_seed: Optional[int] = None,
                 perturb_max_ms: float = 0.0, lazy_core: bool = False,
                 lazy_verify: bool = False) -> None:
    """Runs in the spawned child: the mapper role over upstream ``learn``."""
    det_error = None
    sink = None
    host = None
    try:
        torch.set_num_threads(1)
        if not no_det_algos:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:  # pragma: no cover
                det_error = repr(exc)

        model = mb.Net(obs_shape, num_actions, flags.use_lstm)
        model.load_state_dict(init_state)
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=flags.learning_rate, momentum=flags.momentum,
            eps=flags.epsilon, alpha=flags.alpha)
        T, B = flags.unroll_length, flags.batch_size

        def lr_lambda(epoch):                       # monobeast.py:395-396
            return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if leak_params is not None:
            # S_free: hand ``learn`` a Net over the single shared copy, so
            # ``:295`` performs upstream's in-place hogwild publish verbatim.
            publish_target = mb.Net(obs_shape, num_actions, flags.use_lstm)
            for key, param in publish_target.named_parameters():
                param.data = leak_params[key]
        else:
            publish_target = _CapturePublish()

        sink = JsonlSink(sink_path)
        sink(s="learner_start", det_algos_error=det_error,
             torch=torch.__version__, threads=torch.get_num_threads(),
             leak=leak_params is not None, pid=os.getpid())
        host = LearnerHost(flags, schedule, model, optimizer, scheduler,
                           publish_target, sink, fabric=fabric,
                           lazy_core=lazy_core, lazy_verify=lazy_verify)
        perturb = Perturber(perturb_seed, perturb_max_ms, MAPPER)
        role = MapperRole(fabric, schedule, host, trace_path, perturb,
                          regime=CLAIM_BY_SCHEDULE)
        role.run()
        queue.put({
            "status": "ok",
            "exit_digest": host.exit_digest(),
            "credits": host.credits,
            "publications": host.publications,
            "suppressed_publishes": getattr(publish_target, "n", None),
            "perturb": perturb.to_dict(),
            "lazy_core": {"enabled": lazy_core, "verify": lazy_verify,
                          "forced": host.lazy_forced,
                          "recycled": host.lazy_recycled,
                          "view_after_ack": host.lazy_view_after_ack_code},
            "det_algos_error": det_error,
        })
    except BaseException as exc:  # noqa: BLE001
        queue.put({"status": "error", "error": repr(exc),
                   "code": getattr(exc, "code", None),
                   "context": getattr(exc, "context", None),
                   "lazy_core": None if host is None else {
                       "enabled": lazy_core, "verify": lazy_verify,
                       "forced": host.lazy_forced,
                       "recycled": host.lazy_recycled,
                       "view_after_ack": host.lazy_view_after_ack_code},
                   "credits": None if host is None else host.credits,
                   "traceback": traceback.format_exc()})
    finally:
        if sink is not None:
            sink.close()


# ------------------------------------------------------------------- driver --

def _validate_flags(flags: Any) -> None:
    require(flags.num_actors == 1, "UNSUPPORTED_TOPOLOGY",
            "this phase table is written for one actor", num_actors=flags.num_actors)
    require(flags.num_learner_threads == 1, "UNSUPPORTED_TOPOLOGY",
            "two learner threads add a second sigma over optimizer-step order "
            "(T01 §4d)", num_learner_threads=flags.num_learner_threads)
    # ``use_lstm`` used to fail-stop here.  That guard was a placeholder for W2
    # ("registered prediction, not this round"), not a contract requirement:
    # the LSTM initial state is now a declared field of the rollout payload and
    # rides the same slot discipline as the rest of the unroll.
    require(flags.disable_cuda, "UNSUPPORTED_TOPOLOGY",
            "contract mode is CPU-only in this round")


def run(args: Any, flags: Any) -> Dict[str, Any]:
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    _validate_flags(flags)

    flags.device = torch.device("cpu")
    if flags.num_buffers is None:                    # monobeast.py:329-330
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)

    T, B = flags.unroll_length, flags.batch_size
    learner_steps = flags.total_steps // (T * B)
    require(learner_steps > 0, "TOO_SHORT", "total_steps below one learner step",
            total_steps=flags.total_steps, T=T, B=B)
    frames = learner_steps * B                      # U, the declared stop tag

    torch.set_num_threads(1)
    det_error = None
    if not args.no_det_algos:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:  # pragma: no cover
            det_error = repr(exc)

    # RNG pinning, W1 recipe.  Seed first, then reproduce upstream's construction
    # order: create_env (:347), actor Net (:349), [fork], learner Net (:383).
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)

    shape_env = mb.create_env(flags)
    obs_shape = shape_env.observation_space.shape
    num_actions = shape_env.action_space.n

    actor_net = mb.Net(obs_shape, num_actions, flags.use_lstm)
    actor_rng_state = torch.get_rng_state()          # what the fork would inherit
    learner_net = mb.Net(obs_shape, num_actions, flags.use_lstm)
    init_state = {key: value.detach().clone()
                  for key, value in learner_net.state_dict().items()}

    staging_flags = argparse.Namespace(**vars(flags))
    staging_flags.num_buffers = 1
    staging = mb.create_buffers(staging_flags, obs_shape, num_actions)
    # upstream ``train`` :356-:360, with num_buffers = 1 because the host grants
    # exactly one staging index.  ``initial_state`` is all zeros and consumes no
    # RNG, so this allocation cannot move any recorded number.
    core_template = actor_net.initial_state(batch_size=1)
    staging_core = [tuple(t.detach().clone() for t in core_template)]

    leak_params = None
    if args.mode == "free_running":
        leak_params = {key: value.detach().clone().share_memory_()
                       for key, value in actor_net.state_dict().items()}

    ctx = mp.get_context("spawn")
    fabric = MonobeastFabric(
        ctx,
        specs_from_buffers(staging) + specs_from_core_state(staging_core[0]),
        specs_from_state_dict(init_state),
        rollout_slots=max(2, B),
        weight_slots=args.weight_slots,
        watchdog_s=args.watchdog_s,
        verify=not args.no_verify,
        leak_frames=(frames if args.mode == "free_running" else None))
    for name, digest in fabric.digests.items():
        if getattr(digest, "__name__", "") == "<lambda>" or "<locals>" in getattr(
                digest, "__qualname__", ""):
            raise ProtocolError("UNPICKLABLE_DIGEST",
                                "channel digest must be a module-level function",
                                channel=name)

    schedule = build_schedule(frames, B, mode=args.mode,
                              num_slots=args.weight_slots, slack=args.slack)

    # This host declares u(t) = floor(t/B) - 1, so the driver's default
    # frontier — wait for the
    # mapper's P4 of frame t-1 — is B frames stricter than S requires and puts
    # the two processes in lock-step.  Default here is therefore the declared
    # frontier; ``--frontier outcome`` keeps the lock-step walk reachable as a
    # comparison arm.  The mode is bit-neutral by construction and by gate: it
    # moves no recorded event, so every digest is expected unchanged.
    frontier = (FRONTIER_BY_SCHEDULE if args.frontier == "schedule"
                else FRONTIER_BY_OUTCOME)

    trace_dir = os.path.join(outdir, args.run_id)
    os.makedirs(trace_dir, exist_ok=True)
    tracker_trace = os.path.join(trace_dir, "tracker.l1.txt")
    mapper_trace = os.path.join(trace_dir, "mapper.l1.txt")

    queue = ctx.Queue()
    child = ctx.Process(
        target=learner_main,
        args=(flags, fabric, schedule, obs_shape, num_actions, init_state,
              leak_params, mapper_trace,
              os.path.join(trace_dir, "learner.jsonl"), queue,
              args.no_det_algos, args.perturb_seed, args.perturb_max_ms,
              args.lstm_lazy, args.lstm_lazy_verify))

    started = time.time()
    if args.start_order == "learner_first":
        child.start()

    torch.set_rng_state(actor_rng_state)
    actor_sink = JsonlSink(os.path.join(trace_dir, "actor.jsonl"))
    actor_sink(s="run_start", run_id=args.run_id, seed=args.seed,
               mode=args.mode, frames=frames, learner_steps=learner_steps,
               det_algos_error=det_error, torch=torch.__version__,
               numpy=np.__version__, pid=os.getpid(),
               flags={k: str(v) for k, v in sorted(vars(flags).items())})

    host = ActorHost(flags, fabric, schedule, actor_net, staging, args.seed,
                     actor_sink, leak_params=leak_params,
                     pin_assert=not args.no_pin_assert,
                     staging_core=staging_core)
    perturb = Perturber(args.perturb_seed, args.perturb_max_ms, TRACKER)
    tracker: Dict[str, Any] = {"status": "ok"}
    try:
        # G-PERT start-order swap (spec §12).  ``act``'s prologue -- env
        # construction, the seeded ``gym_env.seed`` and one forward -- runs
        # before its first ``free_queue.get()``, so starting the thread first
        # lets it overlap the learner's spawn instead of following it.  Neither
        # order moves a contract event: the actor thread touches no channel
        # (the role does), and the child is spawned, so the parent's RNG state
        # at ``child.start()`` is not part of what it inherits.
        host.start()
        if args.start_order == "actor_first":
            child.start()
        TrackerRole(fabric, schedule, host, tracker_trace, perturb,
                    regime=CLAIM_BY_SCHEDULE,
                    frontier=frontier).run()
        tracker["exit_digest"] = host.exit_digest()
        tracker["consumes"] = host._consumes
        tracker["unrolls"] = host._counter["unrolls"]
    except BaseException as exc:  # noqa: BLE001
        tracker = {"status": "error", "error": repr(exc),
                   "code": getattr(exc, "code", None),
                   "context": getattr(exc, "context", None),
                   "traceback": traceback.format_exc(),
                   "dump": fabric.dump()}
        fabric.abort.set()
    tracker["perturb"] = perturb.to_dict()

    try:
        learner = queue.get(timeout=args.watchdog_s)
    except Exception as exc:  # noqa: BLE001
        fabric.abort.set()
        learner = {"status": "timeout", "error": repr(exc)}
    child.join(timeout=120.0)
    if child.is_alive():
        child.terminate()
        child.join(timeout=10.0)

    wall = time.time() - started
    actor_sink(s="run_end", wall_s=round(wall, 3),
               tracker_status=tracker.get("status"),
               learner_status=learner.get("status"))
    actor_sink.close()

    layers = write_layers([tracker_trace, mapper_trace], trace_dir, "run")
    lines = merge_traces([tracker_trace, mapper_trace])
    counts: Dict[str, int] = {}
    for line in lines:
        event = line.split("|")[3]
        counts[event] = counts.get(event, 0) + 1

    ok = (tracker.get("status") == "ok" and learner.get("status") == "ok"
          and child.exitcode == 0)
    # EOS assertion (spec §7): nothing was published or consumed after the stop.
    post_stop = None
    if ok:
        dump = fabric.dump()
        post_stop = {
            "rollout_next_generation": dump[CHANNEL_ROLLOUT]["next_generation"],
            "weights_next_generation": dump[CHANNEL_WEIGHTS]["next_generation"],
            "declared_unrolls": frames,
            "declared_generations": len(schedule.published_generations()),
        }
        ok = (post_stop["rollout_next_generation"] == frames
              and post_stop["weights_next_generation"]
              == post_stop["declared_generations"]
              and learner.get("credits") == learner_steps)

    result = {
        "run_id": args.run_id,
        "mode": args.mode,
        "frontier": frontier,
        "slack": args.slack,
        "start_order": args.start_order,
        "use_lstm": bool(flags.use_lstm),
        "lstm_lazy": {"enabled": bool(args.lstm_lazy),
                      "verify": bool(args.lstm_lazy_verify)},
        "perturb": {"seed": args.perturb_seed, "max_ms": args.perturb_max_ms,
                    "tracker": tracker.get("perturb"),
                    "mapper": learner.get("perturb")},
        "seed": args.seed,
        "status": "ok" if ok else "failed",
        "exit_code": child.exitcode,
        "wall_s": round(wall, 3),
        "env_steps": frames * T,
        "sps": round(frames * T / wall, 1) if wall > 0 else None,
        "frames": frames,
        "learner_steps": learner_steps,
        "layers": layers,
        "event_counts": dict(sorted(counts.items())),
        "post_stop": post_stop,
        "l3": {
            "learner_state_sha256": learner.get("exit_digest"),
            "actor_trajectory_digest": tracker.get("exit_digest"),
            "events": layers["events"],
        },
        "tracker": tracker,
        "learner": learner,
        "schedule": schedule.to_dict() if frames <= 64 else {
            "name": schedule.name, "frames": schedule.frames,
            "blocks_per_frame": 1,
            "total_credits": schedule.total_credits(),
            "published_generations": len(schedule.published_generations()),
            "skipped_generations": schedule.skipped_generations(),
            "rule": "m(t)=1 iff (t+1)%%B==0; pub(t)=m(t); u(t)=floor(t/B)-%d"
                    % args.slack,
        },
        "fabric": fabric.to_dict(),
        "det_algos_error": det_error,
        "flags": {k: str(v) for k, v in sorted(vars(flags).items())},
    }
    with open(os.path.join(trace_dir, "run.json"), "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", default="closure",
                        choices=("closure", "free_running"))
    parser.add_argument("--frontier", default="schedule",
                        choices=("schedule", "outcome"),
                        help="tracker P0 frontier wait: 'schedule' waits only "
                             "for u(t)'s generation (declared frontier, the "
                             "default since the G1 relaxation); 'outcome' is "
                             "the driver's original lock-step wait for the "
                             "mapper's P4 of frame t-1")
    parser.add_argument("--slack", type=int, default=1,
                        help="u(t) = floor(t/B) - slack.  1 = the W2/W3 "
                             "closure S (unchanged, same name, same digests); "
                             "2 = the W4 slack schedule, a DIFFERENT S with "
                             "its own reference digests")
    parser.add_argument("--perturb_seed", type=int, default=None,
                        help="G-PERT: seed for the sleep injector on both "
                             "actors (spec §12).  Unset = perturber disabled, "
                             "which is what every W2/W3 gate run used")
    parser.add_argument("--perturb_max_ms", type=float, default=0.0,
                        help="G-PERT: upper bound of the uniform sleep drawn "
                             "at every role gate boundary")
    parser.add_argument("--start_order", default="learner_first",
                        choices=("learner_first", "actor_first"),
                        help="G-PERT: whether the learner process or the "
                             "upstream act thread is started first")
    parser.add_argument("--lstm_lazy", action="store_true",
                        help="DIAGNOSTIC, non-conforming: read the LSTM "
                             "initial state out of the rollout slot after "
                             "LAST_ACK, reproducing upstream's :216-before-"
                             ":219 ordering inside the contract")
    parser.add_argument("--lstm_lazy_verify", action="store_true",
                        help="with --lstm_lazy, fail-stop when the deferred "
                             "read no longer hashes to the payload latched at "
                             "CLAIM")
    parser.add_argument("--weight_slots", type=int, default=2)
    parser.add_argument("--watchdog_s", type=float, default=300.0)
    parser.add_argument("--no_verify", action="store_true")
    parser.add_argument("--no_pin_assert", action="store_true")
    parser.add_argument("--no_det_algos", action="store_true")
    return parser


def main() -> int:
    args, rest = build_parser().parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    flags = mb.parser.parse_args(rest)
    result = run(args, flags)
    print(json.dumps({k: result[k] for k in
                      ("run_id", "mode", "status", "wall_s", "sps",
                       "event_counts", "l3")}, indent=2, default=str))
    print("L1 %s  L2 %s  events %d" % (
        result["layers"]["l1_digest"], result["layers"]["chain_final"],
        result["layers"]["events"]))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
