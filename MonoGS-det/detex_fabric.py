"""The exchange fabric for MonoGS — spec §1 and §11, phase table §5.1.

This is the whole of the contract's footprint inside the MonoGS port: a channel
topology plus the payload layout that replaces MonoGS's two
``torch.multiprocessing.Queue`` hand-offs.

Replaced by these objects (phase table §2, §3.2):

    backend_queue  ["init"] / ["keyframe"]            -> keyframe + camera channels
    frontend_queue ["init"]/["keyframe"]/["sync_backend"] -> mapstate channel
    ``requested_keyframe`` pending counter            -> LTC frontier + slot pin
    backend free-running ``map()`` + ``last_sent>=10``-> CreditBoard
    frontend 3 FPS wall-clock throttle                -> deleted (S carries tempo)
    ["pause"] / ["stop"]                              -> StopBoard three-phase EOS

Unlike GS-ICP, MonoGS has no pre-existing shared-memory payload objects: the
queues carry whole Python objects (a ``Camera`` ``nn.Module``, a deep-copied
``GaussianModel``) and rely on ``torch.multiprocessing``'s implicit CUDA IPC to
make that cheap.  There is therefore nothing to hand the channels as
``external_slots``; the payload is declared field by field with ``PayloadSpec``
and the hosts pack and unpack it.  That is also why the Gaussian cap is a hard
input here (phase table §2, plan §4: ``CAP_DEFAULT``) — the M->T channel is
pre-allocated, so exceeding it is a fail-stop, never a silent truncation.

Registration order is part of the run's identity (see ``detex.fabric``), so the
three ``register`` calls stay in phase-table §5.1 order: keyframe, camera,
mapstate.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from detex.channel import Channel, PayloadSpec
from detex.errors import require
from detex.fabric import TO_MAPPER, TO_TRACKER, FabricBase
from detex.trace import digest_tensors

# Outcome kinds.  The T->M direction carries MonoGS's two backend messages;
# the M->T direction has exactly one shape (``push_to_frontend``), so one kind.
KIND_NONE = 0
KIND_INIT = 1
KIND_KEYFRAME = 2
KIND_MAPSTATE = 1
KIND_NAMES = {KIND_NONE: "NONE", KIND_INIT: "INIT", KIND_KEYFRAME: "KEYFRAME"}

# Plan §4: cap = 2x the worst observed peak of the fr1_rpy logs (479,985).
CAP_DEFAULT = 1_000_000

CHANNEL_KEYFRAME = "keyframe"
CHANNEL_CAMERA = "camera"
CHANNEL_MAPSTATE = "mapstate"


class ConsumerAckChannel(Channel):
    """A channel whose consumer-side ACK records its own completion event.

    ``TrackerRole`` calls ``channel.last_ack(generation, tag)`` with no
    ``done_event`` (``detex/role.py:310``, ``:419``, ``:503``), while
    ``Channel.last_ack`` requires one for a CUDA-domain channel
    (``detex/channel.py:301-307``, ``ACK_WITHOUT_COMPLETION``).  The mismatch
    was invisible until now because the only M->T channel the driver had ever
    seen — GS-ICP's ``target`` — is CPU-domain.  MonoGS's ``mapstate`` is
    CUDA-domain (phase table §5.1), so the first frame would abort.

    The event is recorded here at exactly the point ``MapperRole`` records it
    for its own inbound CUDA channels (``detex/role.py:606-607``), so this is
    the same completion discipline, not a relaxation of it: the pin is handed
    back only after the consumer's reads have actually completed on the device.

    This is a **host-side subclass, not a detex edit** — zero lines of
    ``detex/`` change for this port.  The driver-side fix (thread ``done_event``
    through those three call sites, exactly as ``MapperRole`` already does) is
    written back to the spec as a gap the second real host exposed, for the
    organiser to rule on rather than for this port to absorb.
    """

    def last_ack(self, generation, tag, done_event=None,
                 _unsafe_skip_completion=False):
        if done_event is None and not _unsafe_skip_completion:
            done_event = self.sync.consumer_done_event(self.has_cuda)
        return super().last_ack(
            generation, tag, done_event=done_event,
            _unsafe_skip_completion=_unsafe_skip_completion)


def sh_rest_count(sh_degree: int) -> int:
    """Width of ``_features_rest`` — 0 at the RGB-D default ``sh_degree: 0``.

    For degree D, the DC coefficient tensor has shape (N, 1, 3), and the
    remaining coefficients have shape (N, (D+1)^2-1, 3). Interface reference:
    https://github.com/muskie82/MonoGS/blob/6c9254c319d8bff5caeef65259e6bb0941a9b9f6/gaussian_splatting/scene/gaussian_model.py
    This documents the adapter layout; it does not redistribute that source.
    """
    return (int(sh_degree) + 1) ** 2 - 1


# ---------------------------------------------------------------------------
# payload digests (spec §10: the snapshot must not change between READY and the
# last ACK).  Module level, and it has to be: the fabric is pickled into the
# spawned backend and a closure dies there with UNPICKLABLE_DIGEST
# (HARNESS_PORT_READOUT_20260824.md §6.1).  Only the valid prefix is hashed —
# the buffers are allocated for the Gaussian cap and their tails are stale by
# design.
# ---------------------------------------------------------------------------

def digest_keyframe(fields: Dict[str, Any]) -> str:
    """Fixed-shape payload: one frame's images.  No prefix to take."""
    return digest_tensors([
        fields["image"], fields["depth_gt"], fields["depth_prior"],
        fields["grad_mask"],
    ])


def digest_camera(fields: Dict[str, Any]) -> str:
    window_len = int(fields["window_len"][0].item())
    return digest_tensors([
        fields["uid"], fields["R"], fields["T"], fields["exposure"],
        fields["window"][:window_len], fields["window_len"],
        fields["intrinsics"],
    ])


def digest_mapstate(fields: Dict[str, Any]) -> str:
    n = int(fields["n"][0].item())
    vis_n = int(fields["vis_n"][0].item())
    kf_n = int(fields["kf_n"][0].item())
    return digest_tensors([
        fields["n"],
        fields["xyz"][:n], fields["f_dc"][:n], fields["f_rest"][:n],
        fields["opacity"][:n], fields["scaling"][:n], fields["rotation"][:n],
        fields["max_radii2D"][:n], fields["sh_degree"],
        fields["vis"][:vis_n, :n], fields["vis_ids"][:vis_n], fields["vis_n"],
        fields["kf_R"][:kf_n], fields["kf_T"][:kf_n], fields["kf_ids"][:kf_n],
        fields["kf_n"],
    ])


class MonoGSFabric(FabricBase):
    """Phase table §5.1: three channels, two slots each.

    ``domain`` overrides the per-channel domain for every channel at once.  It
    exists for the CPU protocol smoke: this box has no CUDA IPC, so the only
    thing testable locally is the protocol walk, and it has to be the *real*
    fabric declaration — same registration order, same field names, same
    module-level digests — with the payloads moved to host memory.
    """

    def __init__(
        self,
        ctx: Any,
        height: int,
        width: int,
        window_size: int,
        cap: int = CAP_DEFAULT,
        sh_degree: int = 0,
        num_slots: int = 2,
        watchdog_s: float = 900.0,
        verify: bool = True,
        sync_path: str = "M",
        domain: Optional[str] = None,
    ) -> None:
        super().__init__(ctx, num_slots=num_slots, watchdog_s=watchdog_s,
                         verify=verify, sync_path=sync_path)
        require(domain in (None, "cpu", "cuda"), "CHANNEL_DOMAIN",
                "fabric domain override must be None, 'cpu' or 'cuda'",
                domain=domain)

        self.height = int(height)
        self.width = int(width)
        self.window_size = int(window_size)
        self.cap = int(cap)
        self.sh_degree = int(sh_degree)
        self.sh_rest = sh_rest_count(sh_degree)
        self.domain_override = domain

        H, W_img = self.height, self.width
        W = self.window_size
        cap = self.cap
        rest = self.sh_rest

        kf_domain = domain or "cuda"
        cam_domain = domain or "cpu"
        map_domain = domain or "cuda"
        self.channel_domains = {
            CHANNEL_KEYFRAME: kf_domain,
            CHANNEL_CAMERA: cam_domain,
            CHANNEL_MAPSTATE: map_domain,
        }

        # D1: tracker -> mapper.  The images ``request_keyframe`` used to hand
        # over inside the ``Camera`` object (slam_frontend.py:288-291), plus the
        # depth prior ``add_new_keyframe`` computes (:106-108).
        keyframe_specs = [
            PayloadSpec("image", (3, H, W_img), torch.float32, kf_domain),
            PayloadSpec("depth_gt", (H, W_img), torch.float32, kf_domain),
            PayloadSpec("depth_prior", (H, W_img), torch.float32, kf_domain),
            PayloadSpec("grad_mask", (H, W_img), torch.uint8, kf_domain),
        ]
        # D2: tracker -> mapper, CPU domain.  Pose, exposure, the window the
        # tracker decided on (slam_frontend.py:436) and the intrinsics the
        # mapper needs to rebuild the Camera.  ``cam_rot_delta`` /
        # ``cam_trans_delta`` are deliberately absent: ``update_pose`` zeroes
        # them on every call (utils/pose_utils.py:91-92), so they are identically
        # zero at publish time and the mapper builds fresh ones.
        camera_specs = [
            PayloadSpec("uid", (1,), torch.int64, cam_domain),
            PayloadSpec("R", (3, 3), torch.float32, cam_domain),
            PayloadSpec("T", (3,), torch.float32, cam_domain),
            PayloadSpec("exposure", (2,), torch.float32, cam_domain),
            PayloadSpec("window", (W,), torch.int64, cam_domain),
            PayloadSpec("window_len", (1,), torch.int64, cam_domain),
            PayloadSpec("intrinsics", (6,), torch.float64, cam_domain),
        ]
        # D3: mapper -> tracker.  ``push_to_frontend`` (slam_backend.py:355-365)
        # deep-copies the whole GaussianModel, optimizer state and config
        # included; the frontend only ever reads what ``render`` needs
        # (gaussian_renderer/__init__.py:76-141) plus the visibility map and the
        # keyframe poses, so only those cross.
        mapstate_specs = [
            PayloadSpec("n", (1,), torch.int64, map_domain),
            PayloadSpec("xyz", (cap, 3), torch.float32, map_domain),
            PayloadSpec("f_dc", (cap, 1, 3), torch.float32, map_domain),
            PayloadSpec("f_rest", (cap, rest, 3), torch.float32, map_domain),
            PayloadSpec("opacity", (cap, 1), torch.float32, map_domain),
            PayloadSpec("scaling", (cap, 3), torch.float32, map_domain),
            PayloadSpec("rotation", (cap, 4), torch.float32, map_domain),
            PayloadSpec("max_radii2D", (cap,), torch.float32, map_domain),
            PayloadSpec("sh_degree", (1,), torch.int64, map_domain),
            PayloadSpec("vis", (W, cap), torch.uint8, map_domain),
            PayloadSpec("vis_ids", (W,), torch.int64, map_domain),
            PayloadSpec("vis_n", (1,), torch.int64, map_domain),
            PayloadSpec("kf_R", (W, 3, 3), torch.float32, map_domain),
            PayloadSpec("kf_T", (W, 3), torch.float32, map_domain),
            PayloadSpec("kf_ids", (W,), torch.int64, map_domain),
            PayloadSpec("kf_n", (1,), torch.int64, map_domain),
        ]

        common = dict(ctx=ctx, abort=self.abort, sync=self.sync,
                      num_slots=num_slots, watchdog_s=watchdog_s)
        self.keyframe = self.register(CHANNEL_KEYFRAME, Channel(
            CHANNEL_KEYFRAME, keyframe_specs, domain=kf_domain, **common),
            digest_keyframe, TO_MAPPER)
        self.camera = self.register(CHANNEL_CAMERA, Channel(
            CHANNEL_CAMERA, camera_specs, domain=cam_domain, **common),
            digest_camera, TO_MAPPER)
        self.mapstate = self.register(CHANNEL_MAPSTATE, ConsumerAckChannel(
            CHANNEL_MAPSTATE, mapstate_specs, domain=map_domain, **common),
            digest_mapstate, TO_TRACKER)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "window_size": self.window_size,
            "cap": self.cap,
            "sh_degree": self.sh_degree,
            "sh_rest": self.sh_rest,
            "num_slots": self.num_slots,
            "verify": self.verify,
            "domain_override": self.domain_override,
            "channel_domains": dict(self.channel_domains),
            "channel_order": list(self.channel_order),
        }


class MonoGSCreditSchedule:
    """closure-S for MonoGS: S is a *rule*, not a table (spec §6).

    Phase table §4 and plan §4 fix it:

        m(0, 0)    = ``Training.init_itr_num`` — the ``initialize_map`` block
                     (slam_backend.py:86-140), which upstream also runs exactly
                     once, at the ``init`` message.
        m(t>0, 0)  = ``k`` (default 10 = upstream's async ``iter_per_kf``,
                     slam_backend.py:421).  Upstream's work is keyframe-
                     conditional (10 + 1 prune on a keyframe, 1 per free-running
                     round otherwise); a constant k is a *different but legal* S,
                     and the first round only claims closure, not equivalence.
                     Making m depend on the outcome kind is the optional H-B of
                     phase table §4 and would cost a ``detex/role.py:626`` edit.
        prune      happens at ``j == m-1`` — that is upstream's
                     ``self.map(window, prune=True)`` after
                     ``self.map(window, iters=iter_per_kf)`` (:473-474), folded
                     into the single block the driver walks
                     (``detex/role.py:356-361``, ``BLOCKS_UNSUPPORTED``).

    ``kf(t)`` / ``pub(t)`` / ``u(t)`` are not tabulated: the keyframe decision
    reads the rendered visibility (slam_frontend.py:415), so it is a function of
    (X, seed, exchange history) and is recorded in the trace rather than fixed
    in advance.  ``tail_credits`` is empty because ``eval_rendering`` is off and
    ``color_refinement`` is therefore not enabled (phase table §4, H-A).

    The role calls exactly ``schedule.m(t, b)`` (``detex/role.py:626``) and
    ``schedule.tail_credits`` (``:664``); no detex change is needed to swap a
    rule object in for ``ConstantCreditSchedule``.
    """

    def __init__(self, init_itr_num: int, k: int = 10, blocks_per_frame: int = 1,
                 tail_credits: Any = (), name: str = "monogs-closure-S") -> None:
        self.init_itr_num = int(init_itr_num)
        self.k = int(k)
        self.blocks_per_frame = int(blocks_per_frame)
        self.tail_credits: List[int] = [int(c) for c in tail_credits]
        self.name = "%s-init%d-k%d" % (name, self.init_itr_num, self.k)

    def m(self, t: int, b: int) -> int:
        return self.init_itr_num if int(t) == 0 else self.k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": "rule",
            "m": "init_itr_num=%d at t=0, %d per frame at t>0" % (
                self.init_itr_num, self.k),
            "init_itr_num": self.init_itr_num,
            "k": self.k,
            "prune_at": "j == m-1",
            "blocks_per_frame": self.blocks_per_frame,
            "tail_credits": self.tail_credits,
            "kf": "data-dependent (is_keyframe), recorded in trace",
            "pub": "every frame (mapper publishes mapstate unconditionally)",
            "u": "latest generation, outcome-driven, recorded in trace",
        }


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)
