"""The exchange fabric for GS-ICP-SLAM — spec §1 and §11.

This is the whole of the contract's footprint inside the SLAM fork: a channel
topology declared over the *existing* shared objects.  ``SharedGaussians``,
``SharedCam`` and ``SharedTargetPoints`` are handed to the channels as external
slots, one per buffer, so no numerical path changes — only who may touch which
buffer, and when.

Replaced by these objects (spec §11):

    is_tracking_keyframe_shared / is_mapping_keyframe_shared  -> keyframe/camera
                                                                 outcomes + CLAIM/ACK
    target_gaussians_ready                                    -> target outcome + LTC
    free-running mapper loop                                  -> CreditBoard
    end_of_dataset race                                       -> StopBoard

Boards, abort, recorder attachment and the state dump come from
``detex.fabric.FabricBase``; what is left below is only what GS-ICP itself
knows — which buffers exist, which way they run, and how to hash them.
Registration order is part of the run's identity (see FabricBase), so the three
``register`` calls stay in the order the original hand-off used.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from scene.shared_objs import SharedCam, SharedGaussians, SharedTargetPoints

from detex.channel import Channel
from detex.fabric import TO_MAPPER, TO_TRACKER, FabricBase
from detex.schedule import ConstantCreditSchedule
from detex.trace import digest_tensors

KIND_NONE = 0
KIND_MAPPING = 1
KIND_TRACKING = 2
KIND_NAMES = {KIND_NONE: "NONE", KIND_MAPPING: "MAPPING", KIND_TRACKING: "TRACKING"}


class ExchangeFabric(FabricBase):
    def __init__(
        self,
        ctx: Any,
        num_gaussian_points: int,
        num_target_points: int,
        cam_template: Dict[str, Any],
        num_slots: int = 2,
        watchdog_s: float = 600.0,
        verify: bool = True,
        sync_path: str = "M",
    ) -> None:
        super().__init__(ctx, num_slots=num_slots, watchdog_s=watchdog_s,
                         verify=verify, sync_path=sync_path)

        def slots(factory):
            made = []
            for _ in range(self.num_slots):
                obj = factory()
                obj.share_memory()
                made.append(obj)
            return made

        common = dict(ctx=ctx, abort=self.abort, sync=self.sync, watchdog_s=watchdog_s)
        # D1: tracker -> mapper, CUDA domain (SharedGaussians lives on the device)
        self.keyframe = self.register("keyframe", Channel(
            "keyframe", [], external_slots=slots(lambda: SharedGaussians(num_gaussian_points)),
            domain="cuda", **common), digest_gaussians, TO_MAPPER)
        # D2: tracker -> mapper, CPU domain
        self.camera = self.register("camera", Channel(
            "camera", [], external_slots=slots(lambda: SharedCam(**cam_template)),
            domain="cpu", **common), digest_camera, TO_MAPPER)
        # D3: mapper -> tracker, CPU domain
        self.target = self.register("target", Channel(
            "target", [], external_slots=slots(lambda: SharedTargetPoints(num_target_points)),
            domain="cpu", **common), digest_target, TO_TRACKER)


# ---------------------------------------------------------------------------
# payload digests (spec §10: the snapshot must not change between READY and the
# last ACK).  Only the valid prefix is hashed — the buffers are allocated for
# the worst case and their tails are stale by design.
# ---------------------------------------------------------------------------

def digest_gaussians(shared: SharedGaussians) -> str:
    used = int(shared.using_idx[0].item())
    size = int(shared.filter_size[0].item())
    return digest_tensors([
        shared.xyz[:used, :], shared.colors[:used, :], shared.rots[:used, :],
        shared.scales[:used, :], shared.z_values[:used],
        shared.trackable_filter[:size],
    ])


CAMERA_FIELDS = ("R", "t", "cam_idx", "original_image", "original_depth_image",
                 "world_view_transform", "full_proj_transform", "camera_center")


def digest_camera(shared: SharedCam) -> str:
    fields = [getattr(shared, name) for name in CAMERA_FIELDS]
    _field_trace("camera", CAMERA_FIELDS, fields)
    return digest_tensors(fields)


def _field_trace(channel: str, names, tensors) -> None:
    """Opt-in per-field digests, for localising a payload difference.

    A combined payload digest says *that* two runs differ, never *where*.  When
    a difference has to be tracked down — the 2-slot / 3-slot split found by
    G-FULL is the case this was written for — set
    ``GSICP_EXCHANGE_FIELD_DIGESTS`` to a path and every digested field is
    appended there.

    Off by default and untouched by any published gate: enabling it adds host
    copies and file writes, which is exactly the kind of thing that must not be
    silently present in a run whose digests are on record.
    """
    path = os.environ.get("GSICP_EXCHANGE_FIELD_DIGESTS")
    if not path:
        return
    with open(path, "a") as handle:
        for name, tensor in zip(names, tensors):
            handle.write("%s,%s,%s\n" % (channel, name, digest_tensors([tensor])))


def digest_target(shared: SharedTargetPoints) -> str:
    used = int(shared.using_idx[0].item())
    return digest_tensors([
        shared.xyz[:used, :], shared.rots[:used, :], shared.scales[:used, :]])


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
