# Copyright (c) 2024 Lab. of AI and Robotics (LAIR)
# Upstream-derived portions: MIT; see LICENSE and ../LICENSES/GS-ICP-SLAM-MIT.txt.
# Modified for this project: deterministic exchange integration and execution controls.
# Source: https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/blob/5f996a872a979406b270fe0ee3b0a8f25c5e9ae3/mp_Mapper.py
"""Mapper host under the deterministic exchange contract — spec §2, §5, §7, §11.

The phase-table walk, credit accounting and EOS drain now live in
``detex.role.MapperRole``.  What is left here is the part only GS-ICP knows:

    extract()     read a claimed payload out, between CLAIM and ACK
    ingest()      add the keyframe to the map, and say what to publish back
    train_once()  exactly one credit's worth of training
    exit_digest() the optimiser-visible end state

The training step itself — render, losses, prune, optimizer — is transcribed
line for line from the frozen ``mp_Mapper.py``.  Three things the contract
changed:

    while True: (free-running, as many iterations as the clock allows)
        -> exactly m(t,b) credits per block, then MAP_BLOCK_DONE (spec §5)
    if self.is_tracking_keyframe_shared[0]: / elif self.is_mapping_keyframe...
        -> read the outcome for tag (t,0), then CLAIM (spec §5)
    self.target_gaussians_ready[0] = 1
        -> commit + TARGET_PUBLISH at P4, then LTC at P5 (spec §2)

The target snapshot is *computed* at ingest, exactly where the original computes
it — from the map with the new points added and this frame's training not yet
applied — and the role holds the filled slot until P4.  Publishing the value
computed after the credits would be a different algorithm, and this gate is
about the exchange, not about the algorithm.

Viewpoint sampling moves from the process-global ``random`` stream to the
stateless keyed RNG of spec §8, so the draw at (t, b, j) does not depend on how
many draws happened before it.
"""

from __future__ import annotations

import copy
import hashlib
import os

import numpy as np
import torch

from detex.errors import ProtocolError
from detex.perturb import Perturber
from detex.role import MapperRole, Publication, WriteSpec
from detex.tags import MAPPER
from detex.trace import digest_tensors

from detex_fabric import KIND_MAPPING, KIND_NONE, KIND_TRACKING
from mp_Mapper import Mapper
from utils.general_utils import seed_from_env
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_3


def keyed_choice(seed: int, operation: str, t: int, b: int, j: int, count: int) -> int:
    """Stateless keyed RNG — spec §8.

    ``R(component, operation, t, j, object-domain-version)``.  A seeded global
    stream would make every draw depend on how many draws came before it, so a
    schedule change would shift every subsequent sample; keying on the tag
    keeps the draw a function of where you are, not of how you got there.
    """
    key = "%d|mapper|%s|%d|%d|%d|%d" % (seed, operation, t, b, j, count)
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % count


class ExchangeMapper(Mapper):
    def __init__(self, slam):
        super().__init__(slam)
        self.fabric = slam.fabric
        self.schedule = slam.schedule
        self.recorder_path = slam.mapper_trace_path
        self.max_frames = slam.max_frames
        self.num_images = slam.num_images
        self.verify = slam.fabric.verify
        self.map_state_path = slam.map_state_path
        self.seed = int(os.environ.get("GSICP_SEED", "0"))
        self.leak_mode = os.environ.get("GSICP_EXCHANGE_LEAK", "none")
        self.perturb = Perturber(slam.perturb_seed, slam.perturb_ms, MAPPER)
        self._initialised = False
        self._state_digest = None

    def run(self):
        seed_from_env()
        self.prepare()
        MapperRole(self.fabric, self.schedule, self,
                   self.recorder_path, self.perturb,
                   leak_mode=self.leak_mode).run()
        self.report()

    # ---- host protocol -------------------------------------------------------

    def prepare(self):
        if self.rerun_viewer or self.verbose:
            raise ProtocolError(
                "UNSUPPORTED_MODE",
                "viewer/verbose introduce wall-clock branches; not allowed under "
                "the exchange contract")

    def frame_count(self):
        return int(self.num_images)

    def extract(self, name, view, tag):
        """Read one claimed payload out.  Valid only between CLAIM and ACK."""
        if name == "keyframe":
            return view.fields.get_values()
        if name == "camera":
            return copy.deepcopy(view.fields)
        raise ProtocolError("UNKNOWN_CHANNEL", "no extractor for channel",
                            channel=name)

    def ingest(self, t, extracted, kind, tag):
        points, colors, rots, scales, z_values, trackable_filter = extracted["keyframe"]
        newcam = extracted["camera"]
        newcam.on_cuda()

        publication = None
        if kind == KIND_TRACKING:
            if not self._initialised:
                self.gaussians.create_from_pcd2_tensor(
                    points, colors, rots, scales, z_values, trackable_filter)
                self.gaussians.spatial_lr_scale = self.scene_extent
                self.gaussians.training_setup(self)
                self.gaussians.update_learning_rate(1)
                self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
                self._initialised = True
            else:
                self.gaussians.add_from_pcd2_tensor(
                    points, colors, rots, scales, z_values, trackable_filter)

                target_points, target_rots, target_scales = \
                    self.gaussians.get_trackable_gaussians_tensor(
                        self.trackable_opacity_th)

                def fill_target(fields):
                    fields.input_values(target_points, target_rots, target_scales)

                publication = Publication(KIND_TRACKING, {
                    "target": WriteSpec(
                        fill=fill_target,
                        valid_length=lambda fields: int(fields.using_idx[0].item()),
                        digest=self.fabric.digests["target"]),
                })
        elif kind == KIND_MAPPING:
            self.gaussians.add_from_pcd2_tensor(
                points, colors, rots, scales, z_values, [])
        else:
            raise ProtocolError(
                "UNKNOWN_KIND", "keyframe outcome carries no kind", t=t, kind=kind)

        self.mapping_cams.append(newcam)
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams) - 1)
        return publication

    def exit_k(self):
        return self.train_iter

    def exit_digest(self):
        if self._state_digest is None:
            self._state_digest = self._save_map_state()
        return self._state_digest

    def report(self):
        print("[exchange] mapper credits=%d gaussians=%d" % (
            self.train_iter, self.gaussians.get_xyz.shape[0]))
        if self.save_results:
            self.gaussians.save_ply(os.path.join(self.output_path, "scene.ply"))

    # ---- the training step, transcribed from mp_Mapper.mapping() -------------

    def train_once(self, t: int, b: int, j: int) -> None:
        # A credit that silently does nothing would make MAP_BLOCK_DONE claim
        # work that never happened, and the credit trace is the thing the whole
        # gate rests on.  Frame 0 is always a tracking keyframe, so this cannot
        # fire; if it ever does, the schedule and the ingest path disagree.
        if len(self.mapping_cams) == 0:
            raise ProtocolError(
                "CREDIT_WITHOUT_WORK",
                "credit burned before any keyframe was ingested",
                t=t, b=b, j=j)

        if len(self.new_keyframes) > 0:
            train_idx = self.new_keyframes.pop(0)
            viewpoint_cam = self.mapping_cams[train_idx]
        else:
            train_idx = keyed_choice(
                self.seed, "view_select", t, b, j, len(self.mapping_cams))
            viewpoint_cam = self.mapping_cams[train_idx]

        if self.training_stage == 0:
            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth_image = viewpoint_cam.original_depth_image.cuda()
        elif self.training_stage == 1:
            gt_image = viewpoint_cam.rgb_level_1.cuda()
            gt_depth_image = viewpoint_cam.depth_level_1.cuda()
        elif self.training_stage == 2:
            gt_image = viewpoint_cam.rgb_level_2.cuda()
            gt_depth_image = viewpoint_cam.depth_level_2.cuda()

        self.training = True
        render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background,
                              training_stage=self.training_stage)

        depth_image = render_pkg["render_depth"]
        image = render_pkg["render"]

        mask = (gt_depth_image > 0.)
        mask = mask.detach()
        gt_image = gt_image * mask

        Ll1_map, Ll1 = l1_loss(image, gt_image)
        L_ssim_map, L_ssim = ssim(image, gt_image)

        d_max = 10.
        Ll1_d_map, Ll1_d = l1_loss(depth_image / d_max, gt_depth_image / d_max)

        loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
        loss_d = Ll1_d

        loss = loss_rgb + 0.1 * loss_d

        loss.backward()
        with torch.no_grad():
            if self.train_iter % 200 == 0:
                self.gaussians.prune_large_and_transparent(0.005, self.prune_th)

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.training = False
        self.train_iter += 1

    # ---- L3 end state --------------------------------------------------------

    def _save_map_state(self) -> str:
        """Dump the optimiser-visible map state for the bitwise A/B comparison.

        ``scene.ply`` is lossy for this purpose (it reorders and re-encodes);
        the closure claim needs the raw tensors.
        """
        tensors = {
            "xyz": self.gaussians._xyz.detach(),
            "features_dc": self.gaussians._features_dc.detach(),
            "features_rest": self.gaussians._features_rest.detach(),
            "opacity": self.gaussians._opacity.detach(),
            "scaling": self.gaussians._scaling.detach(),
            "rotation": self.gaussians._rotation.detach(),
        }
        arrays = {name: tensor.cpu().numpy() for name, tensor in tensors.items()}
        arrays["train_iter"] = np.array([self.train_iter], dtype=np.int64)
        if self.map_state_path:
            np.savez(self.map_state_path, **arrays)
        return digest_tensors([tensors[name] for name in sorted(tensors)])
