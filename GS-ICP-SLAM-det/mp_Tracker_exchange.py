# Copyright (c) 2024 Lab. of AI and Robotics (LAIR)
# Upstream-derived portions: MIT; see LICENSE and ../LICENSES/GS-ICP-SLAM-MIT.txt.
# Modified for this project: deterministic exchange integration and execution controls.
# Source: https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/blob/5f996a872a979406b270fe0ee3b0a8f25c5e9ae3/mp_Tracker.py
"""Tracker host under the deterministic exchange contract — spec §2, §5, §7, §11.

The phase-table walk, CLAIM/ACK sequencing and three-phase stop now live in
``detex.role.TrackerRole``.  What is left here is the part only GS-ICP knows:

    consume()       install a claimed target into the FastGICP registration
    produce()       one frame of tracking, and what to publish afterwards
    frame_digest()  the pose this frame committed
    on_stop()       hand the trajectory back through shared memory

Every numerical line — downsampling, ``reg.align``, the keyframe decision, the
quaternion composition, the overlap elimination — is transcribed unchanged from
the frozen ``mp_Tracker.py``.  What the contract replaced is exclusively the
hand-off:

    while self.is_tracking_keyframe_shared[0] or ...: time.sleep(1e-15)
        -> gate predicate + slot claim (spec §5)
    while not self.target_gaussians_ready[0]: time.sleep(1e-15)
        -> LTC wait, then outcome read (two separate steps, spec §5)
    while 1/((time.time()-start)/(i+1)) > 30.: time.sleep(1e-15)
        -> deleted; the logical tempo is S's job, not the clock's
    self.end_of_dataset[0] = 1
        -> three-phase stop (spec §7)

One ordering difference from the original, and it is a semantic no-op: the
original blocks for the fresh target at the *end* of the tracking-keyframe frame
t, this consumes it at P0 of frame t+1.  Both install it before ``align`` of
frame t+1, which is the target's only use point (contract C6).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from detex.errors import ProtocolError
from detex.perturb import Perturber
from detex.role import Publication, TrackerRole, WriteSpec
from detex.tags import TRACKER
from detex.trace import digest_tensors

from detex_fabric import KIND_MAPPING, KIND_NONE, KIND_TRACKING
from mp_Tracker import Tracker
from utils.general_utils import seed_from_env


class ExchangeTracker(Tracker):
    def __init__(self, slam):
        super().__init__(slam)
        self.fabric = slam.fabric
        self.schedule = slam.schedule
        self.recorder_path = slam.tracker_trace_path
        self.max_frames = slam.max_frames
        self.verify = slam.fabric.verify
        self.perturb = Perturber(slam.perturb_seed, slam.perturb_ms, TRACKER)
        self._pbar = None

    def run(self):
        seed_from_env()
        self.prepare()
        TrackerRole(self.fabric, self.schedule, self,
                    self.recorder_path, self.perturb).run()
        self.report()

    # ---- host protocol -------------------------------------------------------

    def prepare(self):
        if self.rerun_viewer or self.verbose or self.demo[0]:
            raise ProtocolError(
                "UNSUPPORTED_MODE",
                "viewer/verbose/demo introduce wall-clock branches; not allowed "
                "under the exchange contract")

        self.rgb_images, self.depth_images = self.get_images(f"{self.dataset_path}/images")
        self.num_images = len(self.rgb_images)
        if self.max_frames:
            self.num_images = min(self.num_images, int(self.max_frames))
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)

        self.total_start_time = time.time()
        self._pbar = tqdm(total=self.num_images)

    def frame_count(self):
        return self.num_images

    def consume(self, name, view, tag):
        """Install the claimed target generation — the only inbound channel."""
        target_points, target_rots, target_scales = view.fields.get_values_np()
        self.reg.set_input_target(target_points)
        self.reg.set_target_covariances_fromqs(
            target_rots.flatten(), target_scales.flatten())

    def produce(self, t):
        self.iter_shared[0] = t
        self.iteration_images = t

        current_image = self.rgb_images.pop(0)
        depth_image = self.depth_images.pop(0)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)

        points, colors, z_values, trackable_filter = \
            self.downsample_and_make_pointcloud2(depth_image, current_image)

        kind = KIND_NONE
        rots = scales = None
        if t == 0:
            current_pose = self.poses[-1]

            current_pose = np.linalg.inv(current_pose)
            T = current_pose[:3, 3]
            R = current_pose[:3, :3].transpose()

            points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
            self.reg.set_input_target(points)

            num_trackable_points = trackable_filter.shape[0]
            input_filter = np.zeros(points.shape[0], dtype=np.int32)
            input_filter[(trackable_filter)] = [range(1, num_trackable_points + 1)]

            self.reg.set_target_filter(num_trackable_points, input_filter)
            self.reg.calculate_target_covariance_with_filter()

            rots = self.reg.get_target_rotationsq()
            scales = self.reg.get_target_scales()
            rots = np.reshape(rots, (-1, 4))
            scales = np.reshape(scales, (-1, 3))
            kind = KIND_TRACKING
        else:
            self.reg.set_input_source(points)
            num_trackable_points = trackable_filter.shape[0]
            input_filter = np.zeros(points.shape[0], dtype=np.int32)
            input_filter[(trackable_filter)] = [range(1, num_trackable_points + 1)]
            self.reg.set_source_filter(num_trackable_points, input_filter)

            initial_pose = self.poses[-1]
            current_pose = self.reg.align(initial_pose)
            self.poses.append(current_pose)

            current_pose = np.linalg.inv(current_pose)
            T = current_pose[:3, 3]
            R = current_pose[:3, :3].transpose()

            points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
            target_corres, distances = self.reg.get_source_correspondence()

            len_corres = len(np.where(distances < self.overlapped_th)[0])

            if (self.iteration_images >= self.num_images - 1
                    or len_corres / distances.shape[0] < self.keyframe_th):
                if_tracking_keyframe = True
                self.from_last_tracking_keyframe = 0
            else:
                if_tracking_keyframe = False
                self.from_last_tracking_keyframe += 1

            if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
                if_mapping_keyframe = True
            else:
                if_mapping_keyframe = False

            if if_tracking_keyframe:
                rots = np.array(self.reg.get_source_rotationsq())
                rots = np.reshape(rots, (-1, 4))

                R_d = Rotation.from_matrix(R)
                R_d_q = R_d.as_quat()
                rots = self.quaternion_multiply(R_d_q, rots)

                scales = np.array(self.reg.get_source_scales())
                scales = np.reshape(scales, (-1, 3))

                not_overlapped_indices_of_trackable_points = \
                    self.eliminate_overlapped2(distances, self.overlapped_th2)
                trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]
                kind = KIND_TRACKING
            elif if_mapping_keyframe:
                rots = np.array(self.reg.get_source_rotationsq())
                rots = np.reshape(rots, (-1, 4))

                R_d = Rotation.from_matrix(R)
                R_d_q = R_d.as_quat()
                rots = self.quaternion_multiply(R_d_q, rots)

                scales = np.array(self.reg.get_source_scales())
                scales = np.reshape(scales, (-1, 3))
                kind = KIND_MAPPING

        self._pbar.update(1)
        if kind == KIND_NONE:
            return Publication.nothing()

        depth_for_cam = depth_image.astype(np.float32) / self.depth_scale

        def fill_keyframe(fields):
            fields.input_values(
                torch.tensor(points), torch.tensor(colors),
                torch.tensor(rots), torch.tensor(scales),
                torch.tensor(z_values), torch.tensor(trackable_filter))

        def fill_camera(fields):
            fields.setup_cam(R, T, current_image, depth_for_cam)
            fields.cam_idx[0] = t

        return Publication(kind, {
            "keyframe": WriteSpec(
                fill=fill_keyframe,
                valid_length=lambda fields: int(points.shape[0]),
                digest=self.fabric.digests["keyframe"]),
            "camera": WriteSpec(
                fill=fill_camera,
                valid_length=lambda fields: t,
                digest=self.fabric.digests["camera"]),
        })

    def frame_digest(self, t):
        return digest_tensors([torch.from_numpy(np.asarray(self.poses[-1]))])

    def on_stop(self):
        if self._pbar is not None:
            self._pbar.close()
        poses = torch.tensor(np.asarray(self.poses)).float()
        self.final_pose[:poses.shape[0], :, :] = poses
        self._final_poses = poses

    def exit_digest(self):
        return digest_tensors([self._final_poses])

    def report(self):
        poses = self._final_poses
        print(f"[exchange] tracker frames={self.num_images} poses={poses.shape[0]}")
        print(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses[:poses.shape[0]], self.poses)*100.:.2f}")
