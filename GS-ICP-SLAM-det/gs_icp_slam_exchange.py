#!/usr/bin/env python3
# Copyright (c) 2024 Lab. of AI and Robotics (LAIR)
# Upstream-derived portions: MIT; see LICENSE and ../LICENSES/GS-ICP-SLAM-MIT.txt.
# Modified for this project: deterministic exchange integration and execution controls.
# Source: https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM/blob/5f996a872a979406b270fe0ee3b0a8f25c5e9ae3/gs_icp_slam.py
"""GS-ICP-SLAM under the deterministic logical exchange contract.

Driver for the exchange fork: same arguments as ``gs_icp_slam.py``, same
numerical code, but the two processes are coupled through ``detex`` instead of
through three shared flags and a 30 FPS wall-clock throttle.

    Y = F(X, seed, S)

X is the dataset plus the environment fingerprint, seed is ``GSICP_SEED``, and
S — here the closure-S rule "k credits per frame" — is an explicit input
recorded with the run, not whatever the OS scheduler produced this time.

Artefacts per run directory:

    tracker_poses.npy   estimated trajectory (N,4,4)
    gt_poses.npy        ground truth for the same frames
    map_state.npz       optimiser-visible map tensors (L3 end state)
    trace/*.l1.txt      per-process event traces, plus the canonical merge
    trace/run.l2.txt    chained hash over the merged trace
    exchange_run.json   config, schedule, digests, exit codes

Environment switches (beyond the inherited GSICP_* ones):

    GSICP_EXCHANGE_CREDITS   mapper iterations per frame (default 3)
    GSICP_EXCHANGE_FRAMES    stop after N frames (0 = whole sequence)
    GSICP_EXCHANGE_VERIFY    payload digest verification on/off (default on)
    GSICP_EXCHANGE_WATCHDOG  seconds a gate may block before aborting
    GSICP_EXCHANGE_SLOTS     slots per channel (default 2); an input to the
                             schedule's validity condition, so it is recorded
                             in the run report
    GSICP_EXCHANGE_SYNC      sync path, "M" or "E" (default M; E is deferred and
                             raises).  Recorded in the fingerprint because the
                             two paths are different contracts, not a tuning
                             knob, and results from them must not be pooled.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE), str(_HERE.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
import torch.multiprocessing as mp

from detex.schedule import ConstantCreditSchedule
from detex.trace import digest_of, write_layers

from detex_fabric import ExchangeFabric, env_flag, env_int
from gs_icp_slam import GS_ICP_SLAM
from mp_Mapper_exchange import ExchangeMapper
from mp_Tracker_exchange import ExchangeTracker

TARGET_CAPACITY = 10000000


class ExchangeSLAM(GS_ICP_SLAM):
    def __init__(self, args):
        super().__init__(args)

        self.credits = env_int("GSICP_EXCHANGE_CREDITS", 3)
        self.max_frames = env_int("GSICP_EXCHANGE_FRAMES", 0)
        # Slot count is an input to the schedule's validity condition
        # (LastUse_S(g) <= publish_frame(g + num_slots)), so a run's slot count
        # must be recoverable from its report or the schedule cannot be checked
        # after the fact.  Each extra CUDA slot is a full SharedGaussians buffer.
        self.num_slots = env_int("GSICP_EXCHANGE_SLOTS", 2)
        self.sync_path = os.environ.get("GSICP_EXCHANGE_SYNC", "M").strip() or "M"
        watchdog_s = float(env_int("GSICP_EXCHANGE_WATCHDOG", 900))
        verify = env_flag("GSICP_EXCHANGE_VERIFY", True)
        seed_raw = os.environ.get("GSICP_EXCHANGE_PERTURB_SEED")
        self.perturb_seed = int(seed_raw) if seed_raw not in (None, "") else None
        self.perturb_ms = float(os.environ.get("GSICP_EXCHANGE_PERTURB_MS", "0") or 0)
        self.start_order = os.environ.get("GSICP_EXCHANGE_START_ORDER", "tracker_first")

        self.run_dir = Path(self.output_path)
        self.trace_dir = self.run_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_trace_path = str(self.trace_dir / "tracker.l1.txt")
        self.mapper_trace_path = str(self.trace_dir / "mapper.l1.txt")
        self.map_state_path = str(self.run_dir / "map_state.npz")

        if self.trajmanager.which_dataset == "tum":
            total_frames = len(self.trajmanager.color_paths)
        else:
            total_frames = len(os.listdir(os.path.join(self.dataset_path, "images")))
        self.num_images = min(total_frames, self.max_frames) if self.max_frames else total_frames

        test_rgb_img, test_depth_img = self.get_test_image(f"{self.dataset_path}/images")
        from utils.graphics_utils import focal2fov
        cam_template = dict(
            FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
            image=test_rgb_img, depth_image=test_depth_img,
            cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)

        ctx = mp.get_context("spawn")
        self.fabric = ExchangeFabric(
            ctx,
            num_gaussian_points=int(self.shared_new_gaussians.xyz.shape[0]),
            num_target_points=TARGET_CAPACITY,
            cam_template=cam_template,
            num_slots=self.num_slots,
            watchdog_s=watchdog_s,
            verify=verify,
            sync_path=self.sync_path)
        self.schedule = ConstantCreditSchedule(self.credits, tail_credits=[self.credits])

        self.tracker = ExchangeTracker(self)
        self.mapper = ExchangeMapper(self)

    def run(self):
        started = time.time()
        ctx = mp.get_context("spawn")
        order = [0, 1] if self.start_order != "mapper_first" else [1, 0]
        built = {}
        for rank in order:
            target = self.tracking if rank == 0 else self.mapping
            process = ctx.Process(target=target, args=(rank,))
            process.start()
            built[rank] = process
        processes = [built[0], built[1]]
        for process in processes:
            process.join()
        return {
            "exit_codes": {"tracker": processes[0].exitcode, "mapper": processes[1].exitcode},
            "wall_s": round(time.time() - started, 3),
        }


def perturb_evidence(trace_dir: Path) -> dict:
    """What each process actually slept, read back from its own dump."""
    evidence = {}
    for actor in ("tracker", "mapper"):
        path = trace_dir / ("%s.perturb.json" % actor)
        if path.exists():
            evidence[actor] = json.loads(path.read_text())
    return evidence


def sync_cost(trace_dir: Path) -> dict:
    """Where each actor's wall time went, read back from its own dump.

    Reported per actor and never summed: the two processes run concurrently, so
    a total over both would be a number with no physical meaning.
    """
    cost = {}
    for actor in ("tracker", "mapper"):
        path = trace_dir / ("%s.cost.json" % actor)
        if path.exists():
            cost[actor] = json.loads(path.read_text())
    return cost


def cpu_model() -> str:
    """The CPU's model string, because closure turned out to depend on it.

    Every closure digest in this study up to 2026-08-22 came from Intel
    emeraldrapids nodes; a campaign that happened to land on an AMD zen4 node
    reproduced perfectly *within* that node and disagreed with every recorded
    digest.  The jobs pinned gpu_model and not cputype, so nothing in the
    reports said which CPU had produced them, and three rounds of diagnosis went
    into the wrong variable before the host names were compared.

    The GPU fingerprint was recorded from the start; this is the same
    discipline applied to the half of the computation that runs on the host.
    """
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment_fingerprint(sync_path: str = "M") -> dict:
    fingerprint = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sync_path": sync_path,
        "perturb_seed": os.environ.get("GSICP_EXCHANGE_PERTURB_SEED"),
        "perturb_ms": os.environ.get("GSICP_EXCHANGE_PERTURB_MS"),
        "start_order": os.environ.get("GSICP_EXCHANGE_START_ORDER", "tracker_first"),
        "leak_mode": os.environ.get("GSICP_EXCHANGE_LEAK", "none"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": os.environ.get("GSICP_SEED"),
        "deterministic_mapper": os.environ.get("GSICP_DETERMINISTIC_MAPPER"),
        "tracker_threads": os.environ.get("GSICP_TRACKER_THREADS"),
        "exchange_slots": os.environ.get("GSICP_EXCHANGE_SLOTS", "2"),
    }
    if torch.cuda.is_available():
        fingerprint["gpu_name"] = torch.cuda.get_device_name(0)
        fingerprint["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return fingerprint


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", help="dataset path", default="dataset/Replica/room0")
    parser.add_argument("--config", help="caminfo", default="configs/Replica/caminfo.txt")
    parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument("--keyframe_th", default=0.7)
    parser.add_argument("--knn_maxd", default=99999.0)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--overlapped_th", default=5e-4)
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    parser.add_argument("--downsample_rate", default=10)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save_results", action="store_true", default=None)
    parser.add_argument("--rerun_viewer", action="store_true", default=False)
    args = parser.parse_args()

    slam = ExchangeSLAM(args)
    outcome = slam.run()

    run_dir = Path(args.output_path)
    poses = slam.final_pose.numpy()[: slam.num_images]
    np.save(run_dir / "tracker_poses.npy", poses)
    gt = np.asarray(slam.trajmanager.gt_poses[: slam.num_images], dtype=np.float32)
    np.save(run_dir / "gt_poses.npy", gt)

    layers = write_layers(
        [slam.tracker_trace_path, slam.mapper_trace_path], slam.trace_dir, "run")

    map_state_digest = None
    map_state = run_dir / "map_state.npz"
    if map_state.exists():
        with np.load(map_state) as data:
            map_state_digest = digest_of(
                *[data[name].tobytes() for name in sorted(data.files)])

    report = {
        "run": str(run_dir),
        "status": "ok" if all(code == 0 for code in outcome["exit_codes"].values()) else "failed",
        "exit_codes": outcome["exit_codes"],
        "wall_s": outcome["wall_s"],
        "frames": slam.num_images,
        "schedule": slam.schedule.to_dict(),
        "verify": slam.fabric.verify,
        "perturbation": {"seed": slam.perturb_seed, "max_ms": slam.perturb_ms,
                         "start_order": slam.start_order,
                         "evidence": perturb_evidence(slam.trace_dir)},
        "environment": environment_fingerprint(slam.sync_path),
        "sync_cost": sync_cost(slam.trace_dir),
        "layers": {key: value for key, value in layers.items()},
        "pose_digest": digest_of(poses.tobytes()),
        "map_state_digest": map_state_digest,
        "arguments": vars(args),
    }
    (run_dir / "exchange_run.json").write_text(json.dumps(report, indent=2, default=str))

    print("[exchange] %s events=%d chain=%s poses=%s map=%s" % (
        report["status"], layers["events"], layers["chain_final"],
        report["pose_digest"], map_state_digest))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
