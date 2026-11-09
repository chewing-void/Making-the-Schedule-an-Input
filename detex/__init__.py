"""detex — deterministic logical exchange contract for async coupled GS-SLAM.

The package is deliberately free of any SLAM dependency. Host-specific
adapters live in the three sibling ``*-det`` directories.
"""

from __future__ import annotations

from .errors import ProtocolError, require
from .tags import (
    ACTORS,
    MAPPER,
    PHASES,
    PHASE_ACTOR,
    PHASE_INDEX,
    TRACKER,
    Tag,
    make_tag,
    parse_tag,
)
from .shared import Abort
from .sync import SyncPath
from .channel import Channel, Outcome, PayloadSpec, SlotView, FREE, WRITING, READY, READING
from .boards import CreditBoard, LtcBoard, StopBoard
from .schedule import Schedule
from .fabric import FabricBase, TO_MAPPER, TO_TRACKER
from .role import MapperRole, Publication, TrackerRole, WriteSpec
from .trace import EventRecorder, chain_hash, digest_of, digest_tensors, merge_traces

__all__ = [
    "Abort",
    "ACTORS",
    "Channel",
    "Outcome",
    "CreditBoard",
    "EventRecorder",
    "FabricBase",
    "FREE",
    "LtcBoard",
    "MapperRole",
    "MAPPER",
    "PHASES",
    "PHASE_ACTOR",
    "PHASE_INDEX",
    "PayloadSpec",
    "Publication",
    "ProtocolError",
    "READING",
    "READY",
    "Schedule",
    "SlotView",
    "StopBoard",
    "SyncPath",
    "TO_MAPPER",
    "TO_TRACKER",
    "TRACKER",
    "TrackerRole",
    "Tag",
    "WRITING",
    "WriteSpec",
    "chain_hash",
    "digest_of",
    "digest_tensors",
    "make_tag",
    "merge_traces",
    "parse_tag",
    "require",
]
