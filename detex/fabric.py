"""Fabric base — the channel topology a host declares, and the boards it gets free.

A host says which channels exist, which direction each runs, and how to digest
its payload.  Everything else — the three control boards, the abort flag, the
sync path, recorder attachment, the state dump — is the same for every host and
is built here.

The direction lists are not decoration: ``TrackerRole`` and ``MapperRole`` use
them to decide who writes, who claims, and in what order, and ``channel_order``
fixes the sequence in which quiescence is asserted at Q(stop).  Declaration
order is therefore part of the run's identity — reordering the channels changes
the recorded event sequence, so a host must not shuffle them between runs it
intends to compare.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .boards import CreditBoard, LtcBoard, StopBoard
from .errors import require
from .shared import Abort
from .sync import SyncPath

TO_MAPPER = "to_mapper"
TO_TRACKER = "to_tracker"


class FabricBase:
    """Channels plus boards.  Subclass it, build the channels, register them."""

    def __init__(self, ctx: Any, num_slots: int = 2, watchdog_s: float = 600.0,
                 verify: bool = True, sync_path: str = "M") -> None:
        self.abort = Abort(ctx)
        self.sync = SyncPath(sync_path)
        self.verify = bool(verify)
        self.num_slots = int(num_slots)

        self.channels: Dict[str, Any] = {}
        self.digests: Dict[str, Callable[[Any], str]] = {}
        self.to_mapper: List[str] = []
        self.to_tracker: List[str] = []

        self.ltc = LtcBoard(ctx, self.abort, watchdog_s=watchdog_s)
        self.credit = CreditBoard(ctx, self.abort, watchdog_s=watchdog_s)
        self.stop = StopBoard(ctx, self.abort, watchdog_s=watchdog_s)

    # ---- topology ------------------------------------------------------------

    def register(self, name: str, channel: Any, digest: Callable[[Any], str],
                 direction: str) -> Any:
        require(direction in (TO_MAPPER, TO_TRACKER), "CHANNEL_DIRECTION",
                "direction must be to_mapper or to_tracker",
                channel=name, direction=direction)
        require(name not in self.channels, "CHANNEL_DUPLICATE",
                "channel already registered", channel=name)
        self.channels[name] = channel
        self.digests[name] = digest
        (self.to_mapper if direction == TO_MAPPER else self.to_tracker).append(name)
        return channel

    @property
    def channel_order(self) -> List[str]:
        return self.to_mapper + self.to_tracker

    # ---- lifecycle -----------------------------------------------------------

    def objects(self) -> List[Any]:
        return ([self.channels[name] for name in self.channel_order]
                + [self.ltc, self.credit, self.stop])

    def attach(self, actor: int, recorder: Any) -> None:
        for obj in self.objects():
            obj.attach(actor, recorder)

    def dump(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            name: self.channels[name].dump() for name in self.channel_order}
        state["ltc"] = None if self.ltc.current() is None else str(self.ltc.current())
        return state
