"""Trace layers L1/L2/L3 — spec §9.

L1 is one line per contract-visible event.  Each process writes its own file;
the two files are merged **by canonical key**, never by arrival order, so the
merged trace is a pure function of the logical run:

    sort key = (t, b, phase, actor, per-actor sequence number)

The per-actor sequence number only breaks ties between two events the same
actor emits inside one phase; the phase table (spec §2) already assigns at most
one actor to each phase.

L2 chains a hash over the merged L1 lines.  The spec writes it as an inductive
per-event hash; computing it offline over the canonically merged sequence gives
the identical chain without forcing a shared serialisation point between the
two processes, which would itself be a synchronisation the contract does not
want.  First differing chain entry = first divergent event.

L3 is the end-state digest (poses, map tensors, RNG), produced by the caller
through ``digest_tensors``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .tags import ACTORS, PHASES, PHASE_INDEX, Tag

DIGEST_HEX = 16          # payload digests: 8 bytes rendered as 16 hex chars
CHAIN_HEX = 32           # chain hash: 16 bytes


def _blake(data: bytes, size: int) -> str:
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def digest_of(*chunks: bytes) -> str:
    """Stable 16-hex-char digest of an ordered byte sequence."""
    h = hashlib.blake2b(digest_size=DIGEST_HEX // 2)
    for chunk in chunks:
        h.update(len(chunk).to_bytes(8, "little"))
        h.update(chunk)
    return h.hexdigest()


def digest_tensors(tensors: Sequence[Any]) -> str:
    """Digest of raw tensor bytes, including dtype and shape.

    CUDA tensors are copied host-side to be hashed; the caller must only do
    this inside a region where the payload is protocol-stable (between CLAIM
    and LAST_ACK for a read, before COMMIT for a write).
    """
    chunks: List[bytes] = []
    for tensor in tensors:
        host = tensor.detach().contiguous().cpu()
        chunks.append(str(host.dtype).encode())
        chunks.append(str(tuple(host.shape)).encode())
        chunks.append(host.numpy().tobytes())
    return digest_of(*chunks)


class EventRecorder:
    """Per-process L1 sink.  One line per contract-visible event."""

    def __init__(self, path: Any, actor: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.actor = int(actor)
        self._fh = self.path.open("w")
        self._seq = 0

    def record(self, tag: Tag, event: str, generation=None, k=None, digest=None) -> None:
        line = "%d,%d,%s|%s|%d|%s|%s|%s|%s" % (
            tag.t,
            tag.b,
            PHASES[tag.phase],
            ACTORS[self.actor],
            self._seq,
            event,
            "-" if generation is None else str(int(generation)),
            "-" if k is None else str(int(k)),
            "-" if digest is None else str(digest),
        )
        self._seq += 1
        self._fh.write(line + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def _sort_key(line: str):
    address, actor, seq, _rest = line.split("|", 3)
    t, b, phase = address.split(",")
    return (int(t), int(b), PHASE_INDEX[phase], ACTORS.index(actor), int(seq))


def merge_traces(paths: Iterable[Any]) -> List[str]:
    """Canonically merge per-process L1 files (spec §9)."""
    lines: List[str] = []
    for path in paths:
        with Path(path).open() as fh:
            lines.extend(line.rstrip("\n") for line in fh if line.strip())
    lines.sort(key=_sort_key)
    return lines


def chain_hash(lines: Sequence[str]) -> List[str]:
    """L2: h(e) = H(h(prev), event-line).  Returns the per-event chain."""
    chain: List[str] = []
    previous = "0" * CHAIN_HEX
    for line in lines:
        previous = _blake((previous + "\n" + line).encode(), CHAIN_HEX // 2)
        chain.append(previous)
    return chain


def write_layers(trace_paths: Iterable[Any], out_dir: Any, stem: str) -> Dict[str, Any]:
    """Write the merged L1 and the L2 chain; return their summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = merge_traces(trace_paths)
    chain = chain_hash(lines)
    l1_path = out_dir / ("%s.l1.txt" % stem)
    l2_path = out_dir / ("%s.l2.txt" % stem)
    l1_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    l2_path.write_text("\n".join(chain) + ("\n" if chain else ""))
    return {
        "l1_path": str(l1_path),
        "l2_path": str(l2_path),
        "events": len(lines),
        "l1_digest": digest_of("\n".join(lines).encode()),
        "chain_final": chain[-1] if chain else None,
    }


def first_divergence(lines_a: Sequence[str], lines_b: Sequence[str]) -> Dict[str, Any]:
    """Locate the first differing merged event between two runs."""
    for index, (left, right) in enumerate(zip(lines_a, lines_b)):
        if left != right:
            return {"index": index, "a": left, "b": right}
    if len(lines_a) != len(lines_b):
        longer = lines_a if len(lines_a) > len(lines_b) else lines_b
        index = min(len(lines_a), len(lines_b))
        return {
            "index": index,
            "a": lines_a[index] if longer is lines_a else None,
            "b": lines_b[index] if longer is lines_b else None,
            "note": "length mismatch",
        }
    return {}
