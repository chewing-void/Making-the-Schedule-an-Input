# License scope for this partial release

The root `LICENSE` retains the project's Apache-2.0 grant for original detex
implementation, original adapter contributions, and new release documentation
and verification tooling. It does not relicense third-party content.

## File scope

- `detex/*.py`, `GS-ICP-SLAM-det/detex_fabric.py`,
  `MonoGS-det/detex_fabric.py`, `torchbeast-det/m6_detex_fabric.py`: project
  implementations/adapters under the existing Apache-2.0 project declaration.
  Static comparison supports the MonoGS fabric's interface-only scope, but
  cannot establish authorship or exclude every possible prior source.
- The six other retained GS-ICP Python files contain copied or adapted logic
  from GS-ICP at the fixed revision in `SOURCE_MAP.json`. MIT applies to those
  upstream portions, copyright (c) 2024 Lab. of AI and Robotics (LAIR). The
  existing `GS-ICP-SLAM-det/LICENSE` matches the fixed upstream license byte for
  byte. Original integration contributions do not remove upstream obligations.
- `torchbeast-det/monobeast_exchange.py`: project changes plus upstream-adapted
  TorchBeast logic, Apache-2.0. The exact Facebook header is restored, and the
  upstream license is included separately in `LICENSES/`.
- Both rasterizer `.patch` files: Gaussian-Splatting license, **not Apache-2.0**.
  Its noncommercial research/evaluation limitations and redistribution conditions
  remain in force. Distribution is not permission for commercial use.
- FastGICP `.patch`: BSD-3-Clause with SMRT-AIST attribution, not a blanket
  Apache-2.0 relabeling.
- Upstream license texts and notices retain their respective copyright/terms.

All included third-party license texts are verbatim downloads at fixed commits;
URLs and SHA-256 digests are recorded in `SOURCE_MAP.json`. The source revision
used by each derivative and every file touched by a patch is listed there.
The applicable full text controls over these short descriptions.

## Excluded material and remaining review

See [EXCLUSIONS.md](EXCLUSIONS.md). The restrictive MonoGS main-project license
must not be confused with the separate rasterizer license. No broad permission
for the excluded MonoGS implementation is asserted here. GS-ICP's top-level
MIT license was not treated as overriding Inria/Graphdeco file headers.

No attribution or complete license can by itself cure an unavailable grant.
Before public release, the project owner should confirm rights in original
contributions and institutional/coauthor permissions. No ownership assignment,
legal opinion, remote publication or permission change was performed.
