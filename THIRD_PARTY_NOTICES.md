# Third-party notices and source provenance

## GS-ICP SLAM

Copyright (c) 2024 Lab. of AI and Robotics (LAIR).
[Exact fixed-commit MIT license](LICENSES/GS-ICP-SLAM-MIT.txt).
Six retained files copy or adapt upstream SLAM, tracker or mapper code. Their
headers identify the upstream source and project modifications. The adapter
fabric is separately identified as original project integration code.
Inria-headed renderer/utilities are deliberately excluded, not relabeled MIT.

## TorchBeast

Copyright (c) Facebook, Inc. and its affiliates.
[Exact fixed-commit Apache-2.0 license](LICENSES/TorchBeast-Apache-2.0.txt).
`torchbeast-det/monobeast_exchange.py` contains adapted control logic. Its
original copyright/license header is restored and modifications are marked.
The exact pinned upstream tree had no `NOTICE` file. No Facebook endorsement
is asserted.

## Rasterizer patches

[Full scoped attribution](patches/NOTICES.md) applies to both patches, with the
complete [GS license](LICENSES/GS-rasterizer-LICENSE.md) and
[MonoGS rasterizer license](LICENSES/MonoGS-rasterizer-LICENSE.md).
These license bytes are identical at the two pinned revisions. They impose
noncommercial research/evaluation limits and distribution conditions. They are
not an OSI-style unrestricted open-source grant and are not replaced by Apache.
The original texts, including their internal cross-references, are preserved.

## FastGICP patch

Copyright (c) 2020, SMRT-AIST.
[Complete BSD-3-Clause license](LICENSES/FastGICP-BSD-3-Clause.txt), including
its notice, disclaimer and non-endorsement condition. Neither touched source
file has an additional in-file copyright header.

## Exact mapping

[SOURCE_MAP.json](SOURCE_MAP.json) lists each retained Python/patch file, its
source revision or original-project status, upstream path, applicable license,
modification summary, and authoritative full-license SHA-256 digest. License
links use exact immutable commit IDs rather than a moving main branch.

Only files listed as retained in the manifest are distributed. Full third-party
source trees, additional upstream dependencies and their subdirectories are not
included. Installing them separately may introduce additional obligations.
