# Separately licensed dependency patches

These diffs contain upstream-derived source. They are not licensed wholesale
under the root Apache-2.0 file. See [NOTICES.md](NOTICES.md), the complete
[third-party notices](../THIRD_PARTY_NOTICES.md), and the exact file mapping in
[SOURCE_MAP.json](../SOURCE_MAP.json).

- `gsicp-diff-gaussian-rasterization-deterministic-backward.patch`: GS rasterizer
  at `95dbb69d81449ae56783628803198bea2f42e2c9`; noncommercial research/evaluation
  Gaussian-Splatting license in [LICENSES](../LICENSES/GS-rasterizer-LICENSE.md).
- `monogs-diff-gaussian-rasterization-deterministic-backward.patch`: pose rasterizer
  at `43e21bff91cd24986ee3dd52fe0bb06952e50ec7`; same restrictive terms in
  [LICENSES](../LICENSES/MonoGS-rasterizer-LICENSE.md). Its separate license does
  not grant redistribution permission for the MonoGS frontend/backend.
- `gsicp-fast_gicp-reliable-neighbours.patch`: FastGICP at
  `e2954b1aa06a563d1b2ef2ec5d1fddd1e911d8f4`; [BSD-3-Clause](../LICENSES/FastGICP-BSD-3-Clause.txt).

The rasterizer changes replace floating-point atomic accumulation with a
fixed-order backward reduction. FastGICP changes address reliable-neighbour
counts and degenerate covariances. These intended changes have not been compiled
or behaviorally verified in this release preparation.

For a check-only dry run, set `DET_REPO` to this package and change into the
corresponding clean, exact upstream source directory first:

```bash
# In GS rasterizer source:
git apply --check -p1 "$DET_REPO/patches/gsicp-diff-gaussian-rasterization-deterministic-backward.patch"
# In FastGICP source:
git apply --check -p1 "$DET_REPO/patches/gsicp-fast_gicp-reliable-neighbours.patch"
# In MonoGS pose rasterizer source (extra directory prefix in this diff):
git apply --check -p2 "$DET_REPO/patches/monogs-diff-gaussian-rasterization-deterministic-backward.patch"
```

All three checks passed on the downloaded pinned source subsets. No build,
numerical test, benchmark, or complete host run was performed. Preserve existing
headers when applying patches, and accompany redistributed derived works with
the full relevant license and unchanged notices. Commercial exploitation or
commercial distribution of the rasterizer requires prior explicit licensor
consent under the included terms.
