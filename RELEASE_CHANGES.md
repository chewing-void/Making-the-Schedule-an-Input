# Changes in this separate partial-release copy

- Excluded six files listed in `EXCLUSIONS.md`; original working directory untouched.
- Retained all 12 detex files, seven GS-ICP Python files, the single MonoGS fabric,
  two TorchBeast adapters, and all three dependency patches.
- Replaced two code quotations in the MonoGS fabric docstring with interface prose.
- Added source, copyright and modification headers to six GS-ICP-derived files.
- Restored the complete upstream Facebook header and qualified the additive-only
  description in the TorchBeast driver. No executable statements were changed.
- Preserved the three original patch files byte for byte.
- Added five exact pinned license copies, complete rasterizer attribution,
  per-file source mappings, and explicit scope for the root Apache license.
- Rewrote README/REPRODUCE and patch documentation to remove unsupported complete
  reproduction claims. GS-ICP and MonoGS are explicitly incomplete; TorchBeast
  runtime remains untested.
- Added dependency-free verification, a hashed inventory, static check record
  and `.gitignore`. No git metadata, credentials, datasets or generated results
  were copied. Heuristic scanning cannot guarantee absence of all secrets.
- No remote repository was created, no upload/publication occurred, and no
  sharing or repository permissions were changed.

Remaining steps before any public distribution: confirm original-contribution
rights and collaborator/institution permissions; review the restrictive
rasterizer terms for the intended distribution/use; decide whether the omitted
code can later be authorized or independently replaced; run actual compatible
builds and end-to-end tests before making scientific reproducibility claims.
