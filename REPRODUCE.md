# Verification and reproduction limits

## Status

This partial release replaces the former complete three-system reproduction
claim. The release preparation performed static checks only. It did not recreate
historical hardware/software environments, rerun a paper experiment, validate
accuracy/timing figures, build CUDA/GICP code, or certify L1/L2/L3 closure.
Earlier source comments may describe development observations; those are not
newly verified claims about this package.

- **detex:** source inspection and Python syntax checks only. Process/GPU protocol
  tests and full behavioral tests have not run.
- **GS-ICP:** incomplete and not runnable as published. See [EXCLUSIONS.md](EXCLUSIONS.md).
- **MonoGS:** incomplete and not runnable as published. Only a fabric example
  and a separately licensed dependency patch remain.
- **TorchBeast/monobeast:** adapter files are present, but no compatible environment,
  Atari ROM import, rollout, learning run or deterministic repeat was tested.

Do not infer results from equal syntax, license or file hashes. These checks
validate packaging and basic syntax, not algorithm correctness or scientific
reproduction. No historical result digests or generated experiment reports
are included as a substitute for runtime evidence.

## Static checks available in this package

From this directory, with Python 3.10 or newer:

```bash
python tools/verify_release.py
```

This uses only the standard library. It parses all Python files without importing
host packages; verifies the release manifest and exact third-party license bytes;
checks exclusions, local Markdown links, file inventory and obvious secret
patterns. It writes no files. The heuristic secret check is not exhaustive.
The included check record states the actual interpreter/version used.

Three `git apply --check` runs succeeded against only the exact 16 upstream
files touched by the patches, downloaded at the pinned revisions below. This
means patch context matched, not that builds or numerical behavior succeeded.
No patch was applied to the original working directory.

## Fixed upstream versions

- GS-ICP: `Lab-of-AI-and-Robotics/GS_ICP_SLAM`,
  `5f996a872a979406b270fe0ee3b0a8f25c5e9ae3`.
- MonoGS (interface/reference only): `muskie82/MonoGS`,
  `6c9254c319d8bff5caeef65259e6bb0941a9b9f6`.
- TorchBeast: `facebookresearch/torchbeast`,
  `0af07b051a2176a8f9fd20c36891ba2bba6bae68`.
- GS rasterizer: `Lab-of-AI-and-Robotics/diff-gaussian-rasterization`,
  `95dbb69d81449ae56783628803198bea2f42e2c9`.
- MonoGS rasterizer: `rmurai0610/diff-gaussian-rasterization-w-pose`,
  `43e21bff91cd24986ee3dd52fe0bb06952e50ec7`.
- FastGICP: `Lab-of-AI-and-Robotics/fast_gicp`,
  `e2954b1aa06a563d1b2ef2ec5d1fddd1e911d8f4`.

Full URLs, file mappings and license hashes are in [SOURCE_MAP.json](SOURCE_MAP.json).
Obtain upstream projects separately under their own licenses. Reviewing a source
link or downloading it does not imply permission to redistribute excluded files.

## Dependency patch inspection

After obtaining the exact applicable upstream revision in a separate directory,
review its license first. Dry-run commands are documented in
[patches/README.md](patches/README.md). These are deliberately check-only commands;
this package does not provide an installation/build or full SLAM reproduction
recipe while required source and rights remain unresolved.

## Experimental TorchBeast integration

The adapter imports `torchbeast.monobeast` and requires PyTorch, NumPy, Gym,
Atari bindings, and a legally obtained ROM. Historical documentation named
Python 3.8.20, PyTorch 2.4.1+cpu, Gym 0.19.0 and atari-py 0.2.5; that combination
and its availability were **not validated here**. No dependency lockfile or
fresh-install guarantee is provided. ROMs and their license are separate.

If a suitably licensed, compatible upstream installation is already available,
`monobeast_exchange.py --help` can describe the driver arguments. Its imports
may fail before help on an incomplete installation. This is an experimental
starting point, not a validated run instruction or certified result.

For future numerical validation, use repeated independent runs in one declared
hardware/software environment and compare complete reported status, L1 event
digests, L2 chain tails, and both relevant L3 end states. Ensure a failed or
incomplete run cannot be counted as closure. No such test has run for this copy.
