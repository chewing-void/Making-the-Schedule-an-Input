# Files omitted from the partial release

The source working directory was left unchanged. These files were not copied
into this release:

- `MonoGS-det/slam_frontend_exchange.py`: Transcribed MonoGS frontend; restrictive upstream license; not distributed.
- `MonoGS-det/slam_backend_exchange.py`: Transcribed MonoGS backend; restrictive upstream license; not distributed.
- `MonoGS-det/slam_exchange.py`: Upstream-derived boundary unresolved; depends on excluded frontend/backend.
- `MonoGS-det/monogs_det_setup.py`: Provenance of locally added GS-ICP seed helper remains unconfirmed; excluded conservatively.
- `GS-ICP-SLAM-det/gaussian_renderer/__init__.py`: Inria noncommercial header conflicts with blanket MIT treatment; exact inherited license chain not resolved.
- `GS-ICP-SLAM-det/utils/general_utils.py`: Inria header and Plenoxels/JaxNeRF attribution; exact inherited license chain not resolved.

## MonoGS review boundary

At commit `6c9254c319d8bff5caeef65259e6bb0941a9b9f6`, MonoGS has a restrictive
main-project license: https://github.com/muskie82/MonoGS/blob/6c9254c319d8bff5caeef65259e6bb0941a9b9f6/LICENSE.md
The frontend and backend adapter files describe and contain transcribed logic.
They are excluded, rather than relying on attribution as permission.

The driver was excluded because its upstream-derived boundary remains unresolved
and its required hosts are absent. The setup helper was excluded conservatively:
its seed function comes from a local GS-ICP overlay addition, **not** the pinned
upstream file. Its authorship has not been confirmed. This is not a finding that
that helper necessarily violates an upstream license.

The retained fabric was reviewed against pinned MonoGS `slam.py`, frontend,
backend, Gaussian model and general utilities. No normalized four-line matching
window was found in that comparison. This limited heuristic is not proof of
independent authorship. Two quoted implementation expressions in its docstring
were replaced with a natural-language interface layout and fixed source link.

## GS-ICP review boundary

The two excluded GS-ICP files carry Inria/Graphdeco noncommercial-research
headers despite the top-level MIT license. `general_utils.py` also references
Plenoxels/JaxNeRF ancestry. The pinned GS-ICP root lacks the referenced
`LICENSE.md`. A full, exact inherited permission chain was not established,
so these two files were omitted rather than labeled MIT or covered by an
assumed rasterizer license.

Removing them breaks the retained GS-ICP integration. The stock upstream
renderer does not contain the unpublished integration changes, and stock
`general_utils.py` lacks the added `seed_from_env` function. No complete
GS-ICP or MonoGS experiment can be reconstructed from this package alone.
