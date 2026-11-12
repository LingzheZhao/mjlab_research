# Research fork

This repository follows two sources:

| Remote | Role | Baseline for the MuJoCo 4 upgrade |
| --- | --- | --- |
| `upstream` | [mujocolab/mjlab](https://github.com/mujocolab/mjlab): simulation, dependencies and fixes | `51d54fcc029dbf537a4f090413be0ba07d78e9a9` |
| `research` | [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research): SuperMimic tracking extensions | `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` |

The histories share `936f7a12537556416addff31babbb1512dd24d70`. Keep both
histories through merge commits so later synchronizations use the correct merge
base. `main` holds reviewed integrations; consumer projects should pin a full
commit SHA. MuJoCo and MuJoCo Warp are 4.0.1; the Python package is still named
`mjlab` and keeps upstream's package version.

## Install this fork

```bash
git clone https://github.com/LingzheZhao/mjlab_research.git
cd mjlab_research
git checkout <reviewed-full-commit-sha>
uv sync --extra cu130 --group dev --frozen
```

Use `--extra cpu` for CPU checks, or upstream's `cu126` extra where appropriate.
When upgrading an existing environment from `mujoco` to `mujoco-core`, add
`--reinstall-package mujoco-core` to `uv sync`: the two distributions share files.

## Synchronize

Configure these remotes once:

```bash
git remote add upstream https://github.com/mujocolab/mjlab.git
git remote add research https://github.com/HybridRobotics/mjlab_research.git
```

For each update, start from the current fork and review both source histories:

```bash
git fetch origin
git fetch upstream
git fetch research
git switch -c sync/upstreams origin/main
git merge --no-ff upstream/main
git merge --no-ff research/main
```

Resolve real conflicts, validate, then push the branch and open a pull request.
Do not squash the synchronization PR: both upstream parents must remain ancestors
of `main`. A conflict or failed check stops the update for review.

The fork's `upstream/mjlab` and `upstream/research` branches record the imported
source commits. They contain no local changes. They can be updated with ordinary
fast-forward pushes; use the source remotes above to fetch the latest commits.

## Preserve the research interfaces

The upgrade keeps root/joint motion files, packed multi-clip loading and
`motion_ends`, differentiable `ForwardKinematics`, the `body_state()` reference
cache, external reset sampling, clip-end timeouts, and both ghost drawing APIs.
Sampling rules and motion formats stay as defined by the research source. The
tracking ONNX exporter obtains body references through `body_state()`.

After syncing the locked CPU environment, run:

```bash
uv sync --extra cpu --group dev --frozen
UV_NO_SYNC=1 MUJOCO_GL=disable make check
UV_NO_SYNC=1 FORCE_CPU=1 MUJOCO_GL=disable make test
```

The regression checks include MuJoCo/FK pose and twist agreement, finite-difference
input gradients, root-state resets with reordered tracked bodies, partial resets,
external sampling, per-clip timeouts, and ONNX reference outputs. CPU checks do
not establish CUDA graph/compile behavior or training equivalence. Before a
consumer adopts a new pin, run its generalized- and maximal-coordinate paths on
the target GPU, including a short rollout and training update with its motions.

The initial integration passed `make check` and the full CPU suite (1,205 passed,
13 skipped) on Python 3.13.5, MuJoCo/MuJoCo Warp 4.0.1, Torch 2.14.0+cpu and Warp
1.18.0. The skips follow the tests' runtime conditions. Native ghost geometry and
the official SuperMimic imports/configurations below were checked separately.

## Adopt from SuperMimic or BeyondMimic

Keep the simulation dependency here and algorithm changes in the consumer.
For a reproducible uv consumer, replace any existing `mjlab` source entry with:

```toml
[tool.uv.sources]
mjlab = { git = "https://github.com/LingzheZhao/mjlab_research.git", rev = "<reviewed-full-commit-sha>" }
```

Regenerate and commit that consumer's `uv.lock`, then sync its chosen CPU/CUDA
extra. A sibling editable checkout remains useful for development, but is not a
reproducible dependency pin.

Official SuperMimic at `abc124c1bd5976341034668af3d12c087f2f1f55` imports and
constructs generalized/maximal-coordinate configurations in the upgraded fork's
environment. Its existing lock still uses MuJoCo 3.10, Torch 2.10 and an editable
`../mjlab_research`; it must be refreshed separately. These import/configuration
checks do not claim complete training parity or a completed BeyondMimic migration.

The research extensions are the work of HybridRobotics. Original license and
copyright notices remain in place.
