"""Modal cloud GPU entry point for Stage 4 (AGZ) training on TSP.

Mirrors `modal_run_train.py` (Stage 1) but runs `train_alphazero.py` and
extends the image with a C++ toolchain so the MCTS extension builds
inside the container.

Single-job usage (smoke / one-off):
  modal run src/scripts/modal_run_train_alphazero.py::run_one -- \
      --n_iterations 1 --M_instances 10 --n_simulations_train 20 \
      --train_steps_per_iter 5 --val_size 100 --gate_every 99 \
      --leaf_eval rollout --dirichlet_epsilon 0.05 \
      --run_name smoke

Parallel-batch usage (the four F.4 + F.3 jobs):
  modal run src/scripts/modal_run_train_alphazero.py::run_all

Download results:
  modal volume get am-alphagozero-volume outputs/

W&B logging convention (aligned with Stage 1 for side-by-side comparison):
  - Stage 4 runs land in the same W&B group `tsp_{graph_size}` as Stage 1
    runs (set in `am_baseline.training.coach.MCTSCoach.__init__`).
  - One Stage 4 iteration ≡ one Stage 1 epoch on the W&B `epoch` x-axis:
    `MetricsLogger.log_iteration` emits `epoch`, `val_avg_cost`, `lr`,
    `epoch_duration` (= mcts_wall_s + train_wall_s), and `baseline_updated`
    (= 1 if gate accepted, else 0) alongside the Stage-4-specific
    `iteration` / `val_avg_cost_iter` / `policy_loss_mean` series.
  - Stage 4 per-train-step logs emit `global_step` (cumulative across
    iterations) and `value_loss` aliases on Stage 1's `global_step` x-axis,
    alongside `iteration` / `policy_loss_step` / `value_loss_step`.
  Modify these via `src/am_baseline/training/logging.py` (the actual W&B
  payload construction) — this Modal entrypoint just forwards CLI args.
"""
import os
import sys
from pathlib import Path

import modal

APP_NAME = "am-alphagozero"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/vol"
DEFAULT_GPU = "A10"
volume = modal.Volume.from_name("am-alphagozero-volume", create_if_missing=True)

# Same exclusion set as the Stage 1 wrapper.
IGNORE_PATTERNS = [
    "**/outputs/**",
    "**/ref/**",
    "**/__pycache__/**",
    "**/.git/**",
    "**/*.egg-info/**",
    "**/.claude/**",
]


def load_gitignore_patterns() -> list[str]:
    if not modal.is_local():
        return []
    patterns = list(IGNORE_PATTERNS)
    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return patterns
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# Build container image (refactored 2026-05-07: pip → uv).
#   1. `uv_sync()` installs every locked dependency from uv.lock (torch + cu124,
#      wandb, scipy, etc.) into the image's Python env. The lock file lives at
#      the repo root and pins the cu124 torch wheels via the
#      [tool.uv.sources] linux marker in pyproject.toml.
#   2. `add_local_dir` copies the project sources to PROJECT_DIR — required at
#      image-build time because step 3 needs the source files present.
#   3. `pip install -e . --no-deps` installs OUR package (editable) at
#      PROJECT_DIR/src/am_baseline and triggers the pybind11 build of the C++
#      MCTS extension. `--no-deps` because uv_sync already provided everything.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "g++", "make")
    .uv_sync()
)

if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )

image = image.add_local_dir(
    ".",
    remote_path=PROJECT_DIR,
    ignore=load_gitignore_patterns(),
    copy=True,
)

# Build the C++ MCTS extension once at image-build time on top of uv-installed
# deps. setup.py uses pybind11 (already in build-system.requires) and emits
# `_mcts_cpp.cp311-linux_x86_64.so` next to the source files at
# PROJECT_DIR/src/am_baseline/search/mcts_cpp/.
image = image.run_commands(
    [
        f"cd {PROJECT_DIR} && pip install -e . --no-deps",
    ]
)

app = modal.App(APP_NAME)

run_env = {"PYTHONPATH": f"{PROJECT_DIR}/src"}
_wandb_key = os.environ.get("WANDB_API_KEY", "")
if _wandb_key:
    run_env["WANDB_API_KEY"] = _wandb_key


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 24,
    env=run_env,
    image=image,
    gpu=DEFAULT_GPU,
)
def train_alphazero_remote(*train_args: str) -> None:
    """Run train_alphazero.py with the given CLI args on a Modal A10."""
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

    # Symlink outputs/ -> volume mount so checkpoints persist across runs
    # AND so --load_path can read Stage 1 from the volume.
    outputs_vol = Path(VOLUME_PATH) / "outputs"
    outputs_vol.mkdir(parents=True, exist_ok=True)

    outputs_link = Path(PROJECT_DIR) / "outputs"
    if outputs_link.is_dir() and not outputs_link.is_symlink():
        import shutil
        shutil.rmtree(outputs_link)
    elif outputs_link.exists() or outputs_link.is_symlink():
        outputs_link.unlink()
    outputs_link.symlink_to(outputs_vol)

    # Sanity check: confirm the C++ extension is importable on this image.
    try:
        from am_baseline.search.mcts_cpp import HAVE_CPP_MCTS  # noqa: F401
        print(f"[modal] HAVE_CPP_MCTS = {HAVE_CPP_MCTS}", flush=True)
    except Exception as e:
        print(f"[modal] WARNING: could not import mcts_cpp ({e}); will fall back to Python MCTS",
              flush=True)

    # Hand the args off to the launcher (it has parse_opts/_finalize_opts/run
    # rather than a single main(), so reproduce the __main__ block here).
    sys.argv = ["train_alphazero.py"] + list(train_args)
    from scripts.train_alphazero import parse_opts, _finalize_opts, run
    opts = _finalize_opts(parse_opts())
    run(opts)
    volume.commit()


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def run_one(*train_args: str) -> None:
    """Run a single training job. Forwards CLI args to train_alphazero.py."""
    print(f"[modal] launching one job: {' '.join(train_args)}")
    train_alphazero_remote.remote(*train_args)
    print("[modal] done")


# ---------------------------------------------------------------------------
# F.4 + F.3 four-job batch (the recipes locked in 2026-04-30 with the user)
# ---------------------------------------------------------------------------

STAGE1_CKPT = (
    "outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt"
)


def _common_args() -> list[str]:
    """Args shared by all four jobs.

    W&B logging is ON by default. The Modal image already includes wandb in
    its pip deps and forwards either ~/.netrc (mounted into the image at
    build time) or the local WANDB_API_KEY env var into the container's
    run env. Each job's `--run_name` differs, so W&B runs are distinct.
    """
    return [
        "--load_path", STAGE1_CKPT,
        "--graph_size", "20",
        "--leaf_eval", "rollout",
        "--dirichlet_epsilon", "0.05",
        "--temperature_schedule", "step30",
        "--batch_size", "512",
        "--val_size", "10000",
        # Enable W&B (matches Stage 1's modal_run_train.py convention).
        # Override per-call by passing --no_wandb if you want CSV-only.
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]


def _f4_args(run_name: str, *, k: int, gate_mode: str, gate_every: int = 5) -> list[str]:
    """F.4 main-run scale: 100 iter × M=1000 × train_steps=200 × buffer=200K."""
    return _common_args() + [
        "--n_iterations", "100",
        "--M_instances", "1000",
        "--n_simulations_train", str(k),
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--gate_every", str(gate_every),
        "--gate_mode", gate_mode,
        "--run_name", run_name,
    ]


def _f3_args(run_name: str, *, extra: list[str]) -> list[str]:
    """F.3-scale pilot: 20 iter × M=1000 × K=200 × train_steps=100 × buffer=50K.

    Used for the (b) hypothesis tests (freeze_encoder, lr=1e-5).
    """
    return _common_args() + [
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "200",
        "--train_steps_per_iter", "100",
        "--buffer_capacity", "50000",
        "--gate_every", "5",
        "--gate_mode", "ttest",
        "--run_name", run_name,
        *extra,
    ]


@app.local_entrypoint()
def run_f60_grid(timestamp: str = "") -> None:
    """Phase F.6.0 pre-flight grid probe — does the warm-start best recipe transfer to from-scratch?

    Spawns 12 jobs in parallel on Modal A10 to probe three knobs that may not
    transfer cleanly from b2's warm-start best (lr=1e-5, rollout, ε=0.05) to
    from-scratch random init:

        leaf_eval ∈ {value_head, rollout}          (AGZ-canonical vs AM-paper-rollout)
        ε         ∈ {0.0, 0.05, 0.25}              (off, warm-start-safe, AGZ-canonical)
        gate_mode ∈ {ttest, always}                (AGZ-style vs AZ-style)

    **lr is FIXED at 1e-4** (AM-paper-original / Stage 1 default) for apples-to-apples
    sample efficiency vs Stage 1 — see plan F.6 setup note. lr=1e-5 sweep is
    a separate G.9 ablation, not part of F.6's main-line claim.

    Each variant: from-scratch (no --load_path), 50 iter × M=1000 × K=100,
    train_steps=100, buffer=50K, gate_every=5, val_seed=42.
    Pin val_seed=42 so per-iter val_avg_cost is comparable across variants.

    Wall-clock estimate: rollout jobs ~3.3 h, value_head jobs ~25 min;
    in parallel ≈ ~3.3 h total. Cost ≈ ~$10-15 in Modal credits.

    Decision rule for picking F.6.1 recipe: lowest val_avg_cost at iter 50
    (or steepest decline if multiple variants tie within ±0.001 noise band).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    base_args = [
        "--graph_size", "20",
        "--n_iterations", "50",
        "--M_instances", "1000",
        "--n_simulations_train", "100",
        # train_steps_per_iter and buffer_capacity MATCH F.6.1 so the
        # F.6.0 winner's iter-49.pt + buffer.pt cleanly resume into F.6.1.
        # (buffer.load() hard-fails on capacity mismatch — see coach.py:366-370.)
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--lr_model", "1e-4",  # FIXED at AM-paper / Stage 1 default for fair comparison
        "--val_size", "10000",
        "--val_seed", "42",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
        # NOTE: no --load_path — from-scratch random init (proposal Phase F.6).
    ]

    grid = []
    for leaf_label, leaf_val in [("vh", "value_head"), ("rol", "rollout")]:
        for eps_label, eps_val in [("0", "0.0"), ("05", "0.05"), ("25", "0.25")]:
            for gate_label, gate_val in [("ttest", "ttest"), ("always", "always")]:
                run_name = f"f60_le{leaf_label}_eps{eps_label}_g{gate_label}_{timestamp}"
                args = base_args + [
                    "--leaf_eval", leaf_val,
                    "--dirichlet_epsilon", eps_val,
                    "--gate_mode", gate_val,
                    "--run_name", run_name,
                ]
                grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel jobs (timestamp={timestamp})")
    for label, _ in grid:
        print(f"  {label}")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} F.6.0 grid jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f605_lr_validation(timestamp: str = "") -> None:
    """Phase F.6.0.5 — 4-variant 2x2 lr × wd grid (value_head leaf eval, K=40, 20 iter).

    Tests the lr × wd interaction at value_head leaf eval, which is ~5x cheaper
    per iter than rollout (K=100 rollout was ~440 s/iter; K=40 value_head ≈
    ~80 s/iter). 20 iter at K=40 makes this a fast-iteration probe.

    Why value_head here (vs F.6.0's rollout): F.6.0 showed value_head trailing
    rollout by ~0.2, but the user wants to retest with the F.6.0.5-derived
    lr/wd choices to see if better optimizer settings rescue value_head's
    leaf-eval signal. Cheaper per-iter compute also makes the 4-variant grid
    feasible.

    Four variants — full 2x2 lr × wd grid:

        V1 (control)        lr=1e-4  wd=1e-4   AGZ-canonical defaults
        V2 (analytical)     lr=5e-4  wd=1e-4   higher lr at AGZ wd
        V3 (lr+wd-zero)     lr=5e-4  wd=0      higher lr + Stage-1 wd convention
        V4 (wd-zero only)   lr=1e-4  wd=0      Stage-1 wd convention at AGZ lr

    Hold-fixed: leaf_eval=value_head, dirichlet_epsilon=0.25, gate_mode=ttest,
    n_simulations=40, M=1000, train_steps_per_iter=200, buffer=200K,
    batch_size=512, max_grad_norm=1.0, val_seed=42, no --load_path,
    n_iterations=20.

    Decision rule (two-axis):
      LR axis: compare {V1, V4} (lr=1e-4) vs {V2, V3} (lr=5e-4).
      WD axis: compare {V1, V2} (wd=1e-4) vs {V3, V4} (wd=0).
      If higher lr helps on average AND wd doesn't matter → F.6.1 = V2 settings.
      If higher lr helps AND wd=0 helps → F.6.1 = V3 settings.

    Cost: ~$5-8 Modal credits, ~30 min wall-clock parallel (value_head-K=40;
    much cheaper than F.6.0's rollout-K=100 setup).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    base_args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "40",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--dirichlet_epsilon", "0.25",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
        # NOTE: no --load_path — from-scratch random init.
    ]

    # (label, lr, weight_decay)
    variants = [
        ("f605vh_lr1e4_wd1e4", "1e-4", "1e-4"),  # V1 control (AGZ defaults)
        ("f605vh_lr5e4_wd1e4", "5e-4", "1e-4"),  # V2 analytical (higher lr, AGZ wd)
        ("f605vh_lr5e4_wd0",   "5e-4", "0.0"),   # V3 lr+wd-zero (Stage-1 wd)
        ("f605vh_lr1e4_wd0",   "1e-4", "0.0"),   # V4 wd-zero only (AGZ lr, Stage-1 wd)
    ]

    grid = []
    for label_stem, lr_val, wd_val in variants:
        run_name = f"{label_stem}_{timestamp}"
        args = base_args + [
            "--lr_model", lr_val,
            "--weight_decay", wd_val,
            "--run_name", run_name,
        ]
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel F.6.0.5 jobs (timestamp={timestamp})")
    for label, _ in grid:
        print(f"  {label}")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} F.6.0.5 lr-validation jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f605_lr_validation_raw_target(timestamp: str = "") -> None:
    """Phase F.6.0.5b — 4-variant 2x2 lr × wd grid with RAW cost-to-go value target.

    Identical to `run_f605_lr_validation` (value_head leaf eval, K=40, 20 iter)
    EXCEPT it sets `--value_target_norm none` — the value head is trained on
    raw cost_to_go instead of cost_to_go / bl_val. MCTS divides by current
    bl_val at leaf-eval time. Eliminates three distribution-shift problems:
      (1) Calibration drift between training-time bl_val and MCTS-time bl_val.
      (2) Buffer-time non-stationarity as bl_val evolves with gating.
      (3) Across-instance variance collapse at random init (where cost_to_go
          and bl_val are correlated, making z ≈ 1 near-constant).

    Hypothesis: if (1)-(3) were the dominant reason value_head trailed rollout
    in F.6.0 + F.6.0.5, this raw-target re-run should narrow the gap or even
    flip the leaf-eval ordering.

    Four variants (mirrors `run_f605_lr_validation` exactly except the target):

        V1 (control)        lr=1e-4  wd=1e-4   AGZ-canonical defaults
        V2 (analytical)     lr=5e-4  wd=1e-4   higher lr at AGZ wd
        V3 (lr+wd-zero)     lr=5e-4  wd=0      higher lr + Stage-1 wd convention
        V4 (wd-zero only)   lr=1e-4  wd=0      Stage-1 wd convention at AGZ lr

    Hold-fixed (everything else matches the value_head sibling):
    leaf_eval=value_head, dirichlet_epsilon=0.25, gate_mode=ttest,
    n_simulations=40, M=1000, train_steps_per_iter=200, buffer=200K,
    batch_size=512, max_grad_norm=1.0, val_seed=42, n_iterations=20,
    no --load_path, **--value_target_norm none**.

    Cost: ~$5-8 Modal credits, ~30 min wall-clock parallel
    (value_head-K=40, same per-iter cost as F.6.0.5 sibling).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    base_args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "40",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--dirichlet_epsilon", "0.25",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",       # ← key difference vs the sibling
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
        # NOTE: no --load_path — from-scratch random init.
    ]

    # (label, lr, weight_decay)
    variants = [
        ("f605vhraw_lr1e4_wd1e4", "1e-4", "1e-4"),  # V1 control (AGZ defaults)
        ("f605vhraw_lr5e4_wd1e4", "5e-4", "1e-4"),  # V2 analytical
        ("f605vhraw_lr5e4_wd0",   "5e-4", "0.0"),   # V3 lr+wd-zero
        ("f605vhraw_lr1e4_wd0",   "1e-4", "0.0"),   # V4 wd-zero only
    ]

    grid = []
    for label_stem, lr_val, wd_val in variants:
        run_name = f"{label_stem}_{timestamp}"
        args = base_args + [
            "--lr_model", lr_val,
            "--weight_decay", wd_val,
            "--run_name", run_name,
        ]
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel F.6.0.5b (raw-target value_head) jobs (timestamp={timestamp})")
    for label, _ in grid:
        print(f"  {label}")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} F.6.0.5b (raw-target) jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f605_lr_validation_rollout(timestamp: str = "") -> None:
    """Phase F.6.0.5 — 4-variant 2x2 lr × wd grid, rollout leaf eval, K=40, 20 iter.

    Sibling of `run_f605_lr_validation` (which uses value_head leaf eval).
    Same 4-variant lr × wd grid, same K=40, 20 iter, M=1000, train_steps=200,
    buffer=200K, ε=0.25, ttest gate, val_seed=42, from-scratch — but
    `--leaf_eval rollout` instead of value_head.

    Why both: F.6.0 showed rollout > value_head by ~0.2 at K=100. With the
    new lr/wd choices, this entrypoint tests whether rollout still
    outperforms value_head, AND maps the lr × wd response surface in the
    rollout regime. Together with the value_head sibling, we get an 8-cell
    leaf_eval × lr × wd grid for the same training cost as F.6.0.

    Four variants (mirroring the value_head sibling):

        V1 (control)        lr=1e-4  wd=1e-4   AGZ-canonical defaults
        V2 (analytical)     lr=5e-4  wd=1e-4   higher lr at AGZ wd
        V3 (lr+wd-zero)     lr=5e-4  wd=0      higher lr + Stage-1 wd convention
        V4 (wd-zero only)   lr=1e-4  wd=0      Stage-1 wd convention at AGZ lr

    Cost: ~$8-12 Modal credits, ~50-60 min wall-clock parallel
    (rollout-K=40 ≈ ~175 s/iter; 20 iter ≈ ~58 min per variant; 4 in parallel).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    base_args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "40",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "rollout",
        "--dirichlet_epsilon", "0.25",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
        # NOTE: no --load_path — from-scratch random init.
    ]

    # (label, lr, weight_decay)
    variants = [
        ("f605rol_lr1e4_wd1e4", "1e-4", "1e-4"),  # V1 control (AGZ defaults)
        ("f605rol_lr5e4_wd1e4", "5e-4", "1e-4"),  # V2 analytical (higher lr, AGZ wd)
        ("f605rol_lr5e4_wd0",   "5e-4", "0.0"),   # V3 lr+wd-zero (Stage-1 wd)
        ("f605rol_lr1e4_wd0",   "1e-4", "0.0"),   # V4 wd-zero only (AGZ lr, Stage-1 wd)
    ]

    grid = []
    for label_stem, lr_val, wd_val in variants:
        run_name = f"{label_stem}_{timestamp}"
        args = base_args + [
            "--lr_model", lr_val,
            "--weight_decay", wd_val,
            "--run_name", run_name,
        ]
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel F.6.0.5 (rollout) jobs (timestamp={timestamp})")
    for label, _ in grid:
        print(f"  {label}")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} F.6.0.5 (rollout) jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f606_eps_validation(timestamp: str = "") -> None:
    """Phase F.6.0.6 — 2-variant ε sweep at V3 settings (raw-target value_head).

    Holds F.6.0.5b winner's V3 config fixed (lr=5e-4, wd=0,
    --value_target_norm none, leaf_eval=value_head, K=40, 20 iter) and varies
    only --dirichlet_epsilon ∈ {0.0, 0.05}. F.6.0.5b's V3 (ε=0.25, val_avg_cost
    iter-19 = 4.265) acts as the implicit third reference point — not re-run.

    Question: under V3's regime, does any Dirichlet noise help or hurt the
    F.6.1 trajectory? F.6.0 picked ε=0.25 in a different regime (lr=1e-4,
    bl-normalized value_head where the value head was effectively broken).
    Under raw-target (value head actually contributing leaf-discrimination)
    + 5× higher lr (faster policy mode-locking), theory predicts lower ε
    is better. This 2-variant sweep tests the floor (ε=0) and the lower
    edge of the predicted sweet spot (ε=0.05).

    Cost: ~$3-5 Modal credits, ~30 min wall-clock parallel.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    base_args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "40",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--weight_decay", "0.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]

    variants = [
        ("f606_eps0",  "0.0"),
        ("f606_eps05", "0.05"),
    ]

    grid = []
    for label_stem, eps_val in variants:
        run_name = f"{label_stem}_{timestamp}"
        args = base_args + [
            "--dirichlet_epsilon", eps_val,
            "--run_name", run_name,
        ]
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel F.6.0.6 ε-sweep jobs (timestamp={timestamp})")
    for label, _ in grid:
        print(f"  {label}")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} F.6.0.6 ε-sweep jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f607_k20_probe(timestamp: str = "") -> None:
    """Phase F.6.0.7 — single-variant K=20 probe at F.6.0.6 winner settings.

    Holds F.6.0.6 E1 config fixed (ε=0, lr=5e-4, wd=0, value_target_norm=none,
    leaf_eval=value_head, 20 iter) and lowers K from 40 → 20. Compares against
    F.6.0.6 E1 (K=40, val_avg_cost(iter 19) = 4.228) as the implicit reference.

    Question: at TSP-20 (N=20 actions per root), is K=20 enough MCTS budget
    to produce a usable π_t target? K=N gives each action ~1 visit on average;
    PUCT with c_puct·P·√N_total/(1+N_a) will pull a few extra visits to the
    policy's argmax. The π_t target becomes a noisy single-visit estimate
    rather than a smooth distribution.

    Trade-off: K=20 should be ~2× faster per iter than K=40 in raw compute,
    but the noisier π_t target may slow policy convergence. The point of this
    probe is to see whether the wall-clock savings outweigh the per-iter
    learning rate.

    Cost: ~$2-3 Modal credits, ~15-20 min wall-clock single-job.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "20",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", "0.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]
    run_name = f"f607_k20_eps0_{timestamp}"
    args = args + ["--run_name", run_name]

    print(f"[modal] launching F.6.0.7 K=20 probe (timestamp={timestamp})")
    print(f"  {run_name}")

    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f608_k20_bs2048_probe(timestamp: str = "") -> None:
    """Phase F.6.0.8 — K=20 + batch_size=2048 single-variant probe.

    Holds F.6.0.7 config fixed (ε=0, K=20, lr=5e-4, wd=0,
    value_target_norm=none, leaf_eval=value_head, 20 iter, M=1000) and
    raises batch_size 512 → 2048 (4×). Compares against F.6.0.7
    (K=20, batch=512) to isolate batch_size as the sole variable.

    Question: at K=20 (already a thinner MCTS budget producing noisier π_t
    targets), does a 4× larger batch reduce gradient variance enough to
    materially help convergence? Each step now covers 4× more buffer →
    gradient variance ÷ 2 (sqrt rule). Adam adapts to batch size without
    needing lr re-tuning at this scale.

    Trade-off: train_wall_s rises ~3.5s/iter → ~14s/iter (4× train compute).
    Still <15% of mcts_s (~60s at K=20), so wall-clock impact is minor.

    Cost: ~$2-3 Modal credits, ~15-20 min wall-clock single-job.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "20",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "2048",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", "0.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]
    run_name = f"f608_k20_bs2048_{timestamp}"
    args = args + ["--run_name", run_name]

    print(f"[modal] launching F.6.0.8 K=20 + batch=2048 probe (timestamp={timestamp})")
    print(f"  {run_name}")

    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f609_k20_bs2048_rollout_probe(timestamp: str = "") -> None:
    """Phase F.6.0.9 — K=20 + batch=2048 + leaf_eval=rollout probe.

    Mirrors F.6.0.8 (ε=0, K=20, batch=2048, lr=5e-4, wd=0, value_target_norm=none,
    20 iter, M=1000) but switches leaf_eval value_head → rollout.

    Question: at thin K=20 budgets, does rollout leaf eval (fresh Monte Carlo
    estimate per leaf) outperform value_head leaf eval (cached NN prediction)?
    F.6.0 winner was rollout at K=100, but in a totally different regime
    (lr=1e-4, bl-normalized, ε=0.25). Under F.6.0.6 settings (lr=5e-4,
    raw-target, ε=0), the value head learned non-trivial signal — but it's
    still the model's prediction, possibly noisier than 1-sample rollouts at
    leaves.

    Value head still trains on raw cost_to_go (value_target_norm=none) as an
    auxiliary loss; just not used at MCTS time when leaf_eval=rollout.

    Cost: ~$2-4 Modal credits, ~25-30 min wall-clock single-job (rollout adds
    ~30-40% per-iter overhead vs value_head at same K).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "20",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "2048",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "rollout",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", "0.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]
    run_name = f"f609_k20_bs2048_rollout_{timestamp}"
    args = args + ["--run_name", run_name]

    print(f"[modal] launching F.6.0.9 K=20 + batch=2048 + rollout probe (timestamp={timestamp})")
    print(f"  {run_name}")

    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f610_k20_bs2048_M2000_probe(timestamp: str = "") -> None:
    """Phase F.6.1.0 — M=2000 × 10 iter probe (matched-budget vs F.6.0.8).

    Mirrors F.6.0.8 (K=20, batch=2048, ε=0, lr=5e-4, wd=0,
    value_target_norm=none, leaf_eval=value_head, train_steps=200) but:
      - doubles M_instances 1000 → 2000 (more fresh data per iter)
      - halves n_iterations 20 → 10  (keeps total instances = 20K, same as F.6.0.8)

    Question: does concentrating sample budget into more-fresh-per-iter
    (10 × 2000) beat spreading it thin (20 × 1000) at matched 20K total?
    Lifetime sample-per-tuple ratio drops 20.5× → 10.3× — better-balanced
    producer/consumer rates while still over KataGo's empirical 8× cap.

    Primary comparison: F.6.1.0 iter-9 (final) vs F.6.0.8 iter-19 (final),
    both at 20K total instances seen. If F.6.1.0 ≤ F.6.0.8 by ≥ 0.02, the
    M-rebalance hypothesis is validated.

    Cost: ~$2-3 Modal credits, ~20-25 min wall-clock single-job
    (M doubled but iter count halved → similar total).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    args = [
        "--graph_size", "20",
        "--n_iterations", "10",
        "--M_instances", "2000",
        "--n_simulations_train", "20",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "200000",
        "--batch_size", "2048",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", "0.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]
    run_name = f"f610_k20_bs2048_M2000_{timestamp}"
    args = args + ["--run_name", run_name]

    print(f"[modal] launching F.6.1.0 K=20 + batch=2048 + M=2000 × 10 iter probe (timestamp={timestamp})")
    print(f"  {run_name}")

    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f611_buffer_size_probe(timestamp: str = "") -> None:
    """Phase F.6.1.1 — 2-variant buffer-size probe (online vs short window).

    Mirrors F.6.0.7 (K=20, batch=512, M=1000, ε=0, lr=5e-4, wd=0,
    value_target_norm=none, leaf_eval=value_head, 20 iter) — varies only
    --buffer_capacity ∈ {1000, 5000}.

    Question: does cross-iter buffer averaging (denoising MCTS targets via
    multiple policy snapshots) help, or does stale data hurt?
    - buffer=1000 (= M): 1-iter window, fully online — every iter trains
      ONLY on its own self-play, no cross-iter mix.
    - buffer=5000 (= 5×M): 5-iter rolling window, mild cross-iter averaging.
    - F.6.0.7 (buffer=200K): 200-iter window, never evicts in 20-iter run
      (val_avg_cost(iter 19) = 4.428 reference).

    Lifetime sample-per-tuple ratio is roughly constant (~5×) across all 3
    settings — only the window (recency mix) varies. Clean isolation.

    Cost: ~$3-5 Modal credits, ~25 min wall-clock parallel.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    base_args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "20",
        "--train_steps_per_iter", "200",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", "0.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]

    variants = [
        ("f611_buf1k",  "1000"),
        ("f611_buf5k",  "5000"),
    ]

    grid = []
    for label_stem, buf_val in variants:
        run_name = f"{label_stem}_{timestamp}"
        args = base_args + [
            "--buffer_capacity", buf_val,
            "--run_name", run_name,
        ]
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel F.6.1.1 buffer-size jobs (timestamp={timestamp})")
    for label, _ in grid:
        print(f"  {label}")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} F.6.1.1 buffer-size jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f612_k40_buf5k_probe(timestamp: str = "") -> None:
    """Phase F.6.1.2 — combined K=40 + buffer=5000 probe.

    Combines the two strongest findings from the F.6.0.7→F.6.1.1 series:
      - K=40 (F.6.0.6 E1's K, beats K=20 by ~0.20 endpoint)
      - buffer_capacity=5000 (F.6.1.1 buf=5K beat buf=200K by ~0.13 at K=20)

    All other settings = F.6.0.6 E1 = F.6.1.1 except buffer + (carry K=40):
    leaf_eval=value_head, batch=512, M=1000, ε=0, lr=5e-4, wd=0,
    value_target_norm=none, gate=ttest, val_seed=42, train_steps=200, 20 iter.

    Expected: if buffer effect (K=20: −0.129) transfers additively to K=40,
    final val_avg_cost should be around 4.228 − 0.129 ≈ 4.10. If multiplicative
    (more efficient training × more visits per root), could go lower.

    Reference: F.6.0.6 E1 (buffer=200K) = 4.228 at iter 19.

    Cost: ~$2-3 Modal credits, ~30 min wall-clock single-job
    (K=40 doubles per-iter mcts_s vs F.6.1.1's K=20).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    args = [
        "--graph_size", "20",
        "--n_iterations", "20",
        "--M_instances", "1000",
        "--n_simulations_train", "40",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "5000",
        "--batch_size", "512",
        "--gate_every", "5",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--gate_mode", "ttest",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", "0.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]
    run_name = f"f612_k40_buf5k_{timestamp}"
    args = args + ["--run_name", run_name]

    print(f"[modal] launching F.6.1.2 K=40 + buffer=5000 probe (timestamp={timestamp})")
    print(f"  {run_name}")

    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


def _f61_args(
    *,
    run_name: str,
    k: int = 40,
    buffer_capacity: int = 5000,
    lr_model: str = "5e-4",
    lr_decay: str = "1.0",
    lr_decay_step_size: int = 1,
    temperature_schedule: str = "step30",
    dirichlet_epsilon: str = "0.0",
    leaf_eval: str = "value_head",
    lambda_v: str = "1.0",
    n_simulations_schedule: str = "const",
    n_simulations_first: int = 5,
    n_simulations_late: int = 10,
    n_simulations_last: int = 1,
) -> list[str]:
    """Shared F.6.1 main-run config builder. Defaults to F.6.0.6/F.6.0.7 winners.

    Locked by the F.6.0.5 → F.6.1.1 ablation series:
      lr=5e-4, wd=0, value_target_norm=none, ε=0, leaf_eval=value_head,
      gate=ttest, gate_every=1, val_seed=42, M=1000, train_steps=200,
      batch=512, n_iterations=100. K and buffer come from F.6.1.2 verdict.

    `temperature_schedule` and `dirichlet_epsilon` are CLI-overridable so
    F.6.2 step10 + ε-sweep variants can reuse this builder without
    duplicating the rest of the recipe.
    """
    args = [
        "--graph_size", "20",
        "--n_iterations", "100",
        "--M_instances", "1000",
        "--n_simulations_train", str(k),
        "--train_steps_per_iter", "200",
        "--buffer_capacity", str(buffer_capacity),
        "--batch_size", "512",
        "--gate_every", "1",
        "--gate_mode", "ttest",
        "--temperature_schedule", temperature_schedule,
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", leaf_eval,
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lambda_v", lambda_v,
        "--lr_model", lr_model,
        "--lr_decay", lr_decay,
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", dirichlet_epsilon,
        "--dirichlet_alpha_factor", "10.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
        "--run_name", run_name,
    ]
    if n_simulations_schedule != "const":
        args.extend([
            "--n_simulations_schedule", n_simulations_schedule,
            "--n_simulations_first", str(n_simulations_first),
            "--n_simulations_late", str(n_simulations_late),
            "--n_simulations_last", str(n_simulations_last),
        ])
    if lr_decay_step_size != 1:
        args.extend(["--lr_decay_step_size", str(lr_decay_step_size)])
    return args


@app.local_entrypoint()
def run_f61_main(
    timestamp: str = "",
    k: str = "40",
    buffer_capacity: str = "5000",
) -> None:
    """Phase F.6.1 main run — 100-iter trajectory probe at the F.6.0.5+F.6.0.6+F.6.1.1 winner recipe.

    Settings (all locked by ablation series; only K and buffer are CLI-overridable
    so the F.6.1.2 verdict can flow through without code edits):

      lr=5e-4, lr_decay=1.0 (no decay), weight_decay=0,
      value_target_norm=none, dirichlet_epsilon=0, leaf_eval=value_head,
      gate=ttest, gate_every=1, M=1000, train_steps=200, batch=512,
      val_seed=42, n_iterations=100, from-scratch (no --load_path).

    Defaults: K=40, buffer_capacity=5000 (override via --k / --buffer-capacity).

    Expected wall-clock: ~3.5-4.5 h Modal A10 single-job, ~$8-12 credits.
    Target val_avg_cost: ≤ 3.85 by iter ~50-60 (Stage 1 ceiling).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    k_int = int(k)
    buf_int = int(buffer_capacity)
    run_name = f"f61_main_K{k_int}_buf{buf_int}_{timestamp}"
    args = _f61_args(run_name=run_name, k=k_int, buffer_capacity=buf_int,
                      lr_model="5e-4", lr_decay="1.0")

    print(f"[modal] launching F.6.1 main run (K={k_int}, buf={buf_int}, lr=5e-4 const, 100 iter)")
    print(f"  {run_name}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f61_main_lrdecay(
    timestamp: str = "",
    k: str = "40",
    buffer_capacity: str = "5000",
) -> None:
    """Phase F.6.1-lrdecay variant — identical to run_f61_main but with
    initial lr=1e-3 and per-iteration lr_decay=0.95.

    lr trajectory: lr(0)=1e-3, lr(20)=3.6e-4, lr(50)=7.7e-5, lr(100)=5.9e-6.
    Geometric mean over 100 iters ≈ 7.7e-5.

    Question: does an aggressive AGZ-style schedule (high lr early, decay
    fast) outperform the constant lr=5e-4 in F.6.1 main? Theory says high
    lr early + decay = bigger early steps when buffer is small + small
    later steps when policy refines. Could converge faster and tighter.

    Risk: lr=1e-3 at random init is at the upper edge of stability for a
    small transformer with no warmup; if early iters spike or diverge,
    fall back to lr_model=5e-4.

    Run this in PARALLEL with run_f61_main (one Modal app each, two A10s).

    Expected wall-clock + cost: same as f61_main (~$8-12, ~4 h).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    k_int = int(k)
    buf_int = int(buffer_capacity)
    run_name = f"f61_main_lrdecay_K{k_int}_buf{buf_int}_{timestamp}"
    args = _f61_args(run_name=run_name, k=k_int, buffer_capacity=buf_int,
                      lr_model="1e-3", lr_decay="0.95")

    print(f"[modal] launching F.6.1 lrdecay variant (K={k_int}, buf={buf_int}, lr=1e-3, decay=0.95/iter, 100 iter)")
    print(f"  {run_name}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f61_tsp50_probe(timestamp: str = "") -> None:
    """Phase F.6.1 TSP-50 probe — same recipe as F.6.1 K=20 winner but at TSP-50.

    Settings: graph_size=50, K=50, ε=0.05, lr=5e-4 const, all other knobs
    inherited from F.6.1 main recipe (M=1000, train_steps=200, batch=512,
    buf=5000, gate_every=1, value_head, value_target_norm=none, wd=0,
    temperature_schedule=step30, n_iterations=100, val_seed=42, from-scratch).

    α = dirichlet_alpha_factor / N = 10/50 = 0.2 per action (vs 0.5 at TSP-20);
    α·N = 10, matching AGZ effective concentration.

    Question: does the F.6.1 recipe generalize to TSP-50? No prior Stage 4
    TSP-50 result; AM-paper TSP-50 lands ~5.7-5.85 greedy.

    Cost: ~$11-15 credits, ~5-7h wall-clock single-job (TSP-50 K=50 is
    ~6.25× more sim compute per instance than TSP-20 K=20).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    args = [
        "--graph_size", "50",
        "--n_iterations", "100",
        "--M_instances", "1000",
        "--n_simulations_train", "50",
        "--train_steps_per_iter", "200",
        "--buffer_capacity", "5000",
        "--batch_size", "512",
        "--gate_every", "1",
        "--gate_mode", "ttest",
        "--temperature_schedule", "step30",
        "--val_size", "10000",
        "--val_seed", "42",
        "--leaf_eval", "value_head",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", "5e-4",
        "--lr_decay", "1.0",
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", "0.05",
        "--dirichlet_alpha_factor", "10.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
    ]
    run_name = f"f61_tsp50_K50_eps05_{timestamp}"
    args = args + ["--run_name", run_name]

    print(f"[modal] launching F.6.1 TSP-50 probe (K=50, ε=0.05, lr=5e-4, 100 iter)")
    print(f"  {run_name}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f62_step10_eps_sweep(timestamp: str = "") -> None:
    """Phase F.6.2 — F.6.1 main recipe with step10 schedule and ε ∈ {0.05, 0.25}.

    Two parallel 100-iter from-scratch runs. All settings inherited from
    F.6.1 main (K=40, M=1000, train_steps=200, buffer_capacity=5000,
    batch=512, lr=5e-4 const, wd=0, value_target_norm=none,
    leaf_eval=value_head, gate=ttest gate_every=1, val_seed=42) — the
    ONLY differences from `run_f61_main` are:
      - --temperature_schedule step10  (cutoff = ⌈0.1·N⌉ = 2 for N=20;
        narrower stochastic-action window than F.6.1 main's step30)
      - --dirichlet_epsilon ∈ {0.05, 0.25}  (sweep)

    Diagnostic motivation (from F.6.1 main iter-99 analysis):
      - At K=40, value_head leaf eval was tied with greedy (Δ = +0.002).
      - π_t entropy collapsed to 0.155 nats by iter 99 → near-one-hot targets
        from the policy's own argmax, no improvement signal to distill.
      - step30 added stochastic τ=1 sampling at first 6/20 steps; combined
        with an uninformative value head, this seemed to hurt MCTS quality
        rather than help (training-time MCTS = greedy + 0.008).

    F.6.2 question: does narrowing the σ_t stochastic window (step10) +
    AGZ-canonical exploration noise (ε=0.25) restore the improvement
    signal? ε=0.05 is the warm-start-safe baseline (proven on Stage-1
    warm-start probes); ε=0.25 is AGZ-canonical for from-scratch.

    The trainer.py per-loss grad-norm logging from F.6.2's prep is also live
    — iterations.csv now reports policy_grad_norm_mean / value_grad_norm_mean
    / grad_norm_mean alongside the existing loss/entropy series.

    Cost: ~$15-22 Modal credits, ~3.5-4.5h wall-clock parallel (2× A10).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    variants = [
        ("f62_step10_eps05", "0.05"),
        ("f62_step10_eps25", "0.25"),
    ]
    grid = []
    for label_stem, eps_val in variants:
        run_name = f"{label_stem}_{timestamp}"
        args = _f61_args(
            run_name=run_name,
            k=40,
            buffer_capacity=5000,
            lr_model="5e-4",
            lr_decay="1.0",
            temperature_schedule="step10",
            dirichlet_epsilon=eps_val,
        )
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel F.6.2 step10 ε-sweep jobs (timestamp={timestamp})")
    for label, _ in grid:
        print(f"  {label}")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} F.6.2 step10 ε-sweep jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_rollout_lambda_ablation(
    timestamp: str = "",
    n_iterations: str = "50",
    include_weak: bool = False,
) -> None:
    """Rollout-teacher value-loss ablation.

    Runs leaf_eval=rollout, K=40, step10, epsilon=0.25, raw value targets,
    and F.6.1 optimizer/replay defaults. The primary axis is lambda_v:
    0.0 for policy-only distillation, 1.0 for the AGZ-style value-loss
    control, and optionally 0.1 for weak auxiliary regularization.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    n_iter = int(n_iterations)
    variants = [
        ("rollout_lv0", "0.0"),
        ("rollout_lv1", "1.0"),
    ]
    if include_weak:
        variants.insert(1, ("rollout_lv01", "0.1"))

    grid = []
    for label_stem, lambda_val in variants:
        run_name = f"{label_stem}_K40_step10_eps25_{n_iter}iter_{timestamp}"
        args = _f61_args(
            run_name=run_name,
            k=40,
            buffer_capacity=5000,
            lr_model="5e-4",
            lr_decay="1.0",
            temperature_schedule="step10",
            dirichlet_epsilon="0.25",
            leaf_eval="rollout",
            lambda_v=lambda_val,
        )
        idx = args.index("--n_iterations")
        args[idx + 1] = str(n_iter)
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} rollout lambda_v ablation jobs (timestamp={timestamp})")
    for label, args in grid:
        lambda_idx = args.index("--lambda_v")
        print(f"  {label} (lambda_v={args[lambda_idx + 1]})")

    handles = {
        label: train_alphazero_remote.spawn(*args) for label, args in grid
    }
    print(f"\n[modal] all {len(handles)} rollout-lambda jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} rollout lambda_v ablation jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_rollout_lv0_resume50_to_iter99(timestamp: str = "") -> None:
    """Resume the rollout lambda_v=0 ablation for +50 more iters (iter 50 -> 99).

    Pairs with run_rollout_lambda_ablation's lv0 variant (W&B `1syc0kk8`,
    name `rollout_lv0_K40_step10_eps25_50iter_20260509T102959_...`). After
    50 iters lv0 reached val_avg_cost = 3.879 with non-saturated downward
    slope. This +50 iter resume tests whether leaf_eval=rollout + lambda_v=0
    breaks 3.85 and approaches Stage 1 canonical greedy (3.83943).

    Same recipe verbatim from _f61_args plus rollout-ablation overrides:
    leaf_eval=rollout, lambda_v=0.0, K=40, step10, eps=0.25,
    value_target_norm=none, lr=5e-4 const, wd=0, buffer=5000, batch=512,
    train_steps=200, val_seed=42, gate=ttest gate_every=1, M=1000,
    mcts_batch_size=1000.

    Cost: ~$8-12 Modal credits, ~1.2h wall on A10.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_20/"
        "rollout_lv0_K40_step10_eps25_50iter_20260509T102959_20260509T103008/"
        "iter-49.pt"
    )
    run_name = f"rollout_lv0_resume50_to99_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    # Override n_iterations 100 -> 50 (resume runs +50 from iter 50 -> 99).
    idx = args.index("--n_iterations")
    args[idx + 1] = "50"
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching rollout lambda_v=0 resume (+50 iter, target 50->99) from iter-49.pt")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_rollout_lv0_resume100_lr1e4_to_iter199(timestamp: str = "") -> None:
    """Resume the rollout lambda_v=0 chain at lr=1e-4 for +100 iters (iter 100->199).

    Continuation of run_rollout_lv0_resume50_to_iter99 (W&B `d8uyrrm1`,
    val_avg_cost=3.8607 at iter 99). At constant lr=5e-4 the recipe saturated
    around iter 70-90 with slope -0.0005/iter (vs -0.010/iter early). Mirrors
    the F.6.1.4.b lever (lr 5e-4 -> 1e-4) that broke F.6.1.4's iter-127
    plateau and dropped val 3.8665 -> 3.8514 in 35 iters.

    Same recipe: leaf_eval=rollout, lambda_v=0, K=40, step10, epsilon=0.25,
    value_target_norm=none, wd=0, buffer=5000, batch=512, train_steps=200,
    val_seed=42, gate=ttest gate_every=1, M=1000, mcts_batch_size=1000.
    Only knob change: lr_model 5e-4 -> 1e-4.

    train_alphazero.py's lr-override-on-resume (added 2026-05-07) applies
    --lr_model AFTER coach.load_checkpoint so the optimizer's loaded lr +
    LambdaLR base_lrs are overwritten to 1e-4.

    Cost: ~$15-20 Modal credits, ~2.2h wall on A10 (100 iter x ~80s/iter).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_20/"
        "rollout_lv0_resume50_to99_20260510T001900_20260510T001908/"
        "iter-99.pt"
    )
    run_name = f"rollout_lv0_resume100_lr1e4_to199_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=5000,
        lr_model="1e-4",                  # KEY: dropped from 5e-4
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    # Override n_iterations 100 -> 100 (already correct from _f61_args default; this
    # asserts intent); resume runs +100 from iter 100 -> 199.
    idx = args.index("--n_iterations")
    args[idx + 1] = "100"
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching rollout lambda_v=0 +100 iter resume at lr=1e-4 (target 100->199) from iter-99.pt")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    print(f"  lr override: 5e-4 -> 1e-4 (applied AFTER load_checkpoint)")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp50_lv0_K50_50iter(timestamp: str = "") -> None:
    """TSP-50 lv0 from-scratch probe — 50 iters at K=50 (Track 4 Phase A+B initial).

    First TSP-50 run with the lv0 winner recipe (leaf_eval=rollout, lambda_v=0)
    that landed in the rollout_value_ablation chain at TSP-20 (lv0 chain best
    3.8486 at iter 197; greedy 0.007 better than F.6.1.6 vh+lambda_v=1).

    Reference points (TSP-50, val_seed=42):
      Stage 1 canonical (123x2qr5 AM+value greedy): 5.7999
      Stage 0 pretrained TSP-50 greedy:             5.7955
      Stage 4 best so far (muckiyvi vh+lambda_v=1, K=50, eps=0.05, 100 iter):
        best 6.060 @ iter 95
      Gurobi TSP-50 (1000 inst seed=1234):          5.6987

    Recipe: lv0 winner verbatim from `_f61_args` plus:
      leaf_eval=rollout, lambda_v=0.0, K=50 (per user's "speed first" choice;
      Phase B may bump to K=100 if K=50 saturates noisily), step10, eps=0.25,
      value_target_norm=none, lr=5e-4 const, wd=0, buffer=5000, batch=512,
      train_steps=200, M=1000, val_seed=42, gate=ttest gate_every=1.

    `mcts_batch_size=1000` set explicitly to assert the production
    cross-instance parallelism (default per F.6.1.4.c sweep, but lifted from
    CLI default into the entrypoint args for clarity).

    Wall estimate: K=50 rollout at TSP-50 ~= 2.5x TSP-20 K=50 rollout per iter
    ~= 100-150 s/iter on A10. 50 iters = 1.5-2 h. Cost: ~$8-15 Modal credits.

    Goals (50-iter probe):
      - Verify lv0 recipe transfers to TSP-50 (no OOM, val_avg_cost descending).
      - Beat existing Stage 4 best 6.060 in 50 iters.
      - If trajectory still descending at iter 49, escalate to lr=1e-4 resume
        +100 iters (mirroring TSP-20 lv0 chain).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    run_name = f"tsp50_lv0_K50_50iter_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=50,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    # Patch graph_size 20 -> 50 (mirrors run_tsp50_k_compare_20iter pattern).
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "50"
    # Override n_iterations 100 -> 50.
    idx_n = args.index("--n_iterations")
    args[idx_n + 1] = "50"
    # Assert production-scale cross-instance parallelism.
    args.extend(["--mcts_batch_size", "1000"])

    print(f"[modal] launching TSP-50 lv0 from-scratch (K=50, 50 iter, leaf=rollout, lambda_v=0)")
    print(f"  {run_name}")
    print(f"  graph_size=50  K=50  lr=5e-4  mcts_batch_size=1000")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp100_lv0_K50_50iter(timestamp: str = "") -> None:
    """TSP-100 lv0 from-scratch probe — 50 iters at K=50 (proposal Stage 5 scaling).

    First TSP-100 run with the lv0 winner recipe (leaf_eval=rollout, lambda_v=0)
    that landed at TSP-20 (lv0 iter-197 best val 3.8486 greedy, beats Stage 1
    canonical 3.83943 at ~6.4x sample efficiency) and is in flight at TSP-50
    (`0d48yqys` finished iter 99 at val 6.0294 last-accept / 6.0259 raw best;
    follow-up `tsp50_lv0_K50_resume100_lr1e4_to199_trackA` at lr=1e-4 launched
    2026-05-13).

    Reference points (TSP-100):
      LKH3 (near-optimal, val_seed=1234):              7.749   (outputs/baselines/tsp100_lkh_seed1234.csv)
      AM-paper released greedy:                        ~8.13
      AM-paper released sampling-1280:                 7.928   (Stage 3 D.3 anchor)
      Stage 1 reduced-compute (bs=1024, ep=640k):      8.21043 (g7jxkixo @ epoch 99; reduced-compute caveat)
      Released AM ckpt + MCTS rollout K=50 (val seed=1234, val_size=1000): 7.9217 (Stage 3 D.4)

    NB: Stage 4 uses val_seed=42, val_size=10000 — DIFFERENT from the seed=1234
    val sets the Stage 1/3 numbers above were measured on. Apples-to-apples
    against Stage 1 TSP-100 (8.21 at seed=1234) would require re-running the
    reduced-compute ckpt on val_seed=42. For headline-purposes the LKH 7.749
    reference is robust across seeds (TSP is rotation/scale-invariant; LKH
    converges to optimum on essentially all uniform-2D instances).

    Recipe: lv0 winner verbatim from `_f61_args` plus graph_size=100:
      leaf_eval=rollout, lambda_v=0.0, K=50, step10, eps=0.25,
      value_target_norm=none, lr=5e-4 const, wd=0, buffer=5000, batch=512,
      train_steps=200, M=1000, val_seed=42, gate=ttest gate_every=1,
      mcts_batch_size=1000.

    **From-scratch (no --load_path).** Matches the TSP-20 lv0 and TSP-50 lv0
    chains. Open question: would warm-starting from the existing Stage 1
    TSP-100 ckpt (`stage1_tsp100_bs1024_ep640k_with_value_20260428T233519/
    epoch-99.pt`, val 8.21) accelerate convergence vs from-scratch? The TSP-50
    lv0 chain only ran from-scratch; we don't have evidence either way. From-
    scratch keeps the sample-efficiency story clean (instances counted from 0)
    but may take longer to reach competitive val at TSP-100. Worth a follow-up
    warm-start variant if from-scratch struggles.

    Wall estimate (extrapolated from TSP-50 K=50 M=1000 = 5.85 min/iter
    post-Track A; TSP-100 scales roughly 2-3x per iter — N=100 vs N=50 doubles
    per-rollout decoder calls AND doubles per-NN-call work, plus K=50 sims at
    100 tour-steps = 2x simulations per instance):
      Per-iter wall:  ~12-18 min on A10
      50 iters:       ~10-15 h
      Cost:           ~$30-50 Modal credits

    Goals (50-iter probe):
      - **Verify lv0 recipe transfers to TSP-100** (no OOM at M=1000, no Triton/
        decoder issues at N=100, val_avg_cost descending iteration-by-iteration).
      - Beat existing Stage 1 reduced-compute baseline 8.21 within 50 iters
        (should be easy; lv0 at TSP-50 was beating Stage 4 prior best by iter ~30).
      - Reach the LKH-gap zone (target val ~7.95-8.05) — would put us in the
        same ballpark as released-AM sampling-1280 (7.928) at <50K instances
        vs the AM paper's 1.28M.
      - If trajectory still descending at iter 49, escalate to lr=1e-4 resume
        +100 iters (mirroring TSP-20/TSP-50 lv0 chains).

    Open knobs to consider before launch (defaults below; flag for review):
      - **K=50 vs K=100**: K=50 matches TSP-50 production and halves wall;
        K=100 doubles search budget per step but doubles wall. The TSP-20
        A.3 finding was "K dominates over batch and M", which suggests K=100
        could help; but at TSP-100 we already get ~2x simulations per
        instance vs TSP-50 K=50 just from the longer tour. Default K=50.
      - **From-scratch vs warm-start** from Stage 1 ckpt: default from-scratch.
      - **50 iters vs 100 iters**: 100 iters would hit Modal's 24h timeout
        envelope at 12-18 min/iter (50 iters = 10-15h is safe). Default 50;
        chain at lr=1e-4 for +100 if the 50-iter slope is still healthy.
      - **A10 vs A100**: default A10 (current image GPU). A100 would cut wall
        by ~2.5x but $/hr is ~2.7x, so cost is similar; A100 wins on real-time.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    run_name = f"tsp100_lv0_K50_50iter_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=50,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    # Patch graph_size 20 -> 100.
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "100"
    # Override n_iterations 100 -> 50 (probe scope; chain at lr=1e-4 for +100 later).
    idx_n = args.index("--n_iterations")
    args[idx_n + 1] = "50"
    # Production-scale cross-instance parallelism.
    args.extend(["--mcts_batch_size", "1000"])

    print(f"[modal] launching TSP-100 lv0 from-scratch (K=50, 50 iter, leaf=rollout, lambda_v=0)")
    print(f"  {run_name}")
    print(f"  graph_size=100  K=50  lr=5e-4  mcts_batch_size=1000  M=1000")
    print(f"  wall estimate: ~12-18 min/iter -> 10-15h total on A10")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp50_lv0_resume_from_iter15_trackA(timestamp: str = "") -> None:
    """Resume TSP-50 lv0 from `oxjyj70e` iter-15.pt for +34 iters (target iter 49).

    Continues the `oxjyj70e` (Fix #5) trajectory where it was stopped at
    iter 15 (best val 6.4959 at iter 8). The new code state has Track A
    (per-row state.i in decoder + merged step groups in rollout/eval),
    which dropped the M=1000 Modal probe wall from ~850s -> 435s = -49%
    on a trained checkpoint.

    Same recipe verbatim from `_f61_args` (lv0 winner: leaf_eval=rollout,
    lambda_v=0.0, K=50, step10, eps=0.25, value_target_norm=none,
    lr=5e-4 const, wd=0, buffer=5000, batch=512, train_steps=200,
    val_seed=42, gate=ttest gate_every=1, M=1000, mcts_batch_size=1000).

    Resume target: complete the 50-iter trajectory (iter 16..49 = 34 more
    iters). At Track A's expected 7-8 min/iter on A10 -> ~4-5h wall,
    ~$15-20 Modal credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_50/"
        "tsp50_lv0_K50_50iter_20260511T051358_20260511T051406/"
        "iter-15.pt"
    )
    run_name = f"tsp50_lv0_K50_resume34_from_iter15_trackA_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=50,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    # Patch graph_size 20 -> 50.
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "50"
    # +34 iters to reach iter 49 (oxjyj70e ran iter 0..15).
    idx_n = args.index("--n_iterations")
    args[idx_n + 1] = "34"
    args.extend(["--mcts_batch_size", "1000"])
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching TSP-50 lv0 resume from iter-15.pt (+34 iter, target iter 49) with Track A")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    print(f"  graph_size=50  K=50  lr=5e-4  mcts_batch_size=1000")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp50_lv0_resume50_to_iter99_trackA(timestamp: str = "") -> None:
    """Continue TSP-50 lv0 chain from `1wpkngg9` iter-49.pt for +50 iters (target iter 99).

    Continuation of the Track A run `1wpkngg9` (oxjyj70e iter-15 -> Track A
    iter-49) which finished at val_avg_cost=6.1338 (best 6.1406 at iter 43),
    0.08 above the Stage 4 prior best muckiyvi 6.060 at 100 iters. This
    +50 iter resume keeps lr=5e-4 const (no decay) and same lv0 recipe to
    see if more iters at the same rate close the gap to muckiyvi 6.060 and
    approach Stage 1 5.80.

    Same recipe verbatim (lv0: leaf_eval=rollout, lambda_v=0.0, K=50,
    step10, eps=0.25, value_target_norm=none, lr=5e-4 const, wd=0,
    buffer=5000, batch=512, train_steps=200, val_seed=42, gate=ttest
    gate_every=1, M=1000, mcts_batch_size=1000).

    Wall: Track A held ~5 min/iter through iter 16-49. Expect 50 iters in
    ~4-5h on A10, ~$15-20 Modal credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_50/"
        "tsp50_lv0_K50_resume34_from_iter15_trackA_20260511T093124_20260511T093133/"
        "iter-49.pt"
    )
    run_name = f"tsp50_lv0_K50_resume50_to99_trackA_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=50,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "50"
    # +50 iters to reach iter 99 (1wpkngg9 finished at iter 49).
    idx_n = args.index("--n_iterations")
    args[idx_n + 1] = "50"
    args.extend(["--mcts_batch_size", "1000"])
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching TSP-50 lv0 resume from 1wpkngg9 iter-49.pt (+50 iter, target iter 99)")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    print(f"  graph_size=50  K=50  lr=5e-4 const  mcts_batch_size=1000")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp50_lv0_resume100_lr1e4_to_iter199(timestamp: str = "") -> None:
    """Continue TSP-50 lv0 chain from `0d48yqys` iter-99.pt for +100 iters at lr=1e-4 (target iter 199).

    TSP-50 analog of `run_rollout_lv0_resume100_lr1e4_to_iter199` (the TSP-20
    D.4 chain that broke the 3.85 greedy ceiling).

    Continuation of `0d48yqys` (`tsp50_lv0_K50_resume50_to99_trackA_20260513T064031`)
    which finished at iter 99 with:
      - best val 6.0259 (iter 97, raw greedy)
      - last gate-accepted best_model 6.0294 (iter 90)
      - segment slope iter 75-99 = ~-0.002/iter (lr=5e-4 well filling but not
        saturated; mirrors TSP-20 D.3 endpoint at iter 99)

    Reference points (TSP-50, val_seed=42):
      Gurobi:                              5.6987
      Stage 1 canonical (AM+value greedy): 5.7999
      Stage 4 prior best (muckiyvi vh+lv1, 100 iters): 6.060
      lv0 chain iter 99 (this resume's input):         6.0294

    Hypothesis (from the TSP-20 D.3 -> D.4 pattern): at constant lr=5e-4 the
    lv0 chain saturates around iter 90-100 with slope -0.001 to -0.002/iter.
    Resuming at lr=1e-4 should fire immediately and unlock another ~0.01-0.02
    in the first 3-5 iters, asymptoting around iter 150-180 at val ~5.95-6.00.
    Still 0.15-0.20 above Stage 1 5.7999 — would need either F.6.1.6 step-decay
    or §E.4 step-decay 400-iter to fully close.

    Same recipe verbatim (lv0: leaf_eval=rollout, lambda_v=0.0, K=50, step10,
    eps=0.25, value_target_norm=none, wd=0, buffer=5000, batch=512,
    train_steps=200, val_seed=42, gate=ttest gate_every=1, M=1000,
    mcts_batch_size=1000). Only knob change: **lr_model 5e-4 -> 1e-4**.

    `train_alphazero.py`'s lr-override-on-resume (added 2026-05-07) applies
    `--lr_model` AFTER `coach.load_checkpoint` so the optimizer's loaded lr +
    LambdaLR base_lrs are overwritten to 1e-4. Verified at TSP-20 (D.4 chain
    fired the -0.006/3-iter drop expected from the F.6.1.4.b pattern).

    Wall estimate: Track A holds ~5.85 min/iter at TSP-50 K=50 M=1000. 100
    iters = ~9.7h on A10. Cost: ~$25-30 Modal credits.

    Open question: does the lr unlock pattern that worked at TSP-20 transfer
    cleanly to TSP-50 where the absolute gap to optimum is ~6x larger? If yes
    -> commits the lv0 recipe as the TSP-N scaling story; opens §E.4 400-iter
    step-decay as the next-tier consolidation. If no (lr=1e-4 doesn't unlock
    or unlocks tiny <0.01) -> suggests TSP-50 needs more structural changes
    (decoupled value head, separate encoder, etc.).
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_50/"
        "tsp50_lv0_K50_resume50_to99_trackA_20260513T064031_20260513T064039/"
        "iter-99.pt"
    )
    run_name = f"tsp50_lv0_K50_resume100_lr1e4_to199_trackA_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=50,
        buffer_capacity=5000,
        lr_model="1e-4",                  # KEY: dropped from 5e-4
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "50"
    # +100 iters to reach iter 199 (0d48yqys finished at iter 99).
    idx_n = args.index("--n_iterations")
    args[idx_n + 1] = "100"
    args.extend(["--mcts_batch_size", "1000"])
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching TSP-50 lv0 +100 iter resume at lr=1e-4 (target iter 199) from 0d48yqys iter-99.pt")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    print(f"  graph_size=50  K=50  mcts_batch_size=1000")
    print(f"  lr override: 5e-4 -> 1e-4 (applied AFTER coach.load_checkpoint)")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f62_eps25_resume50(timestamp: str = "") -> None:
    """Resume f62_step10_eps25 from iter-99.pt for +50 iterations.

    Same recipe as run_f62_step10_eps_sweep ε=0.25 (step10, K=40, M=1000,
    train_steps=200, buffer=5000, batch=512, lr=5e-4 const, wd=0,
    value_target_norm=none, leaf_eval=value_head, gate=ttest gate_every=1,
    val_seed=42). Loads model + best_model + optimizer + lr_scheduler + RNG
    from iter-99.pt and the sibling buffer.pt; coach resumes at iter 100
    and runs through iter 149.

    Picks up the value-grad split telemetry added after the original run
    finished — iterations.csv on the resumed run will include
    `value_grad_norm_vh_mean` + `value_grad_norm_shared_mean` so we can
    compute cos(θ_shared) iter-by-iter.

    Cost: ~$8-12 Modal credits, ~1.5-2h wall-clock single A10.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_20/"
        "f62_step10_eps25_20260507T024345_20260507T024355/iter-99.pt"
    )
    run_name = f"f62_step10_eps25_resume50_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
    )
    # Override n_iterations 100 -> 50 (resume runs +50 from iter 100 → 149).
    idx = args.index("--n_iterations")
    args[idx + 1] = "50"
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching F.6.2 ε=0.25 resume (+50 iter, target 100→149) from iter-99.pt")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f64_eps25_lr1e4_resume50(timestamp: str = "") -> None:
    """Resume F.6.1.4 (f62_step10_eps25_resume50_*) from iter-149.pt for +50
    more iterations at lr=1e-4 (down from 5e-4).

    Diagnostic motivation: F.6.1.4 last gate accept was at iter 127 (val=3.866);
    iter 128-149 stalled (gate rejected every time). Working-model val_avg_cost
    bounced 3.866-3.883 across those 22 rejected iters — neither degrading
    nor improving past iter 127.

    Hypothesis: at iter 127 the policy is close enough to a fixed point that
    lr=5e-4 is overshooting per-iter. A smaller lr=1e-4 might let the model
    refine into a tighter optimum without the gradient noise that's been
    knocking it around. (lr=1e-4 was the AM/Stage 1 canonical; the F.6.0.5
    derivation that picked 5e-4 was for from-scratch random-init drift, which
    no longer applies at the F.6.1.4 endpoint.)

    train_alphazero.py now honors --lr_model on a resumed run by overriding
    the optimizer's loaded lr + the LambdaLR base_lrs (added 2026-05-07; was
    previously a silent no-op because optimizer.load_state_dict restores the
    saved lr).

    Cost: ~$5-7 Modal credits, ~75 min wall-clock single A10. Outputs to
    `outputs/tsp_20/f62_step10_eps25_resume50_lr1e4_<timestamp>_*/`.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_20/"
        "f62_step10_eps25_resume50_20260507T052131_20260507T052139/iter-149.pt"
    )
    run_name = f"f62_step10_eps25_resume50_lr1e4_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=5000,
        lr_model="1e-4",                  # ← KEY: dropped from 5e-4
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
    )
    idx = args.index("--n_iterations")
    args[idx + 1] = "50"
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching F.6.1.4.b ε=0.25 +50 iter resume at lr=1e-4 (from iter-149.pt)")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    print(f"  lr override: 5e-4 → 1e-4 (applied AFTER load_checkpoint)")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_mcts_batch_size_smoke(bsz: str = "512", timestamp: str = "") -> None:
    """Quick 5-iter smoke at a non-default mcts_batch_size to verify GPU
    saturation lever (analysis: see plan
    `interesting-can-you-analyze-tidy-bubble.md`).

    mcts_batch_size is the instance-parallelism CHUNK SIZE in
    CppBatchMCTSSolver.solve_batch (NOT the per-NN-forward batch). With
    M=1000 instances and the current default mcts_batch_size=64, each
    coach iter sequentially processes 16 chunks. Bumping the chunk size
    fewer-but-larger chunks AND larger NN-eval batches per call.

    Recipe = F.6.1.3 eps=0.25 verbatim except n_iterations=5 and
    --mcts_batch_size <bsz>. Cost: ~5 min wall, ~$0.20 credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    bsz_int = int(bsz)
    run_name = f"mcts_bsz{bsz_int}_smoke_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
    )
    idx = args.index("--n_iterations")
    args[idx + 1] = "5"
    args.extend(["--mcts_batch_size", str(bsz_int)])

    print(f"[modal] launching mcts_batch_size smoke (bsz={bsz_int}, 5 iter)")
    print(f"  {run_name}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_mcts_batch_size_sweep(timestamp: str = "") -> None:
    """4-variant sweep over mcts_batch_size at fixed F.6.1.3 eps=0.25 recipe.

    Variants: {64, 256, 1000, 2000}. The 1000 variant equals M_instances (one
    chunk per iter); 2000 is a sanity check that going past M doesn't break
    anything (effectively a single chunk at min(2000, 1000)=1000 fill). Each
    variant runs 10 iters; we read mean(mcts_s) over iters 1-9 (skip iter 0
    due to first-iter overhead) to compare wall-clock saturation.

    All other settings match F.6.1.3 eps=0.25 (step10, K=40, M=1000,
    train_steps=200, buffer=5000, batch=512, lr=5e-4, val_seed=42).

    Cost: ~$3-5 credits, ~30 min parallel wall. Goal: pick the chunk size
    that minimizes mcts_s/iter; commit it as the new default in
    train_alphazero.py.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    variants = [64, 256, 1000, 2000]
    grid = []
    for bsz in variants:
        run_name = f"mcts_bsz{bsz}_sweep10iter_{timestamp}"
        args = _f61_args(
            run_name=run_name,
            k=40,
            buffer_capacity=5000,
            lr_model="5e-4",
            lr_decay="1.0",
            temperature_schedule="step10",
            dirichlet_epsilon="0.25",
        )
        idx = args.index("--n_iterations")
        args[idx + 1] = "10"
        args.extend(["--mcts_batch_size", str(bsz)])
        grid.append((run_name, args))

    print(f"[modal] launching {len(grid)} parallel mcts_batch_size sweep jobs (10 iter each)")
    for label, _ in grid:
        print(f"  {label}")

    handles = {label: train_alphazero_remote.spawn(*args) for label, args in grid}
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} mcts_batch_size sweep jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp50_k50_resume50(timestamp: str = "") -> None:
    """Resume the TSP-50 K=50 ablation winner (iter 19 = val 6.651) for +50
    more iterations (iter 20 -> 69) at the same recipe.

    Hypothesis: K=50 was descending smoothly through iter 19 (6.65, still
    accepting late). Continuing for 50 more iters at the same lr=5e-4 const
    (lr_decay step boundary at iter 100 not reached) should drive val below
    6.0 and clarify whether the lr=1e-4 transition (at iter 100 in the
    eventual main run) is the right next lever for TSP-50.

    Same recipe verbatim: K=50, M=1000, train_steps=200, buffer=5000,
    batch=512, lr=5e-4, lr_decay=0.2 step_size=100 (won't trigger at 50
    more iters), step10, eps=0.25, value_target_norm=none,
    leaf_eval=value_head, gate=ttest gate_every=1, val_seed=42,
    mcts_batch_size=1000.

    Cost: ~50 iter * ~155s/iter = ~130 min wall, ~$5-7 credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_50/"
        "tsp50_k50_compare_20iter_20260507T104500_20260507T104508/iter-19.pt"
    )
    run_name = f"tsp50_k50_resume50_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=50,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="0.2",
        lr_decay_step_size=100,
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
    )
    # Override graph_size 20 -> 50 and n_iterations 100 -> 50.
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "50"
    idx_n = args.index("--n_iterations")
    args[idx_n + 1] = "50"
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching TSP-50 K=50 resume +50 iter (iter 20 -> 69)")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp50_k_compare_20iter(timestamp: str = "") -> None:
    """TSP-50 K-comparison ablation: K=50 vs K=100, 20 iters each, parallel.

    Quick decision-driver before committing to a multi-day TSP-50 main run.
    Hypothesis: at TSP-50 with N=50 legal actions at step 0, K=50 gives
    ~1 visit/action — very thin MCTS. K=100 gives ~2 visits/action; π_t
    target quality may be materially better, justifying 2x per-iter wall.

    Decision rule after 20 iters: pick the K with lower val_avg_cost at
    iter 19. If K=100 wins by >=0.05 cost units, use K=100 for the
    400-iter TSP-50 main run; otherwise stick with K=50.

    All other settings = F.6.1.6 TSP-50 mapping (M=1000, train_steps=200,
    batch=512, buffer=5000, lr=5e-4 const within 20 iters,
    temperature_schedule=step10, eps=0.25, dirichlet_alpha_factor=10.0,
    value_target_norm=none, leaf_eval=value_head, gate=ttest gate_every=1,
    val_seed=42, mcts_batch_size=1000 default).

    Per-iter wall predicted: ~100-120s (K=50), ~200-240s (K=100). Run
    both in parallel -> ~70-80 min total wall, ~$8-12 credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    variants = [50, 100]
    grid = []
    for k in variants:
        run_name = f"tsp50_k{k}_compare_20iter_{timestamp}"
        args = _f61_args(
            run_name=run_name,
            k=k,
            buffer_capacity=5000,
            lr_model="5e-4",
            lr_decay="0.2",
            lr_decay_step_size=100,
            temperature_schedule="step10",
            dirichlet_epsilon="0.25",
        )
        # Override graph_size 20 -> 50 and n_iterations 100 -> 20.
        idx_g = args.index("--graph_size")
        args[idx_g + 1] = "50"
        idx_n = args.index("--n_iterations")
        args[idx_n + 1] = "20"
        grid.append((run_name, args))

    print(f"[modal] launching TSP-50 K-comparison ablation (K=50 vs K=100, 20 iter each)")
    for label, _ in grid:
        print(f"  {label}")

    handles = {label: train_alphazero_remote.spawn(*args) for label, args in grid}
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} TSP-50 K-comparison jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp100_k_compare_20iter(timestamp: str = "") -> None:
    """TSP-100 K-comparison ablation: K=50 vs K=100, 20 iters each, parallel (lv0 recipe).

    Mirrors `run_tsp50_k_compare_20iter` at TSP-100 scale with the **lv0
    recipe** (leaf_eval=rollout, lambda_v=0) — the current production winner.
    Decision-driver before committing to a 50-iter or 100-iter TSP-100 main
    run.

    Hypothesis: at N=100 the action space is 2x TSP-50's. K=50 gives ~0.5
    visit/action at root (under-explored); K=100 gives ~1 visit/action. The
    TSP-20 A.3 finding ("K dominates over batch and M, ~10x the batch effect")
    suggests K=100 may help materially at TSP-100 — but at 2x per-iter wall
    cost. This probe answers: is the quality gain at iter 19 worth the wall?

    Decision rule after 20 iters:
      - If K=100 val(19) <= K=50 val(19) - 0.05 (cost units): K=100 wins
        clearly; commit K=100 for the 50-iter main run.
      - If K=50 val(19) is within ~0.05 of K=100 (or better): K=50 wins on
        cost-efficiency; commit K=50.

    Settings (both variants, lv0 production recipe):
      graph_size=100, M=1000, mcts_batch_size=1000, train_steps=200,
      batch=512, buffer=5000, lr=5e-4 const, wd=0, value_target_norm=none,
      lambda_v=0.0, leaf_eval=rollout, step10, dirichlet_epsilon=0.25,
      dirichlet_alpha_factor=10.0, gate=ttest gate_every=1, val_seed=42.

    Per-iter wall predicted (extrapolating from TSP-50 K=50 ~5.85 min/iter
    post-Track A; TSP-100 scales ~2-3x per iter from TSP-50 due to 2x rollout
    length + 2x per-NN-call work; K=100 doubles simulations):
      K=50  variant:  ~12-18 min/iter  -> 20 iter = ~4-6 h
      K=100 variant:  ~24-36 min/iter  -> 20 iter = ~8-12 h
      Parallel total: ~8-12 h wall (max of the two), ~$45-75 credits.

    Both fit well within Modal's 24h timeout.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    variants = [50, 100]
    grid = []
    for k in variants:
        run_name = f"tsp100_k{k}_lv0_compare_20iter_{timestamp}"
        args = _f61_args(
            run_name=run_name,
            k=k,
            buffer_capacity=5000,
            lr_model="5e-4",
            lr_decay="1.0",
            temperature_schedule="step10",
            dirichlet_epsilon="0.25",
            leaf_eval="rollout",
            lambda_v="0.0",
        )
        # Override graph_size 20 -> 100 and n_iterations 100 -> 20.
        idx_g = args.index("--graph_size")
        args[idx_g + 1] = "100"
        idx_n = args.index("--n_iterations")
        args[idx_n + 1] = "20"
        # Production-scale cross-instance parallelism.
        args.extend(["--mcts_batch_size", "1000"])
        grid.append((run_name, args))

    print(f"[modal] launching TSP-100 K-comparison ablation (K=50 vs K=100, 20 iter each, lv0 recipe)")
    for label, _ in grid:
        print(f"  {label}")

    handles = {label: train_alphazero_remote.spawn(*args) for label, args in grid}
    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()
        print(f"[modal] {label} done.", flush=True)
    print(f"\n[modal] all {len(handles)} TSP-100 K-comparison jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp100_k50_lv0_resume30_to49(timestamp: str = "") -> None:
    """TSP-100 K=50 lv0 +30 iter resume from iter-19.pt of the K-comparison run.

    Continues the K=50 arm of `run_tsp100_k_compare_20iter` (wandb `dbv1cr8o`,
    finished 2026-05-13 at iter 19, val_avg_cost=9.1009; best raw 8.9671 at
    iter 17 = last accept). Same recipe verbatim, lr=5e-4 const — no lr-drop
    yet because at iter 19 the trajectory is still improving (8.99 at iter 13
    -> 8.97 at iter 17; the F.6.1.4-style lr-drop is reserved for after the
    5e-4 well empties, mirroring the TSP-50 0->49 -> 50->99 chain shape).

    Resumes from `iter-19.pt` (model + best_model + optimizer + lr_scheduler
    + RNG + buffer.pt) for +30 iters -> target iter 49.

    Wall estimate: K=50 at TSP-100 averaged ~1660 s/iter (~27.7 min) in the
    20-iter compare run. 30 iters ~= 14h on A10. Cost: ~$30-40 Modal credits.

    Decision-driver for the next chain link: if val at iter 49 beats ~8.8
    -> commit lv0 recipe for TSP-100 main run; if plateau at 8.9 -> trigger
    lr-drop to 1e-4 mirroring §B.2 / §E.2.b.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_100/"
        "tsp100_k50_lv0_compare_20iter_20260513T120814_20260513T120821/"
        "iter-19.pt"
    )
    run_name = f"tsp100_k50_lv0_resume20_to49_trackA_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=50,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "100"
    # +30 iters to reach iter 49 (compare run finished at iter 19).
    idx_n = args.index("--n_iterations")
    args[idx_n + 1] = "30"
    args.extend(["--mcts_batch_size", "1000"])
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching TSP-100 K=50 lv0 +30 iter resume (target iter 49) from dbv1cr8o iter-19.pt")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    print(f"  graph_size=100  K=50  mcts_batch_size=1000  lr=5e-4 const")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp20_k10_lv0_step50(timestamp: str = "") -> None:
    """TSP-20 K=10 lv0 100-iter with built-in lr step at iter 50.

    Schedule (one --lr_decay 0.2 --lr_decay_step_size 50 cycle):
        iter  0..49: lr = 5e-4
        iter 50..99: lr = 1e-4   (decay factor 0.2)

    Recipe: lv0 winner verbatim (leaf_eval=rollout, lambda_v=0, step10,
    eps=0.25, value_target_norm=none, wd=0, buffer=5000, batch=512,
    train_steps=200, M=1000, val_seed=42, gate=ttest gate_every=1,
    mcts_batch_size=1000) — only K and lr-schedule are non-default.

    Open question: K=10 is well below the A.3 sub-budget probe's K=20
    lower-bound (which already cost +0.200 vs K=40 at TSP-20). Can the
    early lr-drop compensate for the search-budget starvation, or is K=10
    structurally below the threshold where AGZ policy improvement works
    regardless of optimizer tuning?

    Wall estimate: TSP-20 K=10 ~1 min/iter (Track A scales with K). 100
    iters ~= 1.5-2h on A10. Cost: ~$3-5 Modal credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    run_name = f"tsp20_k10_lv0_step50_100iter_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=10,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="0.2",                  # 5e-4 -> 1e-4 at the step
        lr_decay_step_size=50,           # step at iter 50
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    args.extend(["--mcts_batch_size", "1000"])

    print(f"[modal] launching TSP-20 K=10 lv0 100-iter with lr step at iter 50 (5e-4 -> 1e-4)")
    print(f"  {run_name}")
    print(f"  graph_size=20  K=10  mcts_batch_size=1000")
    print(f"  lr schedule: 5e-4 for iter 0..49, 1e-4 for iter 50..99")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp20_k10_mix_step50(
    mix_lambda: str = "0.5",
    timestamp: str = "",
) -> None:
    """Stage 5 §H — TSP-20 K=10 mix(λ) 100-iter with lr step at iter 50.

    Recipe: B.4 K=10 lv0 step50 verbatim except:
      - leaf_eval: rollout -> mix
      - lambda_v:  0.0 -> 1.0   (value head trained again; lv0 -> lv1)
      - --mix_lambda: caller-supplied λ ∈ [0,1] (default 0.5)

    All other knobs locked to the B.4 winner: K=10, step10, ε=0.25,
    value_target_norm=none, wd=0, buffer=5000, batch=512, train_steps=200,
    M=1000, val_seed=42, gate=ttest gate_every=1, mcts_batch_size=1000.
    lr: 5e-4 for iter 0..49, 1e-4 for iter 50..99 (one step at iter 50).

    Phase B of [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md).
    Apples-to-apples baseline: B.4 reached 3.8576 at iter 100 / 33 min on A10.

    Wall estimate: same K=10 lv0 footprint plus one value-head MLP call per
    leaf (~free since glimpse is already computed for priors). Predict
    ~1.5-2 h on A10. Cost: ~$5-10 Modal credits per λ.

    Usage:
      modal run src/scripts/modal_run_train_alphazero.py::run_tsp20_k10_mix_step50 \\
          --mix-lambda 0.5
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    lam_tag = mix_lambda.replace(".", "p")
    run_name = f"tsp20_k10_mix{lam_tag}_step50_100iter_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=10,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="0.2",                  # 5e-4 -> 1e-4 at the step
        lr_decay_step_size=50,           # step at iter 50
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="mix",
        lambda_v="1.0",                  # value head trained (lv1), unlike B.4's lv0
    )
    args.extend(["--mix_lambda", mix_lambda, "--mcts_batch_size", "1000"])

    print(f"[modal] launching TSP-20 K=10 mix(λ={mix_lambda}) 100-iter with lr step at iter 50")
    print(f"  {run_name}")
    print(f"  graph_size=20  K=10  mix_lambda={mix_lambda}  mcts_batch_size=1000")
    print(f"  lr schedule: 5e-4 for iter 0..49, 1e-4 for iter 50..99")
    print(f"  baseline: B.4 lv0 step50 reached 3.8576 at iter 100 / 33 min")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_tsp50_k25_lv0_step50(timestamp: str = "") -> None:
    """TSP-50 K=25 lv0 100-iter with built-in lr step at iter 50.

    Schedule (one --lr_decay 0.2 --lr_decay_step_size 50 cycle):
        iter  0..49: lr = 5e-4
        iter 50..99: lr = 1e-4   (decay factor 0.2)

    Recipe: lv0 winner verbatim (leaf_eval=rollout, lambda_v=0, step10,
    eps=0.25, value_target_norm=none, wd=0, buffer=5000, batch=512,
    train_steps=200, M=1000, val_seed=42, gate=ttest gate_every=1,
    mcts_batch_size=1000). Graph size 50 and K=25 are the non-default knobs.

    Open question: K=25 sits halfway between the K=50 lv0 chain (which reached
    val 5.93 over 200 iters with manual lr-staircase) and the K=10 TSP-20
    sibling. Does an under-search K=25 budget paired with an early lr-drop
    learn meaningfully on TSP-50, or does the smaller K starve the
    policy-improvement signal? Cross-graph comparison with the K=10 TSP-20
    step50 experiment isolates the search-budget effect across N.

    Wall estimate: TSP-50 K=50 averaged ~4.5 min/iter (Track A); K=25 should
    be ~2-3 min/iter -> 100 iters ~= 4-5h on A10. Cost: ~$10-15 Modal credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    run_name = f"tsp50_k25_lv0_step50_100iter_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=25,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="0.2",                  # 5e-4 -> 1e-4 at the step
        lr_decay_step_size=50,           # step at iter 50
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        leaf_eval="rollout",
        lambda_v="0.0",
    )
    idx_g = args.index("--graph_size")
    args[idx_g + 1] = "50"
    args.extend(["--mcts_batch_size", "1000"])

    print(f"[modal] launching TSP-50 K=25 lv0 100-iter with lr step at iter 50 (5e-4 -> 1e-4)")
    print(f"  {run_name}")
    print(f"  graph_size=50  K=25  mcts_batch_size=1000")
    print(f"  lr schedule: 5e-4 for iter 0..49, 1e-4 for iter 50..99")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f616_400iter_step_decay(timestamp: str = "") -> None:
    """Phase F.6.1.6 — from-scratch 400-iter run with stepped lr decay.

    Consolidates the manual chain F.6.1.3 -> F.6.1.4 -> F.6.1.4.b -> F.6.1.4.c
    (which hit val=3.8498 at iter 225 with lr-staircase 5e-4 -> 1e-4) into
    a single from-scratch run with a built-in lr step schedule:

        iter   0..99:   lr = 5e-4
        iter 100..199:  lr = 1e-4   (decay factor 0.2)
        iter 200..299:  lr = 2e-5
        iter 300..399:  lr = 4e-6

    Achieved via --lr_model 5e-4 --lr_decay 0.2 --lr_decay_step_size 100
    (new flag, validated by local smoke test).

    All other settings = F.6.1.3 eps=0.25 verbatim. mcts_batch_size=1000
    (the new default after the chunk-size sweep). 400 iters total -> 400K
    instances (4x F.6.1.3 budget).

    Hypothesis: the manual chain showed lr=5e-4 saturates at ~3.866 (iter 127)
    and lr=1e-4 unlocks improvement to ~3.85. Continuing with a smaller lr
    might push further. The step schedule lets us probe this in one
    400-iter run instead of three resume chains, with cleaner attribution.

    Expected wall: ~25-30 s/iter at mcts_batch_size=1000 -> ~3 hours total.
    Cost: ~$8-15 Modal credits.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    run_name = f"f616_400iter_step_decay_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="0.2",                  # 5x reduction at each step
        lr_decay_step_size=100,          # step every 100 iters
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
    )
    idx = args.index("--n_iterations")
    args[idx + 1] = "400"

    print(f"[modal] launching F.6.1.6 from-scratch 400-iter step-decay run")
    print(f"  {run_name}")
    print(f"  lr schedule: 5e-4 (0..99) -> 1e-4 (100..199) -> 2e-5 (200..299) -> 4e-6 (300..399)")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_M2000_bsz2000_probe(timestamp: str = "") -> None:
    """High-M probe: M=2000, mcts_batch_size=2000, buffer=10000 (10 iters).

    Rationale: with mcts_batch_size=1000 saturating at M=1000 (per the chunk-
    size sweep), the GPU may have headroom for 2x larger batches. Doubling M
    AND mcts_batch_size to 2000 tests whether we can double per-iter sample
    throughput at roughly the same wall-clock — i.e., spend the saved wall
    on 2x more fresh data per iter.

    Buffer scaled to 10000 (was 5000) to keep the lifetime sample-per-tuple
    window at 5 iters (10000 / 2000 = 5; same as 5000 / 1000 = 5 in the
    baseline). This isolates the "more fresh data" effect from the "different
    consume/produce ratio" effect.

    All other settings = F.6.1.3 eps=0.25 (step10, K=40, train_steps=200,
    batch=512, lr=5e-4, lr_decay=1.0, weight_decay=0, value_target_norm=none,
    leaf_eval=value_head, gate=ttest gate_every=1, val_seed=42, no --load_path).

    Reference points after 10 iters:
    - mcts_bsz1000 (M=1000): predicted ~25-30s mcts_s/iter, ~20K instances.
    - This run (M=2000): predicted similar mcts_s/iter (GPU not saturated at 1000),
      but 2x throughput → ~40K instances seen in 10 iters.

    Cost: ~$1-2 Modal credits, ~10 min wall single A10.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    run_name = f"M2000_bsz2000_buf10k_10iter_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=10000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
    )
    # Override n_iterations to 10 and M_instances to 2000.
    idx = args.index("--n_iterations")
    args[idx + 1] = "10"
    idx_m = args.index("--M_instances")
    args[idx_m + 1] = "2000"
    args.extend(["--mcts_batch_size", "2000"])

    print(f"[modal] launching M=2000 + mcts_batch_size=2000 + buffer=10000 (10 iter)")
    print(f"  {run_name}")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f64c_eps25_lr1e4_resume50_chain(timestamp: str = "") -> None:
    """Phase F.6.1.4.c — chain another +50 iter at lr=1e-4 from F.6.1.4.b's
    iter-199.pt.

    F.6.1.4.b dropped lr from 5e-4 to 1e-4 and immediately broke the iter-127
    saturation: 10 gate accepts in 35 iters, val_avg_cost 3.8665 -> 3.8514
    (last accept iter 184). Then 15 consecutive rejects iter 185-199 -> looks
    like a new (lower) saturation point under lr=1e-4. F.6.1.4.c tests whether
    another +50 at the same lr=1e-4 produces further improvement (chain
    continues), or stalls (suggests further lr decay needed).

    Same recipe as F.6.1.4.b verbatim (step10, eps=0.25, K=40, M=1000,
    train_steps=200, buffer=5000, batch=512, value_target_norm=none,
    leaf_eval=value_head, gate=ttest gate_every=1, val_seed=42, lr=1e-4,
    wd=0). Iter range 200 -> 249. Output:
    `outputs/tsp_20/f62_step10_eps25_resume50_lr1e4_chain_<ts>_*/`.

    First production run on the new uv-built Modal image (uv_sync replaced
    pip_install in the previous commit).

    Cost: ~$5-7 Modal credits, ~75 min wall-clock single A10.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    resume_from = (
        "outputs/tsp_20/"
        "f62_step10_eps25_resume50_lr1e4_20260507T063714_20260507T063723/iter-199.pt"
    )
    run_name = f"f62_step10_eps25_resume50_lr1e4_chain_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,
        buffer_capacity=5000,
        lr_model="1e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
    )
    idx = args.index("--n_iterations")
    args[idx + 1] = "50"
    args.extend(["--resume_from", resume_from])

    print(f"[modal] launching F.6.1.4.c chain resume (+50 iter at lr=1e-4) from F.6.1.4.b iter-199.pt")
    print(f"  {run_name}")
    print(f"  resume_from={resume_from}")
    print(f"  iter range: 200 -> 249")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.local_entrypoint()
def run_f63_kbracket_step10_eps25(timestamp: str = "") -> None:
    """Phase F.6.1.5 (a.k.a. F.6.3 step10) — F.6.1.3 ε=0.25 recipe with the
    K-step-bracket schedule replacing constant K=40.

    Hypothesis: TSP tour cost is invariant under cyclic rotation, so MCTS at
    t=0 has no optimal-value signal — all 20 starting cities yield the same
    optimal tour. Spending K=40 sims at t=0 is mostly wasted compute. The
    last step (t=N-1) is forced. Mid-tour steps are where decisions matter.

    K schedule (TSP-20, "Interpretation A" — high bucket includes midpoint):
      K(0)        = 5     # rotation-symmetric, just produce ~uniform π_t
      K(1..10)    = 40    # mid-tour, real decisions
      K(11..18)   = 10    # late-tour, narrowing search space
      K(19)       = 1     # forced (one legal action)
    Total = 5 + 10·40 + 8·10 + 1 = 486 sims/instance vs 800 for constant K=40
    (~40% cheaper per iter — not budget-matched; the schedule both
    redistributes AND saves compute).

    All other settings = F.6.1.3 ε=0.25 (step10, M=1000, train_steps=200,
    buffer=5000, batch=512, lr=5e-4 const, wd=0, value_target_norm=none,
    leaf_eval=value_head, gate=ttest gate_every=1, val_seed=42, 100 iter
    from-scratch). Carries the per-loss + value-grad VH/shared telemetry.

    Cost: ~$5-7 Modal credits, ~75 min wall-clock single A10.
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    run_name = f"f63_kbracket_step10_eps25_{timestamp}"
    args = _f61_args(
        run_name=run_name,
        k=40,                          # K_mid — high bucket (1..N/2)
        buffer_capacity=5000,
        lr_model="5e-4",
        lr_decay="1.0",
        temperature_schedule="step10",
        dirichlet_epsilon="0.25",
        n_simulations_schedule="step_bracket",
        n_simulations_first=5,
        n_simulations_late=10,
        n_simulations_last=1,
    )

    print(f"[modal] launching F.6.1.5 K-bracket schedule + step10 + ε=0.25 (100 iter)")
    print(f"  {run_name}")
    print(f"  K schedule: 5, 40 (×10), 10 (×8), 1  →  486 sims/instance/iter")
    h = train_alphazero_remote.spawn(*args)
    print(f"\n[modal] awaiting {run_name} (function_call_id={h.object_id}) ...", flush=True)
    h.get()
    print(f"[modal] {run_name} done.", flush=True)
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,
    env=run_env,
    image=image,
    gpu=DEFAULT_GPU,
)
def probe_mcts_decomp_remote(*probe_args: str) -> str:
    """Run scripts.probe_mcts_decomp on Modal A10 with a checkpoint from the volume.

    Returns the captured stdout so the local entrypoint can echo it back.
    Used by `run_probe_mcts_decomp` below.
    """
    import io
    import contextlib
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

    outputs_vol = Path(VOLUME_PATH) / "outputs"
    outputs_vol.mkdir(parents=True, exist_ok=True)
    outputs_link = Path(PROJECT_DIR) / "outputs"
    if outputs_link.is_dir() and not outputs_link.is_symlink():
        import shutil
        shutil.rmtree(outputs_link)
    elif outputs_link.exists() or outputs_link.is_symlink():
        outputs_link.unlink()
    outputs_link.symlink_to(outputs_vol)

    sys.argv = ["probe_mcts_decomp.py"] + list(probe_args)
    from scripts.probe_mcts_decomp import main as probe_main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        probe_main()
    out = buf.getvalue()
    # Also dump to stderr so it appears in Modal logs live.
    print(out, file=sys.stderr, flush=True)
    return out


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 5,
    env=run_env,
    image=image,
    gpu=DEFAULT_GPU,
)
def probe_triton_diag_remote() -> str:
    """Diagnostic: report PyTorch + Triton install status on the Modal A10 image."""
    import io, contextlib, sys
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import torch
        print(f"torch.__version__ = {torch.__version__}")
        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"torch.version.cuda = {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
            print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")
        try:
            import triton
            print(f"triton.__version__ = {triton.__version__}")
            print(f"triton.__file__ = {triton.__file__}")
        except Exception as exc:
            print(f"triton import failed: {exc!r}")
        try:
            import torch._inductor.utils as _iu
            print(f"torch._inductor.utils.has_triton() = {_iu.has_triton()}")
        except Exception as exc:
            print(f"has_triton probe failed: {exc!r}")
        # Try compiling a trivial function to see if the full pipeline works.
        try:
            @torch.compile(dynamic=True)
            def f(x):
                return x.sin().cos()
            y = f(torch.randn(8, device="cuda"))
            print(f"trivial compile OK: y.sum()={y.sum().item():.4f}")
        except Exception as exc:
            print(f"trivial compile failed: {type(exc).__name__}: {exc}")
    return buf.getvalue()


@app.local_entrypoint()
def run_probe_triton_diag() -> None:
    h = probe_triton_diag_remote.spawn()
    print(f"[modal] awaiting triton diag (id={h.object_id})", flush=True)
    out = h.get()
    print("\n========== triton diagnostic ==========")
    print(out)


@app.local_entrypoint()
def run_probe_mcts_decomp(
    load_path: str = (
        "outputs/tsp_50/"
        "tsp50_lv0_K50_resume50_to99_trackA_20260513T064031_20260513T064039/"
        "iter-68_accepted.pt"
    ),
    m_instances: str = "1000",
    n_simulations: str = "50",
    mcts_batch_size: str = "1000",
    leaf_eval: str = "rollout",
    ckpt_key: str = "best_model",
    cprofile_top: str = "40",
    compile_decoder: str = "false",
    compile_mode: str = "default",
) -> None:
    """Run the MCTS wall decomposition probe on Modal A10 with a trained ckpt.

    Default loads `iter-68_accepted.pt` from the current in-flight TSP-50 run
    `0d48yqys` (val 6.0584 — current best Stage 4 TSP-50 lv0 checkpoint as of
    2026-05-13). Uses production scale M=1000 K=50 leaf_eval=rollout to match
    the live training MCTS wall.

    Compare against the local random-init probe at M=200 to factor out the
    random-init Python-share inflation (F.3 caveat in stage5_progress).
    """
    probe_args = [
        "--graph_size", "50",
        "--n_simulations", str(n_simulations),
        "--M", str(m_instances),
        "--mcts_batch_size", str(mcts_batch_size),
        "--leaf_eval", leaf_eval,
        "--dirichlet_epsilon", "0.25",
        "--c_puct", "0.05",
        "--temperature_schedule", "step10",
        "--device", "cuda",
        "--load_path", load_path,
        "--ckpt_key", ckpt_key,
        "--cprofile_top", str(cprofile_top),
    ]
    if str(compile_decoder).lower() in ("true", "1", "yes"):
        probe_args.extend(["--compile_decoder", "--compile_mode", compile_mode])
    print(f"[modal] launching probe_mcts_decomp on A10")
    print(f"  load_path={load_path}")
    print(f"  M={m_instances}  K={n_simulations}  mcts_batch_size={mcts_batch_size}  leaf_eval={leaf_eval}")
    print(f"  compile_decoder={compile_decoder}  compile_mode={compile_mode}")
    h = probe_mcts_decomp_remote.spawn(*probe_args)
    print(f"\n[modal] awaiting probe (function_call_id={h.object_id}) ...", flush=True)
    result = h.get()
    print("\n========== probe output (from Modal A10) ==========")
    print(result)


@app.local_entrypoint()
def run_all(timestamp: str = "") -> None:
    """Spawn the four F.4 + F.3 jobs in parallel on Modal A10.

    Jobs:
      a1: F.4 plan-default — K=100 rollout step30 ε=0.05, ttest gate, 100 iter
      a2: F.4 v5-best     — K=200 rollout step30 ε=0.05, gate=always (every iter), 100 iter
      b1: F.3 freeze-encoder — K=200, --freeze_encoder, 20 iter
      b2: F.3 lr=1e-5       — K=200, --lr_model 1e-5, 20 iter
    """
    from datetime import datetime, timezone
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    jobs = {
        "a1_F4_plan_default": _f4_args(
            f"f4_a1_plan_default_{timestamp}", k=100, gate_mode="ttest", gate_every=5
        ),
        "a2_F4_v5_best": _f4_args(
            f"f4_a2_v5_best_{timestamp}", k=200, gate_mode="always", gate_every=1
        ),
        "b1_freeze_encoder": _f3_args(
            f"f3_b1_freeze_encoder_{timestamp}", extra=["--freeze_encoder"]
        ),
        "b2_lr_1e-5": _f3_args(
            f"f3_b2_lr_1e-5_{timestamp}", extra=["--lr_model", "1e-5"]
        ),
    }

    print(f"[modal] launching {len(jobs)} parallel jobs (timestamp={timestamp})")
    for label, args in jobs.items():
        print(f"  {label}: {' '.join(args)}")

    handles = {
        label: train_alphazero_remote.spawn(*args)
        for label, args in jobs.items()
    }

    print(f"\n[modal] all {len(handles)} jobs spawned. Awaiting completion...")
    for label, h in handles.items():
        print(f"[modal] awaiting {label} (function_call_id={h.object_id}) ...", flush=True)
        h.get()  # blocks
        print(f"[modal] {label} done.", flush=True)

    print(f"\n[modal] all {len(handles)} jobs complete.")
    print("[modal] download results with: modal volume get am-alphagozero-volume outputs/")
