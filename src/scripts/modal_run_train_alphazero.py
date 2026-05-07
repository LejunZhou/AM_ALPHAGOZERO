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


# Build container image: CUDA torch + project deps + C++ toolchain for the
# pybind11 MCTS extension. The `pip install -e .` step compiles
# _mcts_cpp.cp311-linux_x86_64.so against the project sources.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "g++", "make")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "wandb>=0.18.0",
        "pybind11>=2.11",
    )
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
    # copy=True is REQUIRED because we need a build step (pip install -e .)
    # that runs AFTER the project sources are present. Modal's default mount
    # behavior would only attach files at container start, after image build.
    copy=True,
)

# Build the C++ MCTS extension once at image-build time. setup.py already
# branches on compiler type so `unix` flags (-O3 -std=c++17) are picked up.
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
    temperature_schedule: str = "step30",
    dirichlet_epsilon: str = "0.0",
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
    return [
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
        "--leaf_eval", "value_head",
        "--max_grad_norm", "1.0",
        "--value_target_norm", "none",
        "--lr_model", lr_model,
        "--lr_decay", lr_decay,
        "--weight_decay", "0.0",
        "--dirichlet_epsilon", dirichlet_epsilon,
        "--dirichlet_alpha_factor", "10.0",
        "--wandb_project", "am-alphagozero",
        "--wandb_mode", "online",
        "--run_name", run_name,
    ]


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
