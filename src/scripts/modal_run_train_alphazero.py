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
