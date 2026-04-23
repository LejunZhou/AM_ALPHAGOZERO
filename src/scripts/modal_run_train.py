"""
Modal cloud GPU entry point for AM training on TSP.

Usage:
  modal run src/scripts/modal_run_train.py -- --graph_size 20 --wandb_project am-alphagozero
  modal run src/scripts/modal_run_train.py -- --graph_size 20 --batch_size 32 --epoch_size 640 --n_epochs 3  # smoke test

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

# Directories to exclude from upload (ref/ is 180MB of reference code)
IGNORE_PATTERNS = [
    "**/outputs/**",
    "**/ref/**",
    "**/__pycache__/**",
    "**/.git/**",
    "**/*.egg-info/**",
    "**/.claude/**",
]


def load_gitignore_patterns() -> list[str]:
    """Merge built-in ignore patterns with .gitignore entries."""
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


# Build container image with CUDA-enabled PyTorch.
image = (
    modal.Image.debian_slim(python_version="3.11")
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
    )
)

# W&B auth: copy .netrc if it exists.
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )

# Copy project code (excluding ref/, outputs/, etc.).
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)

app = modal.App(APP_NAME)

# Forward WANDB_API_KEY from local environment if set (alternative to .netrc).
run_env = {"PYTHONPATH": f"{PROJECT_DIR}/src"}
_wandb_key = os.environ.get("WANDB_API_KEY", "")
if _wandb_key:
    run_env["WANDB_API_KEY"] = _wandb_key


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 8,
    env=run_env,
    image=image,
    gpu=DEFAULT_GPU,
)
def train_remote(*train_args: str) -> None:
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

    # Symlink outputs/ -> volume mount so checkpoints persist across runs
    outputs_vol = Path(VOLUME_PATH) / "outputs"
    outputs_vol.mkdir(parents=True, exist_ok=True)

    outputs_link = Path(PROJECT_DIR) / "outputs"
    if outputs_link.is_dir() and not outputs_link.is_symlink():
        import shutil
        shutil.rmtree(outputs_link)
    elif outputs_link.exists() or outputs_link.is_symlink():
        outputs_link.unlink()
    outputs_link.symlink_to(outputs_vol)

    # Parse args and run training
    sys.argv = ["train.py"] + list(train_args)
    from am_baseline.config import Config
    opts = Config.from_args()

    from scripts.train import run
    run(opts)

    volume.commit()
