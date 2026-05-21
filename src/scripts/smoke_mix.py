"""Stage 5 §H mix leaf-eval smoke harness.

Verifies the new `leaf_eval='mix'` mode is wired correctly across Python and
C++ backends:

  - **M1** Python parity at λ=0: leaf_eval='rollout' vs leaf_eval='mix' with
           mix_lambda=0.0 must produce identical tours and costs (the
           value-head contribution is scaled to zero).
  - **M2** Python parity at λ=1: leaf_eval='value_head' vs leaf_eval='mix' with
           mix_lambda=1.0 must match exactly (the rollout contribution is
           scaled to zero).
  - **M3** C++↔Python bit-equivalence at λ=0.5: the sequential C++ Solver must
           produce the same cost as the canonical Python MCTSSolver at
           leaf_eval='mix', mix_lambda=0.5.

The model is a small random-init AttentionModel with value head enabled. The
absolute cost values are arbitrary — only the cross-config equivalence matters.

Run:
    PYTHONPATH=src python -m scripts.smoke_mix
or:
    python src/scripts/smoke_mix.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.search.mcts import MCTSConfig, MCTSSolver


GRAPH_SIZE = 10
N_SIMULATIONS = 16
MODEL_SEED = 4242
INSTANCE_SEED = 7
MCTS_SEED = 13


def _build_model_and_instance() -> tuple[AttentionModel, torch.Tensor]:
    """Random-init AM with value head + one fixed TSP-N instance on CPU."""
    torch.manual_seed(MODEL_SEED)
    cfg = Config(
        graph_size=GRAPH_SIZE,
        embedding_dim=32,
        n_encode_layers=2,
        n_heads=4,
        value_enabled=True,
        value_hidden_dim=32,
    )
    model = AttentionModel(cfg).cpu().eval()

    rng = np.random.default_rng(INSTANCE_SEED)
    coords = torch.from_numpy(rng.random((1, GRAPH_SIZE, 2), dtype=np.float64)).float()
    return model, coords


def _mcts_cfg(leaf_eval: str, mix_lambda: float = 0.5) -> MCTSConfig:
    """A common MCTSConfig — varies only leaf_eval and mix_lambda."""
    return MCTSConfig(
        n_simulations=N_SIMULATIONS,
        c_puct=1.0,
        leaf_eval=leaf_eval,
        mix_lambda=mix_lambda,
        value_target_norm='none',          # raw cost-to-go target convention
        dirichlet_epsilon=0.0,
        temperature=0.0,
        seed=MCTS_SEED,
    )


def _solve_py(model: AttentionModel, coords: torch.Tensor, cfg: MCTSConfig) -> float:
    solver = MCTSSolver(model, cfg, device=torch.device('cpu'))
    costs, _tours = solver.solve_batch(coords)
    return float(costs.view(-1)[0].item())


def _solve_cpp(model: AttentionModel, coords: torch.Tensor, cfg: MCTSConfig) -> float:
    from am_baseline.search.mcts_cpp.solver import CppMCTSSolver, HAVE_CPP_MCTS
    if not HAVE_CPP_MCTS:
        raise RuntimeError("C++ MCTS extension is not available; rebuild via `pip install -e .`")
    solver = CppMCTSSolver(model, cfg, device=torch.device('cpu'))
    costs, _tours = solver.solve_batch(coords)
    return float(costs.view(-1)[0].item())


def _assert_close(name: str, a: float, b: float, atol: float = 0.0, rtol: float = 0.0) -> None:
    diff = abs(a - b)
    threshold = atol + rtol * max(abs(a), abs(b))
    if diff > threshold:
        raise AssertionError(
            f"[{name}] cost mismatch: a={a!r}, b={b!r}, |Δ|={diff:.3e} (tol={threshold:.3e})"
        )
    print(f"  [{name}] OK  a={a:.10f}  b={b:.10f}  |Δ|={diff:.3e}")


def m1_lambda_zero_parity(model: AttentionModel, coords: torch.Tensor) -> None:
    print("M1: leaf_eval='rollout' vs leaf_eval='mix' (mix_lambda=0.0) — Python")
    cost_rollout = _solve_py(model, coords, _mcts_cfg('rollout'))
    cost_mix0 = _solve_py(model, coords, _mcts_cfg('mix', mix_lambda=0.0))
    _assert_close("M1", cost_rollout, cost_mix0)


def m2_lambda_one_parity(model: AttentionModel, coords: torch.Tensor) -> None:
    print("M2: leaf_eval='value_head' vs leaf_eval='mix' (mix_lambda=1.0) — Python")
    cost_vh = _solve_py(model, coords, _mcts_cfg('value_head'))
    cost_mix1 = _solve_py(model, coords, _mcts_cfg('mix', mix_lambda=1.0))
    _assert_close("M2", cost_vh, cost_mix1)


def m3_cpp_python_parity_half(model: AttentionModel, coords: torch.Tensor) -> None:
    print("M3: Python MCTSSolver vs sequential C++ Solver at leaf_eval='mix', mix_lambda=0.5")
    cfg = _mcts_cfg('mix', mix_lambda=0.5)
    cost_py = _solve_py(model, coords, cfg)
    cost_cpp = _solve_cpp(model, coords, cfg)
    # C++ ↔ Python parity historically rides on identical fp ordering. Allow
    # a tight tolerance for any double->float decode_step round-tripping.
    _assert_close("M3", cost_py, cost_cpp, atol=1e-9, rtol=1e-9)


def main() -> None:
    print(f"[smoke_mix] GRAPH_SIZE={GRAPH_SIZE}  N_SIMULATIONS={N_SIMULATIONS}")
    model, coords = _build_model_and_instance()
    m1_lambda_zero_parity(model, coords)
    m2_lambda_one_parity(model, coords)
    m3_cpp_python_parity_half(model, coords)
    print("[smoke_mix] all checks passed")


if __name__ == "__main__":
    main()
