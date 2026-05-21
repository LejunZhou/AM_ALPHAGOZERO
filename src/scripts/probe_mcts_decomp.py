"""Post-Track A MCTS wall decomposition probe.

Runs a single CppBatchMCTSSolver.solve_batch at TSP-50 K=50 with configurable M
on a random-init model. Instruments:

  * Coarse phase wall (encoder/precompute, outer collect/evaluate/apply loop).
  * C++ side: BatchSearch.collect_requests / apply_results / is_done.
  * Python evaluator side: model.decoder.decode_step total wall (NN forward floor).
  * Cache + batch stats from the solver itself.
  * cProfile top-N (sorted by tottime) for function-level hotspots.

Methodology caveat (from F.3 in stage5_progress): random-init gives high cache
hit rate, biasing the share toward per-row Python loops; trained ckpts shift
mix toward GPU. The probe's *structure* (where time goes) is still informative,
but absolute %s on Python overhead would shrink ~2-3x at a trained ckpt.

Usage:
  conda activate AM_AlphaGoZero
  PYTHONPATH=src python -m scripts.probe_mcts_decomp \
      --graph_size 50 --n_simulations 50 --M 200 --mcts_batch_size 200 \
      --leaf_eval rollout --device cuda
"""
import argparse
import cProfile
import io
import pstats
import time
from contextlib import contextmanager

import numpy as np
import torch

from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel
from am_baseline.search import CppBatchMCTSSolver, MCTSConfig
from am_baseline.search.mcts_cpp import _mcts_cpp


@contextmanager
def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_size", type=int, default=50)
    p.add_argument("--n_simulations", type=int, default=50, help="K per tour step")
    p.add_argument("--M", type=int, default=200, help="batch instances")
    p.add_argument("--mcts_batch_size", type=int, default=200,
                   help="cross-instance chunk size; equals M for one chunk")
    p.add_argument("--leaf_eval", choices=["rollout", "value_head", "mix"], default="rollout")
    p.add_argument("--mix_lambda", type=float, default=0.5,
                   help="Convex blend coefficient when --leaf_eval=mix.")
    p.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    p.add_argument("--c_puct", type=float, default=0.05)
    p.add_argument("--temperature_schedule", default="step10")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cprofile_top", type=int, default=35)
    p.add_argument("--load_path", default=None,
                   help="Path to .pt checkpoint to load. If omitted, uses random-init.")
    p.add_argument("--ckpt_key", default="best_model",
                   choices=["best_model", "model"],
                   help="State-dict key within the .pt to load. Stage 4 checkpoints "
                        "carry both `model` (latest) and `best_model` (last accepted).")
    p.add_argument("--compile_decoder", action="store_true",
                   help="Wrap model.decoder.decode_step with torch.compile(dynamic=True). "
                        "Reduces PyTorch dispatch overhead on the hot path. Adds 5-30s of "
                        "compile time on first call (one warmup forward absorbs it). "
                        "Deterministic w.r.t. baseline; verify via paired-seed cost match.")
    p.add_argument("--compile_mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode. 'default' uses dynamic-shape inductor; "
                        "'reduce-overhead' uses cudagraphs (requires static shapes — "
                        "incompatible with our varying B_g per call); 'max-autotune' is "
                        "aggressive inductor with autotuned kernels.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.load_path:
        # Load trained checkpoint via the project's own loader (resolves Config
        # from sibling args.json so the architecture matches the .pt state).
        import os
        from am_baseline.utils.misc import load_args, torch_load_cpu, _remap_ref_keys
        ckpt_dir = os.path.dirname(args.load_path)
        cfg_args = load_args(os.path.join(ckpt_dir, "args.json"))
        cfg_m = Config()
        cfg_m.graph_size = cfg_args["graph_size"]
        cfg_m.embedding_dim = cfg_args["embedding_dim"]
        cfg_m.hidden_dim = cfg_args["hidden_dim"]
        cfg_m.n_encode_layers = cfg_args["n_encode_layers"]
        cfg_m.tanh_clipping = cfg_args["tanh_clipping"]
        cfg_m.normalization = cfg_args["normalization"]
        model = AttentionModel(cfg_m).to(device)
        load_data = torch_load_cpu(args.load_path)
        # Stage 4 checkpoints store best_model + model; pick by --ckpt_key.
        ref_state = load_data.get(args.ckpt_key) or load_data.get("model") or load_data
        our_keys = set(model.state_dict().keys())
        if not set(ref_state.keys()).issubset(our_keys):
            ref_state = _remap_ref_keys(ref_state)
        has_value_head_weights = any(k.startswith("value_head.") for k in ref_state.keys())
        if not has_value_head_weights and model.value_head is not None:
            model.value_head = None
        model.load_state_dict({**model.state_dict(), **ref_state})
        model.eval()
        # Sanity print one weight norm so we can verify load happened.
        emb_norm = next(model.embedder.parameters()).norm().item()
        print(f"[probe] loaded checkpoint {args.load_path}  ckpt_key={args.ckpt_key}  "
              f"embedder.norm={emb_norm:.3f}  graph_size={cfg_m.graph_size}")
        args.graph_size = cfg_m.graph_size
    else:
        cfg_m = Config()
        cfg_m.graph_size = args.graph_size
        model = AttentionModel(cfg_m).to(device).eval()
        print(f"[probe] random-init model  graph_size={cfg_m.graph_size}")

    # MCTSConfig — mirror lv0 production recipe.
    cfg = MCTSConfig(
        n_simulations=args.n_simulations,
        simulation_batch_size=1,
        c_puct=args.c_puct,
        temperature=0.0,
        temperature_schedule=args.temperature_schedule,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=args.dirichlet_epsilon,
        leaf_eval=args.leaf_eval,
        mix_lambda=args.mix_lambda,
        value_norm="bl",
        seed=args.seed,
    )

    # Inputs.
    inputs = torch.rand(args.M, args.graph_size, 2, device=device)

    # Optional torch.compile on the hot path (T2.1).
    if args.compile_decoder:
        # Diagnostic: report Triton availability before attempting compile.
        try:
            import triton  # noqa: F401
            print(f"[probe] triton version: {triton.__version__}")
        except Exception as exc:
            print(f"[probe] triton import failed: {exc!r}  "
                  f"(inductor backend will fall back to eager)")
        try:
            import torch._inductor.utils as _iu
            print(f"[probe] inductor has_triton: {_iu.has_triton()}")
        except Exception as exc:
            print(f"[probe] inductor has_triton probe failed: {exc!r}")

        print(f"[probe] torch.compile(decoder.decode_step, mode={args.compile_mode!r}, dynamic=True)")
        compile_t0 = time.perf_counter()
        # Wrap as a bound method replacement so the rest of the code path stays
        # unchanged. `dynamic=True` is required because B_g (miss batch size)
        # varies per call.
        model.decoder.decode_step = torch.compile(
            model.decoder.decode_step,
            mode=args.compile_mode,
            dynamic=True,
        )
        print(f"[probe] compile wrap installed in {time.perf_counter() - compile_t0:.2f}s "
              f"(first call will trigger trace+codegen).")

    # Instrument C++ BatchSearch via lightweight wrapper around its methods.
    # We patch the BatchSearch class in _mcts_cpp so every call records wall.
    timings = {
        "collect_requests": 0.0,
        "apply_results": 0.0,
        "is_done": 0.0,
        "engine_results": 0.0,
        "decode_step_forward": 0.0,
        "decode_step_calls": 0,
    }
    counts = {"collect_calls": 0, "apply_calls": 0}

    real_BatchSearch = _mcts_cpp.BatchSearch

    class TimedBatchSearch:
        def __init__(self, *a, **kw):
            self._impl = real_BatchSearch(*a, **kw)

        def collect_requests(self):
            t0 = time.perf_counter()
            r = self._impl.collect_requests()
            timings["collect_requests"] += time.perf_counter() - t0
            counts["collect_calls"] += 1
            return r

        def apply_results(self, results):
            t0 = time.perf_counter()
            r = self._impl.apply_results(results)
            timings["apply_results"] += time.perf_counter() - t0
            counts["apply_calls"] += 1
            return r

        def is_done(self):
            t0 = time.perf_counter()
            r = self._impl.is_done()
            timings["is_done"] += time.perf_counter() - t0
            return r

        def results(self):
            t0 = time.perf_counter()
            r = self._impl.results()
            timings["engine_results"] += time.perf_counter() - t0
            return r

    _mcts_cpp.BatchSearch = TimedBatchSearch

    # Instrument decoder.decode_step — that's the only NN forward used inside MCTS
    # (encoder runs once up front).
    real_decode_step = model.decoder.decode_step

    def timed_decode_step(*a, **kw):
        with cuda_sync(device):
            t0 = time.perf_counter()
            r = real_decode_step(*a, **kw)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings["decode_step_forward"] += time.perf_counter() - t0
        timings["decode_step_calls"] += 1
        return r

    model.decoder.decode_step = timed_decode_step

    solver = CppBatchMCTSSolver(
        model, cfg, device=device, mcts_batch_size=args.mcts_batch_size
    )

    print(f"\n[probe] config: TSP-{args.graph_size}  K={args.n_simulations}  "
          f"M={args.M}  leaf_eval={args.leaf_eval}  device={device}")
    print(f"[probe] running solve_batch with cProfile ...")

    profiler = cProfile.Profile()

    # Warmup encoder (so cold compile doesn't pollute).
    with torch.no_grad():
        with cuda_sync(device):
            _ = model.encode(inputs[:1])

    # If torch.compile is on, do a small solve_batch warmup so the trace +
    # codegen cost doesn't get charged to the main timing window. This warms
    # the dynamic-shape inductor cache with realistic MCTS-side shapes.
    if args.compile_decoder:
        warm_M = min(8, args.M)
        warmup_inputs = inputs[:warm_M]
        warm_solver = CppBatchMCTSSolver(
            model, cfg, device=device, mcts_batch_size=warm_M
        )
        warm_t0 = time.perf_counter()
        with torch.no_grad():
            _ = warm_solver.solve_batch(warmup_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(f"[probe] compile warmup solve_batch(M={warm_M}) took "
              f"{time.perf_counter() - warm_t0:.2f}s (trace+codegen included)")
        # Reset solver's per-instance stats accumulators that warm_solver may
        # have populated — we use a fresh `solver` for the real measurement.
        del warm_solver

    t_enc0 = time.perf_counter()
    profiler.enable()
    costs, tours = solver.solve_batch(inputs)
    profiler.disable()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_wall = time.perf_counter() - t_enc0

    # Restore.
    _mcts_cpp.BatchSearch = real_BatchSearch
    model.decoder.decode_step = real_decode_step

    # Report.
    print(f"\n[probe] total wall: {total_wall:.2f}s")
    print(f"[probe] mean cost: {costs.float().mean().item():.4f}")
    print(f"[probe] batch_eval_calls: {solver.batch_eval_calls}")
    print(f"[probe] batch_eval_rows: {solver.batch_eval_rows}")
    if solver.batch_eval_calls > 0:
        print(f"[probe] avg rows/call: {solver.batch_eval_rows / solver.batch_eval_calls:.1f}")
    print(f"[probe] eval_cache_hits: {solver.eval_cache_hits}")
    print(f"[probe] eval_cache_misses: {solver.eval_cache_misses}")
    total_lookups = solver.eval_cache_hits + solver.eval_cache_misses
    if total_lookups > 0:
        print(f"[probe] cache hit rate: {100 * solver.eval_cache_hits / total_lookups:.1f}%")

    print(f"\n[phase] per-phase wall (s):")
    py_other_loop = total_wall - (
        timings["collect_requests"]
        + timings["apply_results"]
        + timings["decode_step_forward"]
        + timings["is_done"]
        + timings["engine_results"]
    )
    rows = [
        ("decode_step (NN fwd, cuda-synced)", timings["decode_step_forward"], timings["decode_step_calls"]),
        ("BatchSearch.collect_requests (C++)", timings["collect_requests"], counts["collect_calls"]),
        ("BatchSearch.apply_results (C++)", timings["apply_results"], counts["apply_calls"]),
        ("BatchSearch.is_done (C++)", timings["is_done"], None),
        ("BatchSearch.results (C++)", timings["engine_results"], None),
        ("Python remainder (evaluator + cache + numpy)", py_other_loop, None),
    ]
    print(f"  {'phase':<50} {'wall_s':>9} {'calls':>9} {'pct':>6}")
    for name, w, c in rows:
        pct = 100 * w / total_wall if total_wall else 0.0
        cs = f"{c}" if c is not None else "-"
        print(f"  {name:<50} {w:>9.2f} {cs:>9} {pct:>5.1f}%")
    print(f"  {'TOTAL':<50} {total_wall:>9.2f} {'-':>9} {100:>5.1f}%")

    # cProfile top-N by tottime.
    print(f"\n[cprofile] top {args.cprofile_top} by tottime (excluding sub-call time):")
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("tottime").print_stats(args.cprofile_top)
    print(buf.getvalue())


if __name__ == "__main__":
    main()
