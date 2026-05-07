"""Validate a Stage 4 checkpoint with MCTS on a fixed-seed val set.

Loads a Stage 4 `iter-{i}.pt` (or `iter-{i}_accepted.pt`) and evaluates the
chosen model (θ★ / `best_model`, the working model, or both) under MCTS, and
reports tour cost vs the greedy θ★ baseline (the `bl_val` convention used
during self-play). Optionally also evaluates an AM-baseline checkpoint
(Stage 1 canonical or the reference pretrained from
`ref/attention-learn-to-route-master/pretrained/`) under greedy + sampling
decoding, on the SAME val set, for paired apples-to-apples comparison.

Mirrors the probe in `probe_mcts_quality.py` but is specialised for Stage 4
checkpoints that carry both `model` and `best_model` state dicts and a
sibling `args.json` describing the training architecture.

Usage examples:
    # Default: best_model, K=50 rollout, ε=0, no temperature schedule.
    python src/scripts/val_stage4_mcts.py \
        --ckpt outputs/tsp_20/stage4_xxx/iter-99.pt --num_test 500

    # Match training-time MCTS exactly (read K / leaf_eval / ε / tsched from args.json).
    python src/scripts/val_stage4_mcts.py \
        --ckpt outputs/tsp_20/stage4_xxx/iter-99.pt --match_train

    # Compare both heads on a custom MCTS config.
    python src/scripts/val_stage4_mcts.py --ckpt ... --which both \
        --K 100 --leaf_eval value_head --eps 0.0

    # Add AM-baseline (Stage 1 canonical) greedy + sampling x1280 on same val set.
    python src/scripts/val_stage4_mcts.py --ckpt ... --num_test 500 \
        --am_ckpt outputs/tsp_20/stage1_tsp20_canonical_*/epoch-99.pt
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.search.mcts import MCTSConfig
from am_baseline.search.mcts_cpp.solver import CppBatchMCTSSolver
from am_baseline.utils.misc import torch_load_cpu


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse():
    p = argparse.ArgumentParser(
        description="Validate a Stage 4 checkpoint with MCTS on a fixed val set.",
    )
    p.add_argument('--ckpt', required=True,
                   help='Path to a Stage 4 iter-{i}.pt (or iter-{i}_accepted.pt). '
                        'Must contain "model" and/or "best_model" state dicts.')
    p.add_argument('--which', choices=['best', 'working', 'both'], default='best',
                   help='Which head to evaluate. "best" = best_model (θ★, the head '
                        'that runs MCTS during self-play; default). "working" = '
                        'candidate model. "both" = run both and report side-by-side.')

    # Val set.
    p.add_argument('--graph_size', type=int, default=None,
                   help='TSP graph size N. Read from sibling args.json if omitted.')
    p.add_argument('--num_test', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260430,
                   help='Val set seed. Default matches probe_mcts_quality.py for '
                        'cross-script comparability.')

    # MCTS config — defaults are post-F.6 sane (rollout, no exploration noise,
    # constant τ) for clean eval. Use --match_train to override from args.json.
    p.add_argument('--K', type=int, default=50,
                   help='MCTS simulations per root.')
    p.add_argument('--leaf_eval', choices=['value_head', 'rollout'], default='rollout',
                   help='MCTS leaf evaluator. Eval default = rollout (more reliable '
                        'on warm-started policies, see probe_mcts_quality.py).')
    p.add_argument('--eps', type=float, default=0.0,
                   help='Dirichlet ε root noise. Default 0 (deterministic eval).')
    p.add_argument('--alpha_factor', type=float, default=10.0,
                   help='Dirichlet α scale: α = factor / N (AGZ default 10/N).')
    p.add_argument('--temperature_schedule', choices=['const', 'step30', 'step50'],
                   default='const',
                   help='Per-tour-step τ schedule. Default const (eval = greedy '
                        'argmax over visit counts; see c_puct discussion in spec).')
    p.add_argument('--c_puct', type=float, default=0.05)
    p.add_argument('--mcts_batch_size', type=int, default=64)
    p.add_argument('--match_train', action='store_true',
                   help='Override --K / --leaf_eval / --eps / --alpha_factor / '
                        '--temperature_schedule with values read from sibling '
                        'args.json (n_simulations_train, leaf_eval, '
                        'dirichlet_epsilon, dirichlet_alpha_factor, '
                        'temperature_schedule). Useful for reproducing the exact '
                        'MCTS quality the coach saw at training time.')

    # AM baseline (Stage 1 canonical or reference pretrained).
    p.add_argument('--am_ckpt', type=str, default=None,
                   help='Optional AM checkpoint to evaluate as a baseline (greedy + '
                        'sampling) on the SAME val set. Accepts either a Stage 1 '
                        'canonical checkpoint (`{"model": state_dict, ...}` format) '
                        'or the reference release '
                        '`ref/attention-learn-to-route-master/pretrained/tsp_N/epoch-99.pt` '
                        '(auto-detects format and applies the test_pretrained.py key remap).')
    p.add_argument('--am_label', type=str, default='AM',
                   help='Label printed in the report block for the AM baseline.')
    p.add_argument('--am_sample_width', type=int, default=1280,
                   help='Sampling decode width for the AM baseline (paper default 1280).')
    p.add_argument('--am_sample_batch_rep', type=int, default=128,
                   help='Per-iter sample replication factor (mirrors evaluate.py).')
    p.add_argument('--am_sample_outer_batch', type=int, default=64,
                   help='Outer batch size for the sampling loop. Total in-flight '
                        'instances = am_sample_outer_batch × am_sample_batch_rep '
                        '(64 × 128 = 8192 by default — safe on consumer GPUs).')
    p.add_argument('--no_am_sample', action='store_true',
                   help='Skip AM sampling (greedy only). Saves wall-clock when only '
                        'the greedy AM baseline is interesting.')

    # Runtime.
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--no_greedy', action='store_true',
                   help='Skip the greedy θ★ baseline evaluation (saves a forward pass).')
    p.add_argument('--batch_size', type=int, default=2048,
                   help='Greedy-eval batch size (MCTS uses --mcts_batch_size).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _read_args_json(ckpt_path):
    """Look for `args.json` next to ckpt_path. Returns dict or None."""
    args_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'args.json')
    if not os.path.exists(args_path):
        return None
    try:
        with open(args_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f'  [warn] failed to read {args_path}: {e}')
        return None


def _build_cfg(train_args):
    """Build the AttentionModel config namespace, preferring train_args fields
    where present. Falls back to canonical Stage 1 defaults otherwise."""
    class Cfg:
        embedding_dim = 128
        hidden_dim = 128
        n_encode_layers = 3
        n_heads = 8
        tanh_clipping = 10.0
        normalization = 'batch'
        feed_forward_hidden = 512
        value_enabled = True
        value_hidden_dim = 128
        value_target_norm = 'bl'
    if train_args is not None:
        for k in ('embedding_dim', 'hidden_dim', 'n_encode_layers', 'n_heads',
                  'tanh_clipping', 'normalization', 'feed_forward_hidden',
                  'value_enabled', 'value_hidden_dim', 'value_target_norm'):
            if k in train_args:
                setattr(Cfg, k, train_args[k])
    return Cfg


def load_model(ckpt, key, train_args, device):
    """Build an AttentionModel and load `ckpt[key]` into it."""
    Cfg = _build_cfg(train_args)
    model = AttentionModel(Cfg())
    if key not in ckpt:
        raise KeyError(
            f"Checkpoint has no '{key}' state dict. Available keys: {list(ckpt.keys())}"
        )
    model.load_state_dict(ckpt[key])
    return model.to(device)


# ---- AM-baseline loader -----------------------------------------------------
#
# Two on-disk shapes are accepted:
#   (a) Stage 1 canonical: `{'model': state_dict, ...}` with our key layout
#       (init_embed.*, embedder.*, decoder.*, value_head.*).
#   (b) Reference release `ref/.../tsp_N/epoch-99.pt`: a bare state dict (or
#       `{'model': bare_state_dict}`) whose decoder projections live at the
#       AttentionModel root (`project_node_embeddings.*`, `W_placeholder`, ...);
#       the existing test_pretrained.py:build_key_mapping() handles the rename.


def _ref_key_mapping():
    """Mirror of test_pretrained.py:build_key_mapping(). Maps reference-release
    parameter names to our model's parameter names. Encoder keys (embedder.*)
    and init_embed.* pass through unchanged."""
    mapping = {
        'init_embed.weight': 'init_embed.weight',
        'init_embed.bias': 'init_embed.bias',
        'W_placeholder': 'decoder.W_placeholder',
    }
    for p in ('project_node_embeddings.weight',
              'project_fixed_context.weight',
              'project_step_context.weight',
              'project_out.weight'):
        mapping[p] = 'decoder.' + p
    return mapping


def _looks_like_ref_state(state_dict):
    """A reference-release state dict has the decoder projections at the
    root; ours has them under `decoder.`. Use root-level
    `project_node_embeddings.weight` as the discriminant."""
    return 'project_node_embeddings.weight' in state_dict


def load_am_model(ckpt_path, device):
    """Load an AM checkpoint (Stage 1 canonical or reference pretrained) and
    return the model on `device`. Auto-detects which on-disk shape the file
    uses and applies the rename when needed.

    Builds with `value_enabled=False` because greedy + sampling decode does
    not consult the value head — and the reference release was trained without
    one. Stage 1 canonical's value-head keys, if present, are silently dropped."""
    ref_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ref_state = ref_data.get('model', ref_data) if isinstance(ref_data, dict) else ref_data

    # Build with no value head — both greedy and sampling decode only use
    # encoder + decoder.
    class Cfg:
        embedding_dim = 128
        hidden_dim = 128
        n_encode_layers = 3
        n_heads = 8
        tanh_clipping = 10.0
        normalization = 'batch'
        feed_forward_hidden = 512
        value_enabled = False
        value_hidden_dim = 128
        value_target_norm = 'bl'
    model = AttentionModel(Cfg())
    our_state = model.state_dict()

    if _looks_like_ref_state(ref_state):
        mapping = _ref_key_mapping()
        mapped = {}
        for ref_key, ref_val in ref_state.items():
            if ref_key in mapping:
                our_key = mapping[ref_key]
            elif ref_key.startswith('embedder.'):
                our_key = ref_key  # encoder passes through
            else:
                continue  # skip anything we don't know how to map
            if our_key in our_state and our_state[our_key].shape == ref_val.shape:
                mapped[our_key] = ref_val
        our_state.update(mapped)
        model.load_state_dict(our_state)
        print(f'  [am] loaded reference release: {len(mapped)}/{len(our_state)} params mapped')
    else:
        # Stage 1 canonical — keys already match. value_head.* keys (if present)
        # are extras that load_state_dict(strict=False) will ignore.
        kept = {k: v for k, v in ref_state.items() if k in our_state}
        our_state.update(kept)
        model.load_state_dict(our_state, strict=False)
        n_extra = len(ref_state) - len(kept)
        print(f'  [am] loaded canonical: {len(kept)}/{len(our_state)} params mapped'
              f'{f" ({n_extra} extra ckpt keys ignored, e.g. value_head)" if n_extra else ""}')

    return model.to(device)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def greedy_eval(model, coords, device, batch_size=2048):
    model.eval()
    model.set_decode_type('greedy')
    out = []
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size].to(device)
            costs, _ = model(chunk)
            out.append(costs.cpu().numpy())
    return np.concatenate(out)


def mcts_eval(model, coords, cfg, device, mcts_batch_size=64):
    solver = CppBatchMCTSSolver(model, cfg, device, mcts_batch_size=mcts_batch_size)
    costs, _ = solver.solve_batch(coords.to(device))
    return costs.cpu().numpy()


def sample_eval(model, coords, device, width=1280, batch_rep=128, outer_batch=64):
    """Sampling decode: for each instance, draw `width` tours and keep the min.

    Mirrors the convention in `evaluate.py`:
        batch_rep = min(width, batch_rep)        # per-iter sample fan-out
        iter_rep  = ceil(width / batch_rep)      # number of inner iterations
    Outer batching keeps GPU memory bounded — total in-flight per outer chunk
    is `outer_batch × batch_rep`.
    """
    model.eval()
    model.set_decode_type('sampling')
    br = min(width, batch_rep)
    ir = (width + br - 1) // br
    out = []
    with torch.no_grad():
        for i in range(0, len(coords), outer_batch):
            chunk = coords[i:i + outer_batch].to(device)
            _, best_cost = model.sample_many(chunk, batch_rep=br, iter_rep=ir)
            out.append(best_cost.cpu().numpy())
    return np.concatenate(out)


def _build_mcts_config(opts, graph_size, train_args):
    """Build an MCTSConfig honoring --match_train overrides."""
    K = opts.K
    leaf_eval = opts.leaf_eval
    eps = opts.eps
    alpha_factor = opts.alpha_factor
    tsched = opts.temperature_schedule

    if opts.match_train:
        if train_args is None:
            raise FileNotFoundError(
                '--match_train requires a sibling args.json next to --ckpt.'
            )
        K = int(train_args.get('n_simulations_train', K))
        leaf_eval = str(train_args.get('leaf_eval', leaf_eval))
        eps = float(train_args.get('dirichlet_epsilon', eps))
        alpha_factor = float(train_args.get('dirichlet_alpha_factor', alpha_factor))
        tsched = str(train_args.get('temperature_schedule', tsched))

    # τ schedule string `const` is encoded as None at the C++ boundary (see
    # CppMCTSSolver._SCHEDULE_TO_INT). MCTSConfig accepts None, 'const', or
    # 'step{30,50}'.
    tsched_arg = None if tsched == 'const' else tsched
    # When there's no schedule, τ=0 picks argmax-of-visits — the typical
    # eval convention. probe_mcts_quality.py uses τ=1 only when a schedule
    # is set; same here.
    temperature = 1.0 if tsched_arg is not None else 0.0

    cfg = MCTSConfig(
        n_simulations=K,
        leaf_eval=leaf_eval,
        value_norm='bl',
        c_puct=opts.c_puct,
        temperature=temperature,
        temperature_schedule=tsched_arg,
        dirichlet_alpha=alpha_factor / graph_size,
        dirichlet_epsilon=eps,
        fpu_mode='running_q',
        fpu_fallback=-1.0,
        root_select='visits',
        tree_reuse=True,
        return_root_visits=False,
        seed=opts.seed,
    )
    return cfg, dict(K=K, leaf_eval=leaf_eval, eps=eps,
                     alpha_factor=alpha_factor, tsched=tsched)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report_block(label, costs, baseline=None):
    """Print mean / SE for a cost vector, plus paired diff vs `baseline` if given."""
    se = costs.std() / np.sqrt(len(costs))
    print(f'  {label:30s} mean={costs.mean():.5f}  SE={se:.5f}')
    if baseline is None:
        return
    d = costs - baseline
    print(f'    paired diff vs greedy θ★:    mean={d.mean():+.5f}  SE={d.std()/np.sqrt(len(d)):.5f}')
    print(f'    n strictly better (MCTS<gd): {(d < 0).sum()}/{len(d)} ({(d < 0).mean() * 100:.1f}%)')
    print(f'    n strictly worse  (MCTS>gd): {(d > 0).sum()}/{len(d)} ({(d > 0).mean() * 100:.1f}%)')
    print(f'    n equal:                     {(d == 0).sum()}/{len(d)} ({(d == 0).mean() * 100:.1f}%)')
    try:
        from scipy import stats
        t, p_two = stats.ttest_rel(costs, baseline)
        p_one = p_two / 2 if t < 0 else 1 - p_two / 2
        print(f'    paired t-test t={t:.3f}, p_one_sided(MCTS<greedy)={p_one:.4f}')
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    opts = parse()
    device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    print(f'device = {device}')
    print(f'ckpt   = {opts.ckpt}')

    train_args = _read_args_json(opts.ckpt)
    if train_args is not None:
        print(f'  [*] read train args.json (run_name={train_args.get("run_name", "?")})')

    # Resolve graph_size: CLI > args.json > error.
    graph_size = opts.graph_size
    if graph_size is None:
        if train_args is None or 'graph_size' not in train_args:
            raise ValueError(
                'graph_size not provided and no sibling args.json — pass --graph_size.'
            )
        graph_size = int(train_args['graph_size'])
    print(f'  graph_size={graph_size}, num_test={opts.num_test}, seed={opts.seed}')

    # Load checkpoint and resolve which head(s) to evaluate.
    ckpt = torch_load_cpu(opts.ckpt)
    print(f'  ckpt keys: {list(ckpt.keys())}')
    heads = []
    if opts.which in ('best', 'both'):
        heads.append(('best_model', 'best_model (θ★)'))
    if opts.which in ('working', 'both'):
        heads.append(('model', 'model (working)'))
    if not heads:
        raise ValueError(f'unrecognized --which: {opts.which}')

    # Pinned val set — same convention as probe_mcts_quality.py.
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    instances = TSP.make_dataset(size=graph_size, num_samples=opts.num_test)
    coords = torch.stack([x for x in instances])

    # Build MCTS config (single shared config across heads; --match_train uses args.json).
    cfg, cfg_summary = _build_mcts_config(opts, graph_size, train_args)
    print(f"\n[mcts cfg] K={cfg_summary['K']} leaf={cfg_summary['leaf_eval']}  "
          f"eps={cfg_summary['eps']}  alpha={cfg_summary['alpha_factor']}/N  "
          f"tsched={cfg_summary['tsched']}  c_puct={opts.c_puct}")

    # ---- AM baseline (Stage 1 / reference) — same val set ------------------
    am_results = {}  # 'greedy' / 'sample' -> cost array
    if opts.am_ckpt is not None:
        print(f'\n=== {opts.am_label} baseline ({os.path.basename(opts.am_ckpt)}) ===')
        am_model = load_am_model(opts.am_ckpt, device)

        t0 = time.time()
        am_greedy = greedy_eval(am_model, coords, device, batch_size=opts.batch_size)
        am_results['greedy'] = am_greedy
        print(f'  {opts.am_label} greedy:        mean={am_greedy.mean():.5f}  '
              f'SE={am_greedy.std() / np.sqrt(len(am_greedy)):.5f}  '
              f'[{time.time() - t0:.1f}s]')

        if not opts.no_am_sample:
            t0 = time.time()
            am_sample = sample_eval(
                am_model, coords, device,
                width=opts.am_sample_width,
                batch_rep=opts.am_sample_batch_rep,
                outer_batch=opts.am_sample_outer_batch,
            )
            am_results['sample'] = am_sample
            elapsed = time.time() - t0
            print(f'  {opts.am_label} sample(x{opts.am_sample_width}):  '
                  f'mean={am_sample.mean():.5f}  '
                  f'SE={am_sample.std() / np.sqrt(len(am_sample)):.5f}  '
                  f'[{elapsed:.1f}s, {elapsed / len(am_sample) * 1000:.1f} ms/instance]')

        # Free AM model GPU memory before Stage 4 heads load.
        del am_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- Stage 4 head eval --------------------------------------------------
    head_results = {}  # (key, 'greedy' | 'mcts') -> cost array
    for key, label in heads:
        print(f'\n=== {label} ===')
        model = load_model(ckpt, key, train_args, device)

        baseline = None
        if not opts.no_greedy:
            t0 = time.time()
            baseline = greedy_eval(model, coords, device, batch_size=opts.batch_size)
            print(f'  greedy θ★ ({key}):       mean={baseline.mean():.5f}  '
                  f'SE={baseline.std() / np.sqrt(len(baseline)):.5f}  '
                  f'[{time.time() - t0:.1f}s]')
            head_results[(key, 'greedy')] = baseline

        t0 = time.time()
        mcts_costs = mcts_eval(model, coords, cfg, device,
                               mcts_batch_size=opts.mcts_batch_size)
        elapsed = time.time() - t0
        _report_block(f'MCTS K={cfg_summary["K"]} {cfg_summary["leaf_eval"]}',
                      mcts_costs, baseline=baseline)
        print(f'    wall: {elapsed:.1f}s ({elapsed / len(mcts_costs) * 1000:.1f} ms/instance)')
        head_results[(key, 'mcts')] = mcts_costs

    # ---- Paired summary across decoders (only if AM baseline ran) -----------
    if am_results:
        print('\n=== summary (same val set, paired) ===')
        rows = []
        for label, costs in am_results.items():
            tag = f'{opts.am_label} {label}' + (
                f'(x{opts.am_sample_width})' if label == 'sample' else ''
            )
            rows.append((tag, costs))
        for (key, kind), costs in head_results.items():
            tag_kind = 'greedy' if kind == 'greedy' else f'MCTS K={cfg_summary["K"]} {cfg_summary["leaf_eval"]}'
            rows.append((f'Stage4 {key} {tag_kind}', costs))

        # Single-row mean/SE table.
        print(f'  {"decoder":40s}  {"mean":>9s}  {"SE":>8s}')
        print('  ' + '-' * 60)
        for tag, costs in rows:
            print(f'  {tag:40s}  {costs.mean():9.5f}  {costs.std() / np.sqrt(len(costs)):8.5f}')

        # Pairwise paired-diffs vs the AM-greedy reference.
        if 'greedy' in am_results:
            ref = am_results['greedy']
            ref_label = f'{opts.am_label} greedy'
            print(f'\n  paired diffs vs {ref_label}:')
            for tag, costs in rows:
                if tag == ref_label:
                    continue
                d = costs - ref
                print(f'    {tag:40s}  Δ={d.mean():+.5f}  SE={d.std() / np.sqrt(len(d)):.5f}')


if __name__ == '__main__':
    main()
