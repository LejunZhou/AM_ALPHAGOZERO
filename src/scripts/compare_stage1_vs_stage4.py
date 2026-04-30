"""Apples-to-apples comparison: Stage 1 vs Stage 4 on identical val sets.

The F.3 pilot acceptance threshold (3.83843) is derived from Stage 1's
canonical val_avg_cost (3.83943) measured on one specific 10K val set seed.
Stage 4 runs construct their val sets fresh, with different sampling, so
direct comparison of Stage 4's val_avg_cost to 3.83843 is confounded by
val-set sampling variance (~±0.003 across 10K random draws).

This script evaluates BOTH models on the SAME (large, fixed-seed) val set
and reports the difference, eliminating that confounder.
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.utils.misc import torch_load_cpu


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_ckpt', required=True)
    p.add_argument('--stage4_ckpt', required=True)
    p.add_argument('--graph_size', type=int, default=20)
    p.add_argument('--num_test', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_cuda', action='store_true')
    return p.parse_args()


def build_model():
    class Cfg:
        embedding_dim = 128; hidden_dim = 128; n_encode_layers = 3; n_heads = 8
        tanh_clipping = 10.0; normalization = 'batch'; feed_forward_hidden = 512
        value_enabled = True; value_hidden_dim = 128; value_target_norm = 'bl'
    return AttentionModel(Cfg())


def greedy_eval(model, coords, device, batch_size=2048):
    model.eval(); model.set_decode_type('greedy')
    out = []
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i+batch_size].to(device)
            costs, _ = model(chunk)
            out.append(costs.cpu().numpy())
    return np.concatenate(out)


def main():
    opts = parse()
    device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    print(f'device = {device}')

    # Build Stage 1 model
    s1 = build_model()
    s1.load_state_dict(torch_load_cpu(opts.stage1_ckpt)['model'])
    s1 = s1.to(device)

    # Build Stage 4 model. Stage 4 checkpoint includes both 'model' (working/candidate)
    # and 'best_model' (theta_star). Compare both.
    s4_ckpt = torch_load_cpu(opts.stage4_ckpt)
    s4_keys = list(s4_ckpt.keys())
    print(f'Stage 4 ckpt keys: {s4_keys}')
    s4_model = build_model()
    s4_model.load_state_dict(s4_ckpt['model']); s4_model = s4_model.to(device)
    s4_best = build_model()
    if 'best_model' in s4_ckpt:
        s4_best.load_state_dict(s4_ckpt['best_model']); s4_best = s4_best.to(device)
    else:
        print('  (no best_model in ckpt; using working model only)')
        s4_best = None

    # Make a deterministic val set
    torch.manual_seed(opts.seed); np.random.seed(opts.seed)
    insts = TSP.make_dataset(size=opts.graph_size, num_samples=opts.num_test)
    coords = torch.stack([x for x in insts])
    print(f'val set: N={opts.graph_size}, num={opts.num_test}, seed={opts.seed}')

    # Evaluate
    s1_costs = greedy_eval(s1, coords, device)
    s4_model_costs = greedy_eval(s4_model, coords, device)
    s4_best_costs = greedy_eval(s4_best, coords, device) if s4_best is not None else None

    print()
    print(f"  Stage 1 (loaded):     mean={s1_costs.mean():.5f} std={s1_costs.std()/np.sqrt(len(s1_costs)):.5f}")
    print(f"  Stage 4 (working):    mean={s4_model_costs.mean():.5f} std={s4_model_costs.std()/np.sqrt(len(s4_model_costs)):.5f}")
    if s4_best_costs is not None:
        print(f"  Stage 4 (best_model): mean={s4_best_costs.mean():.5f} std={s4_best_costs.std()/np.sqrt(len(s4_best_costs)):.5f}")

    # Paired diff: Stage 4 working - Stage 1
    d = s4_model_costs - s1_costs
    print()
    print(f"  Paired diff (S4_working - S1):  mean={d.mean():+.5f}, SE={d.std()/np.sqrt(len(d)):.5f}")
    print(f"    n with S4 strictly better:  {(d < 0).sum()}/{len(d)} ({(d<0).mean()*100:.1f}%)")
    print(f"    n with S4 strictly worse:   {(d > 0).sum()}/{len(d)} ({(d>0).mean()*100:.1f}%)")
    print(f"    n equal:                    {(d == 0).sum()}/{len(d)} ({(d==0).mean()*100:.1f}%)")
    # One-sided paired t-test: H1 = S4 < S1 (improvement)
    from scipy import stats
    t, p_two = stats.ttest_rel(s4_model_costs, s1_costs)
    p_one = p_two / 2 if t < 0 else 1 - p_two / 2
    print(f"    paired t-test t={t:.3f}, p_one_sided(S4<S1)={p_one:.4f}")


if __name__ == '__main__':
    main()
