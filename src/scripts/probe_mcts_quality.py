"""Probe MCTS self-play quality for the warm-started Stage 1 checkpoint.

Question: which (K, leaf_eval, dirichlet_epsilon, temperature_schedule)
combinations produce MCTS tours that BEAT greedy θ★ on a fixed test set?

If a combo's mean tour cost > mean bl_val, the AGZ loop has no improvement
signal to distill from and will degrade the policy (observed in F.3 pilot:
mean(MCTS - greedy) = +0.077 under the canonical F.3 recipe).
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.search.mcts import MCTSConfig
from am_baseline.search.mcts_cpp.solver import CppBatchMCTSSolver
from am_baseline.utils.misc import torch_load_cpu


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--load_path', required=True)
    p.add_argument('--graph_size', type=int, default=20)
    p.add_argument('--num_test', type=int, default=200)
    p.add_argument('--seed', type=int, default=20260430)
    p.add_argument('--no_cuda', action='store_true')
    return p.parse_args()


def build_model(opts, ckpt):
    """Reconstruct the Stage 1 attention model with default arch."""
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
    model = AttentionModel(Cfg())
    model.load_state_dict(ckpt['model'])
    return model


def greedy_eval(model, coords, device):
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        costs, _ = model(coords.to(device))
    return costs.cpu().numpy()


def mcts_eval(model, coords, cfg, device, mcts_batch_size=64):
    solver = CppBatchMCTSSolver(model, cfg, device, mcts_batch_size=mcts_batch_size)
    costs, _ = solver.solve_batch(coords.to(device))
    return costs.cpu().numpy()


def main():
    opts = parse()
    device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    print(f'device = {device}')

    ckpt = torch_load_cpu(opts.load_path)
    model = build_model(opts, ckpt).to(device)

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    instances = TSP.make_dataset(size=opts.graph_size, num_samples=opts.num_test)
    coords = torch.stack([x for x in instances])
    print(f'test set: {opts.num_test} TSP-{opts.graph_size} instances, seed={opts.seed}')

    # Greedy baseline
    bl = greedy_eval(model, coords, device)
    print(f'\n[greedy theta_star] mean tour cost: {bl.mean():.5f} +- {bl.std()/np.sqrt(len(bl)):.5f}')

    # Probe configs
    configs = [
        # (label, K, leaf_eval, eps, tsched)
        ('K=50  vh   no-explore',  50, 'value_head', 0.0,  None),
        ('K=50  vh   step30+e25',  50, 'value_head', 0.25, 'step30'),  # F.3 default
        ('K=50  vh   step30+e10',  50, 'value_head', 0.10, 'step30'),
        ('K=50  vh   step30+e05',  50, 'value_head', 0.05, 'step30'),
        ('K=50  rol  no-explore',  50, 'rollout',    0.0,  None),
        ('K=50  rol  step30+e25',  50, 'rollout',    0.25, 'step30'),
        ('K=100 vh   no-explore', 100, 'value_head', 0.0,  None),
        ('K=100 rol  no-explore', 100, 'rollout',    0.0,  None),
        ('K=200 vh   no-explore', 200, 'value_head', 0.0,  None),
        ('K=200 rol  no-explore', 200, 'rollout',    0.0,  None),
        # Sample-without-Dirichlet probes: keep tau-schedule sampling, drop noise.
        ('K=100 vh   step30+e0  ', 100, 'value_head', 0.0,  'step30'),
        ('K=100 rol  step30+e0  ', 100, 'rollout',    0.0,  'step30'),
        ('K=100 vh   step30+e05 ', 100, 'value_head', 0.05, 'step30'),
        ('K=100 rol  step30+e05 ', 100, 'rollout',    0.05, 'step30'),
    ]

    print(f"\n{'config':30s}  mean_cost   gap_vs_greedy  frac_better  frac_worse")
    print('-' * 90)
    for label, K, leaf, eps, tsched in configs:
        cfg = MCTSConfig(
            n_simulations=K,
            leaf_eval=leaf,
            value_norm='bl',
            c_puct=0.05,
            temperature=1.0 if tsched is not None else 0.0,
            temperature_schedule=tsched,
            dirichlet_alpha=10.0/opts.graph_size,
            dirichlet_epsilon=eps,
            fpu_mode='running_q',
            fpu_fallback=-1.0,
            root_select='visits',
            tree_reuse=True,
            return_root_visits=False,
            seed=opts.seed,
        )
        mcts_costs = mcts_eval(model, coords, cfg, device)
        gap = mcts_costs - bl  # positive = MCTS worse
        print(f'{label:30s}  {mcts_costs.mean():.5f}    {gap.mean():+.5f}        '
              f'{(gap < 0).mean():.3f}       {(gap > 0).mean():.3f}')


if __name__ == '__main__':
    main()
