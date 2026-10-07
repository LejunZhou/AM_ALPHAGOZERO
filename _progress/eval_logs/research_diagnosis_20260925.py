"""Bounded CPU audit; writes evidence without changing training/search code.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python _progress/eval_logs/research_diagnosis_20260925.py
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.state import StateTSP
from am_baseline.search.mcts import MCTSConfig, MCTSSolver
from scripts.val_stage4_mcts import _build_mcts_config


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '_progress/eval_logs/research_diagnosis_20260925.json'
CHECKPOINTS = {
    'stage1': ROOT / 'outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt',
    'f616': ROOT / 'outputs/tsp_20/f616_400iter_step_decay_20260507T101222_20260507T101229/iter-361_accepted.pt',
}


def load_model(path):
    args_path = path.parent / 'args.json'
    train_args = json.loads(args_path.read_text()) if args_path.exists() else {}
    cfg = dict(embedding_dim=128, n_encode_layers=3, n_heads=8,
               tanh_clipping=10., normalization='batch', feed_forward_hidden=512,
               value_enabled=True, value_hidden_dim=128)
    cfg.update({k: train_args[k] for k in cfg if k in train_args})
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = AttentionModel(SimpleNamespace(**cfg))
    key = 'best_model' if 'best_model' in checkpoint else 'model'
    model.load_state_dict(checkpoint[key])
    model.eval()
    model.set_decode_type('greedy')
    return model, key


@torch.no_grad()
def endpoint_alias(model, coords, norm):
    n = coords.shape[1]
    unvisited = n - 1
    pairs = [(a, b) for a in range(n - 1) for b in range(n - 1) if a != b]
    prefixes = torch.tensor([
        [a] + [k for k in range(n - 1) if k not in (a, b)] + [b]
        for a, b in pairs
    ])
    state = StateTSP.initialize(coords.expand(len(pairs), -1, -1))
    for t in range(n - 1):
        state = state.update(prefixes[:, t])
    fixed = model.precompute_decoder(model.encode(coords))
    fixed = fixed[torch.zeros(len(pairs), dtype=torch.long)]
    _, _, glimpse = model.decode_step(fixed, state, return_glimpse=True)
    v = model.value_head(glimpse)
    if norm == 'bl':
        greedy, _ = model(coords)
        v = v * greedy[0]
    loc = coords[0]
    first, prev = prefixes[:, 0], prefixes[:, -1]
    target = (loc[prev] - loc[unvisited]).norm(dim=-1) + (loc[unvisited] - loc[first]).norm(dim=-1)
    assert state.get_mask().logical_not().sum(-1).eq(1).all()
    assert float((glimpse - glimpse[0]).abs().max()) < 1e-6
    assert float(target.max() - target.min()) > .1
    # Verify targets against actual completion, including closing edge.
    terminal = state.update(torch.full((len(pairs),), unvisited, dtype=torch.long))
    observed = (terminal.get_final_cost() - state.lengths).view(-1)
    assert torch.allclose(target, observed, atol=3e-6)
    return dict(
        endpoint_pairs=len(pairs), value_target_norm=norm,
        max_abs_glimpse_difference=float((glimpse - glimpse[0]).abs().max()),
        raw_prediction_min=float(v.min()), raw_prediction_max=float(v.max()),
        true_remaining_cost_min=float(target.min()), true_remaining_cost_max=float(target.max()),
        best_constant_rmse=float(target.std(unbiased=False)),
        observed_head_rmse=float(((v - target)**2).mean().sqrt()),
    )


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(20260430)
    coords = torch.rand(1000, 20, 2, generator=g)
    alias_coords = torch.rand(1, 20, 2, generator=torch.Generator().manual_seed(20260925))
    results = {'seed': 20260430, 'alias_seed': 20260925,
               'device': 'cpu', 'torch_version': torch.__version__,
               'endpoint_alias': {}, 'evaluation_scale': {}}
    models = {}
    for name, path in CHECKPOINTS.items():
        model, key = load_model(path)
        models[name] = model
        result = endpoint_alias(model, alias_coords, 'bl' if name == 'stage1' else 'none')
        result.update(checkpoint=str(path.relative_to(ROOT)), checkpoint_key=key)
        results['endpoint_alias'][name] = result
        print('endpoint_alias', name, json.dumps(result), flush=True)

    opts = SimpleNamespace(K=40, leaf_eval='value_head', mix_lambda=.5, eps=0.,
                           alpha_factor=10., temperature_schedule='const',
                           match_train=False, c_puct=.05, seed=20260430)
    built, _ = _build_mcts_config(opts, 20, {'value_target_norm': 'none'})
    results['evaluation_scale']['train_norm_provided'] = 'none'
    results['evaluation_scale']['built_norm'] = built.value_target_norm
    assert built.value_target_norm == 'bl', 'Audit finding changed; update this probe.'
    model = models['f616']
    greedy, _ = model(coords)
    results['evaluation_scale']['greedy_mean'] = float(greedy.mean())
    results['evaluation_scale']['n_instances'] = len(coords)
    results['evaluation_scale']['K'] = opts.K
    per_instance = {'greedy': greedy.tolist()}
    for name, cfg in (
        ('value_head_wrong_bl', built),
        ('value_head_correct_raw', replace(built, value_target_norm='none')),
        ('rollout', replace(built, leaf_eval='rollout', value_target_norm='none')),
    ):
        solver = MCTSSolver(model, cfg, device=torch.device('cpu'))
        costs = []
        t0 = time.perf_counter()
        for i in range(len(coords)):
            cost, _ = solver.solve_instance(coords[i:i+1], bl_val=float(greedy[i]))
            costs.append(float(cost))
            if (i + 1) % 250 == 0:
                print(name, i + 1, '/', len(coords), flush=True)
        d = np.asarray(costs) - greedy.numpy()
        result = dict(mean_cost=float(np.mean(costs)), delta_vs_greedy=float(d.mean()),
                      paired_se=float(d.std(ddof=1) / len(d)**.5),
                      wall_seconds=time.perf_counter()-t0)
        results['evaluation_scale'][name] = result
        per_instance[name] = costs
        print(name, json.dumps(result), flush=True)
    results['evaluation_scale']['per_instance'] = per_instance
    OUT.write_text(json.dumps(results, indent=2) + '\n')
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
