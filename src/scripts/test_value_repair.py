"""CPU correctness checks for the isolated evaluator experiment.

PYTHONPATH=src python -m unittest scripts.test_value_repair -v
Uses a tiny synthetic checkpoint, never trains or changes the user's policy.
"""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.state import StateTSP
from am_baseline.search.mcts import MCTSConfig, MCTSSolver
from am_baseline.experiments import value_repair as vr


class ValueRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.temp = tempfile.TemporaryDirectory(prefix='am_value_repair_test_')
        self.root = Path(self.temp.name)
        arch = dict(embedding_dim=16, n_encode_layers=1, n_heads=4,
                    tanh_clipping=10., normalization='batch', feed_forward_hidden=32,
                    value_enabled=True, value_hidden_dim=16)
        self.model = AttentionModel(SimpleNamespace(**arch)).eval().requires_grad_(False)
        self.model.set_decode_type('greedy')
        checkpoint = self.root / 'checkpoint.pt'
        torch.save({'best_model': self.model.state_dict()}, checkpoint)
        (self.root / 'args.json').write_text(json.dumps(dict(**arch, value_target_norm='none')))
        self.cfg = vr.ExperimentConfig(checkpoint=str(checkpoint), output_dir=str(self.root/'run'),
            original_value_norm='none', graph_size=6, train_instances=4, val_instances=2,
            probe_instances=2, calibration_instances=1, search_instances=2, head_seeds=(0,),
            epochs=2, batch_size=32, feature_batch_size=2, train_parent_steps=4,
            hidden_dim=16, search_K=2, timing_repeats=1, device='cpu', cpu_threads=1)

    def tearDown(self):
        self.temp.cleanup()

    def test_invariants_and_normalization_parity(self):
        out = Path(self.cfg.output_dir)
        out.mkdir()
        before = vr.policy_digest(self.model)
        result = vr.check_invariants(self.model, self.cfg, out)
        self.assertTrue(result['reference_search_parity']['value_head'])
        loc = torch.rand(1, 6, 2)
        baseline = float(self.model(loc)[0])
        for norm in ('none', 'bl', 'sqrt_n'):
            config = MCTSConfig(n_simulations=8, leaf_eval='value_head', value_target_norm=norm)
            a = MCTSSolver(self.model, config)
            b = vr.EvaluatorSolver(self.model, config, 'checkpoint_head', original_norm=norm)
            ca, ta = a.solve_instance(loc, baseline)
            cb, tb = b.solve_instance(loc, baseline)
            self.assertTrue(torch.equal(ta, tb))
            self.assertAlmostEqual(float(ca), float(cb), places=6)
        self.assertEqual(before, vr.policy_digest(self.model))

    def test_batched_targets_and_feature_indexing(self):
        coords = vr.split_coordinates(self.cfg, 'train')[:2]
        batch = vr.collect_batch(self.model, coords, self.cfg, 'train', 0)
        # For each child of an empty prefix, independently construct its state,
        # complete the tour, and check batch indexing and the raw target.
        for row in torch.where((batch['step'] == 0) & (batch['source'] == 0))[0]:
            i, action = int(batch['instance'][row]), int(batch['action'][row])
            loc = coords[i:i+1]
            fixed = self.model.precompute_decoder(self.model.encode(loc))
            state = StateTSP.initialize(loc).update(torch.tensor([action]))
            rem = vr.greedy_remaining(self.model, fixed, state)
            _, _, glimpse = self.model.decode_step(fixed, state, return_glimpse=True)
            feat = vr.state_features(fixed, state, glimpse)
            self.assertTrue(torch.allclose(rem[0], batch['y'][row], atol=2e-6))
            self.assertTrue(torch.allclose(feat[0], batch['x'][row], atol=2e-6))
        self.assertTrue(torch.isfinite(batch['x']).all())

    def test_capacity_control_and_independent_instance_splits(self):
        width = 4*16+8+vr.GEO_FEATURES
        mean, std = torch.zeros(width), torch.ones(width)
        heads = {v: vr.RefitHead(v, mean, std, 16, 16) for v in vr.VARIANTS}
        counts = {v: sum(p.numel() for p in h.parameters()) for v, h in heads.items()}
        self.assertLess(abs(counts['original_wide']-counts['repaired']), 18)
        self.assertEqual(counts['repaired_geo'] - counts['repaired'], 16 * vr.GEO_FEATURES)
        x = torch.randn(5, width)
        # repaired_geo adds the raw MST-bound column as a residual
        heads['repaired_geo'].mlp[-1].weight.data.zero_(); heads['repaired_geo'].mlp[-1].bias.data.zero_()
        self.assertTrue(torch.allclose(heads['repaired_geo'](x), x[:, 4*16+8]))
        all_graphs = []
        for split in ('train', 'val', 'probe', 'calibration', 'search'):
            all_graphs.extend(x.numpy().tobytes() for x in vr.split_coordinates(self.cfg, split))
        self.assertEqual(len(all_graphs), len(set(all_graphs)))
        self.assertTrue(torch.equal(vr.split_coordinates(self.cfg, 'train'), vr.split_coordinates(self.cfg, 'train')))

    def test_geometry_features_match_control_and_sorted_distances(self):
        loc = torch.rand(3, 9, 2)
        state = StateTSP.initialize(loc).update(torch.tensor([0, 4, 8])).update(torch.tensor([5, 1, 2]))
        geo = vr.geometry_features(state).numpy()
        self.assertEqual(geo.shape, (3, vr.GEO_FEATURES))
        for row in range(3):
            single = state[torch.tensor([row])]
            self.assertAlmostEqual(float(geo[row, 0]), vr.mst_remaining(single), places=9)
            pts = loc[row].double().numpy()
            legal = np.flatnonzero(~state.get_mask()[row, 0].numpy())
            c, f = int(state.prev_a[row]), int(state.first_a[row])
            dc = np.sort(np.linalg.norm(pts[legal] - pts[c], axis=1))
            df = np.sort(np.linalg.norm(pts[legal] - pts[f], axis=1))
            np.testing.assert_allclose(geo[row, 1:4], dc[:3], atol=1e-9)
            np.testing.assert_allclose(geo[row, 4:7], df[:3], atol=1e-9)
            self.assertAlmostEqual(float(geo[row, 7]), float(np.linalg.norm(pts[c] - pts[f])), places=9)
            self.assertAlmostEqual(float(geo[row, 12]), float(dc.mean()), places=9)
        # one remaining city: bound equals the exact closing path
        prefix = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        one = StateTSP.initialize(loc[:1])
        for t in range(8):
            one = one.update(prefix[:, t])
        pts = loc[0].double().numpy()
        expect = np.linalg.norm(pts[7] - pts[8]) + np.linalg.norm(pts[8] - pts[0])
        self.assertAlmostEqual(float(vr.geometry_features(one)[0, 0]), expect, places=9)
        # fewer than three remaining: sorted slots repeat the last distance
        g = vr.geometry_features(one).numpy()[0]
        self.assertEqual(g[1], g[2]); self.assertEqual(g[2], g[3])

    def test_exact_oracle_immediate_edge_and_mst_bound(self):
        import itertools
        loc = torch.rand(1, 6, 2)
        state = StateTSP.initialize(loc).update(torch.tensor([0])).update(torch.tensor([1]))
        points = loc[0].double().numpy()
        legal = [2, 3, 4, 5]
        oracle = vr.exact_child_remaining(points, legal, 0)
        q = oracle + np.linalg.norm(points[legal]-points[1], axis=1)
        exact = min(sum(np.linalg.norm(points[a]-points[b]) for a, b in zip((1, *p), (*p, 0)))
                    for p in itertools.permutations(legal))
        self.assertAlmostEqual(float(q.min()), exact, places=10)
        self.assertLessEqual(vr.mst_remaining(state), exact + 1e-9)

    def test_resume_from_interrupted_epoch_and_manifest_guard(self):
        clean = replace(self.cfg, output_dir=str(self.root/'clean'))
        vr.run_phase(clean, 'train')
        original_save = vr.save_torch
        interrupted = False
        def save_then_interrupt(path, data):
            nonlocal interrupted
            original_save(path, data)
            if Path(path).name == 'original_s0.pt' and data['epoch'] == 1 and not interrupted:
                interrupted = True
                raise RuntimeError('simulated disconnect')
        with patch.object(vr, 'save_torch', save_then_interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated disconnect'):
                vr.run_phase(self.cfg, 'train')
        vr.run_phase(self.cfg, 'train')
        for variant in vr.VARIANTS:
            a = vr.load_pt(Path(clean.output_dir)/'heads'/f'{variant}_s0.pt')
            b = vr.load_pt(Path(self.cfg.output_dir)/'heads'/f'{variant}_s0.pt')
            for key in a['last_state']:
                self.assertTrue(torch.equal(a['last_state'][key], b['last_state'][key]), key)
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            vr.run_phase(replace(self.cfg, data_seed=self.cfg.data_seed+1), 'check')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            vr.load_policy(replace(self.cfg, original_value_norm='bl'), torch.device('cpu'))

    def test_end_to_end_and_resume(self):
        before = vr.policy_digest(self.model)
        vr.run_phase(self.cfg, 'all')
        out = Path(self.cfg.output_dir)
        for filename in ('invariants.json', 'training_history.csv', 'sibling_summary.csv',
                         'sibling_child_scores.npz', 'search_summary.csv', 'paired_contrasts.csv'):
            self.assertTrue((out/filename).exists(), filename)
        summary = (out/'sibling_summary.csv').read_bytes()
        search = (out/'search_summary.csv').read_bytes()
        vr.run_phase(self.cfg, 'all')
        self.assertEqual(summary, (out/'sibling_summary.csv').read_bytes())
        self.assertEqual(search, (out/'search_summary.csv').read_bytes())
        self.assertEqual(before, vr.policy_digest(self.model))


if __name__ == '__main__':
    unittest.main()
