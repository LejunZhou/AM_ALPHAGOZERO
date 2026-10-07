"""CPU correctness checks for the Step 2 student-isolation experiment.

PYTHONPATH=src python -m unittest scripts.test_student_isolation -v
Uses a tiny synthetic checkpoint; never touches the user's checkpoints.
"""
import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from am_baseline.baseline.baselines import RolloutBaseline
from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.training.trainer import rollout
from am_baseline.experiments import student_isolation as si


def reference_gumbel(logp, legal, visits, q, root_value, c_visit, c_scale):
    """Row-by-row transcription of mctx qtransform_completed_by_mix_value +
    softmax(prior_logits + completed_q) restricted to legal actions. root_value
    None = the training variant (unvisited children valued at the visited mean)."""
    out = np.zeros_like(logp)
    for idx in np.ndindex(*logp.shape[:-1]):
        lg, lp, n, qq = legal[idx], logp[idx], visits[idx], q[idx]
        prior = np.where(lg, np.exp(np.where(lg, lp, 0.0)), 0.0)
        vis = n > 0
        if vis.any():
            p = np.maximum(prior, 1e-30)
            weighted = (p[vis] * qq[vis]).sum() / p[vis].sum()
        else:
            weighted = 0.0
        if root_value is None:
            v_mix = weighted
        else:
            rv = root_value[idx] if np.isfinite(root_value[idx]) else weighted
            v_mix = (rv + n.sum() * weighted) / (n.sum() + 1.0)
        completed = np.where(vis, np.where(vis, qq, 0.0), v_mix)
        lo, hi = completed[lg].min(), completed[lg].max()
        q_hat = (completed - lo) / max(hi - lo, 1e-8)
        logits = np.where(lg, lp + (c_visit + n.max()) * c_scale * q_hat, -np.inf)
        z = np.exp(logits - logits[lg].max())
        out[idx] = np.where(lg, z, 0.0) / z[lg].sum()
    return out


class StudentIsolationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory(prefix='am_student_isolation_test_')
        self.root = Path(self.temp.name)
        arch = dict(embedding_dim=16, n_encode_layers=1, n_heads=4, tanh_clipping=10.,
                    normalization='batch', feed_forward_hidden=32, value_enabled=True,
                    value_hidden_dim=16)
        model = AttentionModel(SimpleNamespace(**arch))
        opts = SimpleNamespace(graph_size=6, val_size=16, eval_batch_size=16,
                               device=torch.device('cpu'), no_progress_bar=True)
        baseline = RolloutBaseline(model, TSP, opts, rollout, epoch=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = self.root / 'epoch-3.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        baseline=baseline.state_dict()), checkpoint)
        (self.root / 'args.json').write_text(json.dumps(dict(
            **arch, graph_size=6, lambda_v=1.0, value_target_norm='bl', bl_alpha=0.05,
            max_grad_norm=1.0)))
        self.cfg = si.StudentConfig(
            checkpoint=str(checkpoint), output_dir=str(self.root / 'run'), graph_size=6,
            train_instances=10, val_instances=6, test_instances=6, canonical_val_instances=3,
            teacher_K=4, teacher_batch=4, learning_rates=(1e-4, 3e-4), epochs=2,
            batch_instances=4, eval_every=1, control_epoch_size=64, control_batch_size=16,
            control_eval_every=2, control_bl_val_size=16, eval_batch_size=16, device='cpu')

    def tearDown(self):
        self.temp.cleanup()

    # ------------------------------------------------------------------ targets
    def test_gumbel_policy_matches_reference_transcription(self):
        rng = np.random.default_rng(0)
        b, n = 5, 7
        legal = rng.random((b, n, n)) < 0.7
        legal[..., 0] = True
        raw = rng.normal(size=(b, n, n))
        logp = np.where(legal, raw, -np.inf)
        logp = logp - np.log(np.where(legal, np.exp(np.where(legal, raw, 0)), 0).sum(-1, keepdims=True))
        visits = np.where(legal & (rng.random((b, n, n)) < 0.5), rng.integers(1, 9, (b, n, n)), 0)
        visits[..., 0] = np.maximum(visits[..., 0], 1)
        q = np.where(visits > 0, -1 + 0.1 * rng.normal(size=(b, n, n)), np.nan)
        root_value = -1 + 0.05 * rng.normal(size=(b, n))
        root_value[0, 0] = np.nan  # falls back to the visited-weighted Q
        args = [torch.tensor(v) for v in (logp, legal, visits.astype(np.float64), q)]
        for rv in (root_value, None):  # faithful mctx form and the training variant
            got = si.gumbel_policy(*args, None if rv is None else torch.tensor(rv), 50.0, 0.1).numpy()
            want = reference_gumbel(logp, legal, visits.astype(np.float64), q, rv, 50.0, 0.1)
            np.testing.assert_allclose(got, want, atol=1e-10)
            np.testing.assert_allclose(got.sum(-1), 1.0, atol=1e-12)

    def test_gumbel_training_variant_keeps_prior_when_one_child_visited(self):
        logp = torch.log(torch.tensor([[[0.90, 0.06, 0.04]]], dtype=torch.float64))
        legal = torch.ones(1, 1, 3, dtype=torch.bool)
        visits = torch.tensor([[[40., 0., 0.]]], dtype=torch.float64)
        q = torch.tensor([[[-1.0, math.nan, math.nan]]], dtype=torch.float64)
        same = si.gumbel_policy(logp, legal, visits, q, None)
        self.assertTrue(torch.allclose(same, logp.exp()))
        # The faithful form moves almost all mass to unvisited moves on 1e-4 noise.
        faithful = si.gumbel_policy(logp, legal, visits, q, torch.tensor([[-0.9999]],
                                                                         dtype=torch.float64))
        self.assertLess(float(faithful[0, 0, 0]), 0.05)
        # Two visited children: the better-Q child is promoted strongly.
        visits = torch.tensor([[[30., 10., 0.]]], dtype=torch.float64)
        q = torch.tensor([[[-1.01, -1.00, math.nan]]], dtype=torch.float64)
        improved = si.gumbel_policy(logp, legal, visits, q, None)
        self.assertGreater(float(improved[0, 0, 1]), 0.9)

    def test_visit_policy_and_legal_masks(self):
        traj = torch.tensor([[2, 0, 1, 3]])
        legal = si.legal_masks(traj)
        expected = torch.tensor([[[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]]]).bool()
        self.assertTrue(torch.equal(legal, expected))
        visits = torch.zeros(1, 4, 4)
        visits[0, 0, 2], visits[0, 0, 3] = 3, 1
        visits[0, 1:, :] = expected[0, 1:].float()
        pi = si.visit_policy(visits, legal)
        self.assertAlmostEqual(float(pi[0, 0, 2]), 0.75)
        self.assertTrue(torch.allclose(pi.sum(-1), torch.ones(1, 4)))

    def test_best_tour_prefers_cheaper_and_breaks_float_ties_to_greedy(self):
        data = dict(greedy_cost=np.array([1.0, 2.0, 3.0]),
                    mcts_cost=np.array([0.9, 2.0 - 1e-8, 3.1]),
                    greedy_tour=np.array([[0, 1, 2]] * 3), mcts_tour=np.array([[0, 2, 1]] * 3))
        tours, used = si.best_tours(data)
        self.assertEqual(used.tolist(), [True, False, False])
        self.assertEqual(tours[0].tolist(), [0, 2, 1])
        self.assertEqual(tours[1].tolist(), [0, 1, 2])

    # ---------------------------------------------------------------- pipeline
    def test_end_to_end_with_matched_control_budget(self):
        cfg = self.cfg
        si.run_phase(cfg, 'check')
        si.run_phase(cfg, 'teacher')
        out = Path(cfg.output_dir)
        data = si.load_teacher(out, cfg, 'train', 0)
        self.assertEqual(len(data['greedy_cost']), 10)
        self.assertEqual(int(data['st_inst'].max()), 9)  # offsets across 3 shards
        visits, q = si.dense_stats(data, np.arange(10), 6)
        for i in range(10):  # every step has visits on legal moves only
            legal = si.legal_masks(torch.from_numpy(data['mcts_tour'][i:i + 1].astype(np.int64)))[0]
            self.assertTrue(bool((visits[i].sum(-1) > 0).all()))
            self.assertFalse(bool((visits[i][~legal] > 0).any()))
            self.assertTrue(bool(torch.isfinite(q[i][visits[i] > 0]).all()))
        si.run_phase(cfg, 'students')
        si.run_phase(cfg, 'control')
        control = json.loads((out / 'control/s0/done.json').read_text())
        budget = si.matched_budget(out, cfg, 0)
        self.assertAlmostEqual(control['budget_seconds'], budget['budget_seconds'], places=6)
        self.assertGreaterEqual(control['elapsed_seconds'], budget['budget_seconds'])
        si.run_phase(cfg, 'evaluate')
        decision = si.run_phase(cfg, 'report')
        self.assertIn(decision['screen_outcome'], ('CONTINUE', 'STOP'))
        self.assertEqual(len(decision['students']), 3)
        rows = (out / 'results/policies.csv').read_text()
        for name in ('stage1', 'teacher', 'visits_s0', 'gumbel_q_s0', 'best_tour_s0', 'control_s0'):
            self.assertIn(name, rows)
        for name in ('decision.json', 'training_curves.png', 'test_deltas.png'):
            self.assertTrue((out / 'results' / name).exists())

    def test_student_resume_matches_uninterrupted_run(self):
        cfg = self.cfg
        si.run_phase(cfg, 'teacher')
        out = Path(cfg.output_dir)
        frozen, ckpt, _, device = si.open_run(cfg)
        data = si.load_teacher(out, cfg, 'train', 0)
        prepared = si.build_targets(frozen, cfg, data, 'gumbel_q', device)
        val = si.split_coordinates(cfg, 'val')
        full = si.train_student(cfg, ckpt, out, device, 0, 'gumbel_q', 1e-4, prepared, val)
        full_hist = (si.student_dir(out, 0, 'gumbel_q', 1e-4) / 'history.csv').read_text()
        other = replace(cfg, output_dir=str(self.root / 'run_resume'))
        si.open_run(other)
        real_save = si.save_torch
        calls = {'n': 0}

        def crash_after_first_resume(path, payload):
            real_save(path, payload)
            if Path(path).name == 'resume.pt':
                calls['n'] += 1
                if calls['n'] == 1:
                    raise KeyboardInterrupt('simulated disconnect')

        with patch.object(si, 'save_torch', side_effect=crash_after_first_resume):
            with self.assertRaises(KeyboardInterrupt):
                si.train_student(other, ckpt, Path(other.output_dir), device, 0, 'gumbel_q', 1e-4,
                                 prepared, val)
        resumed = si.train_student(other, ckpt, Path(other.output_dir), device, 0, 'gumbel_q',
                                   1e-4, prepared, val)
        self.assertEqual(full['best_val'], resumed['best_val'])
        self.assertEqual(full['best_step'], resumed['best_step'])
        res_hist = (si.student_dir(Path(other.output_dir), 0, 'gumbel_q', 1e-4)
                    / 'history.csv').read_text()
        strip = lambda text: [line.rsplit(',', 1)[0] for line in text.splitlines()]
        self.assertEqual(strip(full_hist), strip(res_hist))  # identical except wall seconds

    def test_teacher_shard_regenerates_identically(self):
        cfg = self.cfg
        si.run_phase(cfg, 'teacher')
        shard = si.teacher_dir(Path(cfg.output_dir), 'train', 0) / 'shard_001.npz'
        with np.load(shard) as z:
            before = {k: z[k] for k in z.files if not k.endswith('seconds')}
        shard.unlink()
        si.run_phase(cfg, 'teacher')
        with np.load(shard) as z:
            for key, value in before.items():
                np.testing.assert_array_equal(z[key], value, err_msg=key)

    def test_confirm_session_reports_screen_and_confirm_together(self):
        cfg = replace(self.cfg, learning_rates=(1e-4,), epochs=1, control_budget_seconds=0.5,
                      canonical_val_instances=0)
        si.run_phase(cfg, 'all')
        confirm = replace(cfg, seeds=(1,), targets=('visits', 'best_tour'))
        si.run_phase(confirm, 'all')
        out = Path(cfg.output_dir)
        rows = (out / 'results/policies.csv').read_text()
        for name in ('visits_s0', 'gumbel_q_s0', 'best_tour_s0', 'control_s0',
                     'visits_s1', 'best_tour_s1', 'control_s1'):
            self.assertIn(name, rows)
        self.assertNotIn('gumbel_q_s1', rows)
        # Seed 1 used new training graphs; val/test stayed fixed.
        a = si.load_teacher(out, cfg, 'train', 0)['coords']
        b = si.load_teacher(out, cfg, 'train', 1)['coords']
        self.assertFalse(np.array_equal(a, b))
        decision = json.loads((out / 'results/decision.json').read_text())
        self.assertEqual({r['seed'] for r in decision['students']}, {0, 1})

    def test_manifest_allows_new_seeds_but_refuses_protocol_changes(self):
        si.run_phase(self.cfg, 'check')
        si.open_run(replace(self.cfg, seeds=(1, 2), targets=('visits',)))
        with self.assertRaises(ValueError):
            si.open_run(replace(self.cfg, epochs=3))

    # ---------------------------------------------------------------- decision
    def test_decision_rule(self):
        rng = np.random.default_rng(1)
        g0 = 6 + rng.normal(0, .1, 400)
        teacher = g0 - 0.05 + rng.normal(0, .01, 400)
        costs = dict(stage1=g0, teacher=teacher,
                     control_s0=g0 - 0.005 + rng.normal(0, .005, 400),
                     visits_s0=g0 - 0.004 + rng.normal(0, .005, 400),       # loses to control
                     gumbel_q_s0=g0 - 0.030 + rng.normal(0, .005, 400),     # passes, val 5.70
                     best_tour_s0=g0 - 0.020 + rng.normal(0, .005, 400))    # passes, val 5.69
        rows = [dict(policy='control_s0', kind='control', seed=0)]
        for name, val in (('visits', 5.75), ('gumbel_q', 5.70), ('best_tour', 5.69)):
            rows.append(dict(policy=f'{name}_s0', kind='student', seed=0, target=name,
                             lr=1e-4, val_cost=val))
        decision = si.decide(costs, rows, 0.25)
        verdict = {r['target']: r['passes'] for r in decision['students']}
        self.assertEqual(verdict, dict(visits=False, gumbel_q=True, best_tour=True))
        self.assertEqual(decision['screen_outcome'], 'CONTINUE')
        self.assertEqual(decision['continue_with'], 'best_tour')  # chosen on validation
        self.assertAlmostEqual(decision['teacher_gain'], 0.05, delta=0.005)  # Stage 1 - teacher
        costs['gumbel_q_s0'] = g0 - 0.004 + rng.normal(0, .005, 400)
        costs['best_tour_s0'] = g0 - 0.0045 + rng.normal(0, .005, 400)
        self.assertEqual(si.decide(costs, rows, 0.25)['screen_outcome'], 'STOP')
        # Beats the control but keeps too little of the teacher's gain: STOP, and the
        # report must say the floor (not the control) failed.
        costs['control_s0'] = g0 + rng.normal(0, .005, 400)
        decision = si.decide(costs, rows, 0.25)
        self.assertEqual(decision['screen_outcome'], 'STOP')
        self.assertTrue(all(r['beats_control'] for r in decision['students']))
        summary = si.outcome_summary(decision)
        self.assertIn('Screen outcome: STOP', summary)
        self.assertEqual(summary.count('beats the control'), 3)
        self.assertEqual(summary.count('below the floor'), 3)
        self.assertNotIn('{', si.INTERPRETATION.format(outcome=summary))

    def test_retention_interval_brackets_point(self):
        rng = np.random.default_rng(2)
        g0 = 6 + rng.normal(0, .1, 500)
        teacher = g0 - 0.05
        student = g0 - 0.02 + rng.normal(0, .01, 500)
        r = si.retention_ci(g0, teacher, student, reps=500)
        self.assertTrue(r['lo'] <= r['retention'] <= r['hi'])
        self.assertAlmostEqual(r['retention'], 0.4, delta=0.05)


if __name__ == '__main__':
    unittest.main()
