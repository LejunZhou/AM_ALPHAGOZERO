"""Stage 5 §I Step 2 — student isolation at TSP-50 (colab_student_isolation.ipynb).

Question: can a REINFORCE-converged Attention Model absorb the improvements a
rollout-MCTS teacher finds, and does the training target decide it?

Design (see _plans/student_isolation_colab_plan.md):
- One FIXED teacher: the frozen Stage 1 policy plus batched C++ MCTS with greedy
  rollout leaves. It labels each training graph once.
- Three targets trained on IDENTICAL teacher data:
    visits     AlphaZero visit counts N(s,a)/sum N along the teacher trajectory
    gumbel_q   completed-Q improved policy softmax(log pi + sigma(q_hat)) with the
               mctx `qtransform_completed_by_mix_value` defaults
    best_tour  imitation of the better of {MCTS tour, Stage 1 greedy tour}
- Students start from the Stage 1 weights. Validation greedy cost selects the
  checkpoint and the learning rate; the test set only reports.
- Control: REINFORCE continued from the same checkpoint (optimizer and rollout
  baseline restored) for the same wall time.

Phases: check, teacher, students, control, evaluate, report (and all).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.search.mcts import MCTSConfig, MCTSSolver
from am_baseline.search.mcts_cpp import solver as cpp_solver


TARGETS = ('visits', 'gumbel_q', 'best_tour')
SOURCE_FILES = (
    'model/attention_model.py', 'model/encoder.py', 'model/decoder.py',
    'model/value_head.py', 'problem/state.py', 'problem/tsp.py',
    'utils/tensor_ops.py', 'search/mcts.py', 'search/puct.py', 'search/tree.py',
    'search/mcts_cpp/solver.py', 'search/mcts_cpp/mcts.cpp',
    'search/mcts_cpp/mcts.hpp', 'search/mcts_cpp/bindings.cpp',
    'baseline/baselines.py', 'training/trainer.py',
    'experiments/student_isolation.py',
)
# Fields that choose WHICH units of work run (or only change the report). They
# are excluded from the run signature so Screen and Confirm share one directory.
WORK_FIELDS = ('checkpoint', 'output_dir', 'seeds', 'targets', 'control',
               'canonical_val_instances', 'canonical_val_seed', 'min_retention',
               'device', 'eval_batch_size')
SPLIT_OFFSETS = dict(val=23, test=37)
TIE = 1e-6  # MCTS costs are float32; smaller gaps to the greedy cost are ties


@dataclass
class StudentConfig:
    checkpoint: str
    output_dir: str
    checkpoint_key: str = 'model'
    graph_size: int = 50
    # Splits. Train graphs are drawn per seed; val/test are shared by all seeds.
    train_instances: int = 32768
    val_instances: int = 1024
    test_instances: int = 2048
    canonical_val_instances: int = 10000   # torch.manual_seed(42) set; 0 disables
    canonical_val_seed: int = 42
    data_seed: int = 20261003
    seeds: tuple = (0,)
    targets: tuple = TARGETS
    # Teacher: frozen policy + batched C++ MCTS with greedy-rollout leaves.
    teacher_K: int = 40
    c_puct: float = 0.05
    teacher_batch: int = 1024              # graphs per C++ batch and per shard
    # Students.
    learning_rates: tuple = (1e-5, 3e-5, 1e-4)
    epochs: int = 8
    batch_instances: int = 64
    eval_every: int = 32                   # optimizer steps between val checks
    max_grad_norm: float = 1.0
    gumbel_c_visit: float = 50.0           # mctx maxvisit_init
    gumbel_c_scale: float = 0.1            # mctx value_scale
    # Control: REINFORCE continued from the checkpoint at matched wall time.
    control: bool = True
    control_lr: float = 1e-4
    control_epoch_size: int = 1280000
    control_batch_size: int = 512
    control_eval_every: int = 500          # batches between val checks
    control_budget_seconds: float = 0.0    # 0 = matched automatically
    control_bl_val_size: int = 10000
    # Decision and runtime.
    min_retention: float = 0.25
    eval_batch_size: int = 1024
    device: str = 'auto'

    def validate(self):
        for name in ('graph_size', 'train_instances', 'val_instances', 'test_instances',
                     'teacher_K', 'teacher_batch', 'epochs', 'batch_instances',
                     'eval_every', 'control_epoch_size', 'control_batch_size',
                     'control_eval_every', 'control_bl_val_size', 'eval_batch_size'):
            if int(getattr(self, name)) < 1:
                raise ValueError(f'{name} must be positive')
        if self.graph_size < 4:
            raise ValueError('graph_size must be at least 4')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be nonempty and distinct')
        unknown = set(self.targets) - set(TARGETS)
        if not self.targets or unknown:
            raise ValueError(f'targets must be a nonempty subset of {TARGETS}')
        if not self.learning_rates or min(self.learning_rates) <= 0:
            raise ValueError('learning_rates must be positive')
        if self.c_puct <= 0 or self.control_lr <= 0 or self.max_grad_norm <= 0:
            raise ValueError('c_puct, control_lr and max_grad_norm must be positive')
        if self.control_budget_seconds < 0 or not 0 <= self.min_retention <= 1:
            raise ValueError('invalid control budget or retention threshold')


def profile_overrides(profile):
    """Named presets shared by the notebook and the CLI."""
    if profile == 'main':
        return {}
    if profile == 'confirm':
        return dict(seeds=(1, 2))
    if profile == 'pilot':
        return dict(train_instances=512, val_instances=256, test_instances=256,
                    canonical_val_instances=0, teacher_batch=256,
                    learning_rates=(3e-5,), epochs=2, batch_instances=32,
                    eval_every=8, control_budget_seconds=180.0,
                    control_epoch_size=128000, control_eval_every=100)
    if profile == 'cpu_smoke':
        return dict(graph_size=10, train_instances=24, val_instances=8, test_instances=8,
                    canonical_val_instances=0, teacher_K=4, teacher_batch=8,
                    learning_rates=(1e-4,), epochs=2, batch_instances=8, eval_every=2,
                    control_budget_seconds=20.0, control_epoch_size=256,
                    control_batch_size=32, control_eval_every=4,
                    control_bl_val_size=64, eval_batch_size=64, device='cpu')
    raise ValueError(f'unknown profile {profile!r}')


# --------------------------------------------------------------------------- #
# Provenance and file helpers                                                  #
# --------------------------------------------------------------------------- #

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for name in SOURCE_FILES:
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()


def policy_digest(model):
    """Parameters AND batch-normalization buffers."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def save_torch(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(k for k in row if k not in fields)
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_pt(path):
    # The user's own checkpoints and run artifacts (they contain pickled
    # baseline objects, so weights_only=False is required).
    return torch.load(path, map_location='cpu', weights_only=False)


def clean(value):
    """JSON-safe floats (NaN/inf become None)."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# --------------------------------------------------------------------------- #
# Model loading, data, greedy evaluation, teacher forcing                      #
# --------------------------------------------------------------------------- #

def checkpoint_args(cfg):
    path = Path(cfg.checkpoint).parent / 'args.json'
    return json.loads(path.read_text()) if path.exists() else {}


def architecture(cfg):
    arch = dict(embedding_dim=128, n_encode_layers=3, n_heads=8, tanh_clipping=10.,
                normalization='batch', feed_forward_hidden=512, value_enabled=True,
                value_hidden_dim=128)
    args = checkpoint_args(cfg)
    arch.update({k: args[k] for k in arch if k in args})
    return arch


def build_model(cfg, state_dict, device, trainable):
    model = AttentionModel(SimpleNamespace(**architecture(cfg)))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    if trainable:
        model.train().requires_grad_(True)
    else:
        model.eval().requires_grad_(False)
    model.set_decode_type('greedy')
    return model


def checkpoint_state(cfg):
    ckpt = load_pt(cfg.checkpoint)
    if cfg.checkpoint_key not in ckpt:
        raise ValueError(f'checkpoint has no {cfg.checkpoint_key!r} entry')
    return ckpt


def resolve_device(cfg):
    if cfg.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg.device)


def split_coordinates(cfg, split, seed=0):
    """Seeded graphs. Train graphs depend on the seed; val/test do not."""
    n_nodes = cfg.graph_size
    if split == 'train':
        gen = torch.Generator().manual_seed(cfg.data_seed + 11 + 1009 * int(seed))
        return torch.rand(cfg.train_instances, n_nodes, 2, generator=gen)
    if split in SPLIT_OFFSETS:
        gen = torch.Generator().manual_seed(cfg.data_seed + SPLIT_OFFSETS[split])
        count = getattr(cfg, f'{split}_instances')
        return torch.rand(count, n_nodes, 2, generator=gen)
    if split == 'canonical':
        # Same draw as TSP.make_dataset after torch.manual_seed(42): one
        # uniform_(0, 1) tensor per instance from the global CPU generator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.canonical_val_seed)
            rows = [torch.FloatTensor(n_nodes, 2).uniform_(0, 1)
                    for _ in range(cfg.canonical_val_instances)]
        return torch.stack(rows)
    raise ValueError(split)


@torch.no_grad()
def greedy_eval(model, coords, device, batch_size=1024):
    """Greedy tour costs (float64) and tours for every graph."""
    was_training = model.training
    model.eval()
    model.set_decode_type('greedy')
    costs, tours = [], []
    for start in range(0, len(coords), batch_size):
        x = coords[start:start + batch_size].to(device)
        cost, _ll, pi = model(x, return_pi=True)
        costs.append(cost.double().cpu())
        tours.append(pi.cpu())
    model.train(was_training)
    return torch.cat(costs), torch.cat(tours)


def teacher_forced_logp(model, coords, actions):
    """Log-probabilities (B, N, N) at every step along the given action sequences.
    Row t is the policy over the action taken at step t; illegal entries are -inf."""
    embeddings = model.encode(coords)
    fixed = model.precompute_decoder(embeddings)
    state = TSP.make_state(coords)
    out = []
    for t in range(coords.size(1)):
        log_p, _mask = model.decode_step(fixed, state)
        out.append(log_p[:, 0, :])
        state = state.update(actions[:, t])
    return torch.stack(out, 1)


@torch.no_grad()
def batched_logp(model, coords, actions, device, batch_size=512):
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(coords), batch_size):
        out.append(teacher_forced_logp(model, coords[start:start + batch_size].to(device),
                                       actions[start:start + batch_size].to(device)).cpu())
    model.train(was_training)
    return torch.cat(out)


def legal_masks(traj):
    """legal[i, t, a]: city a is unvisited before step t of trajectory i."""
    n_nodes = traj.size(1)
    onehot = torch.nn.functional.one_hot(traj, n_nodes).to(torch.int16)
    visited_before = (onehot.cumsum(1) - onehot) > 0
    return ~visited_before


# --------------------------------------------------------------------------- #
# Run bookkeeping                                                             #
# --------------------------------------------------------------------------- #

def signature(cfg):
    protocol = {k: v for k, v in asdict(cfg).items() if k not in WORK_FIELDS}
    return json.loads(json.dumps(dict(protocol=protocol,
                                      checkpoint_sha256=sha256(cfg.checkpoint),
                                      source_sha256=source_digest())))


def open_run(cfg):
    cfg.validate()
    if not cpp_solver.HAVE_CPP_MCTS:
        raise ImportError('C++ MCTS extension is not built; build it before running.') \
            from cpp_solver._IMPORT_ERROR
    device = resolve_device(cfg)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_state(cfg)
    frozen = build_model(cfg, ckpt[cfg.checkpoint_key], device, trainable=False)
    sig = signature(cfg)
    sig['policy_sha256'] = policy_digest(frozen)
    manifest = output / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['signature'] != sig:
            raise ValueError('Run manifest differs (protocol, checkpoint or source). Use a NEW '
                             'output_dir; existing results are preserved.')
    else:
        save_json(manifest, dict(signature=sig, checkpoint_path=str(cfg.checkpoint),
                                 architecture=architecture(cfg),
                                 created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    with (output / 'sessions.jsonl').open('a') as f:
        f.write(json.dumps(dict(
            time=time.time(), device=str(device), torch=torch.__version__,
            numpy=np.__version__, python=platform.python_version(), platform=platform.platform(),
            cpu_count=os.cpu_count(), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)) + '\n')
    return frozen, ckpt, output, device


def teacher_dir(output, split, seed):
    return output / 'teacher' / (f'train_s{seed}' if split == 'train' else split)


def student_dir(output, seed, target, lr):
    return output / 'students' / f's{seed}' / target / f'lr{lr:g}'


def control_dir(output, seed):
    return output / 'control' / f's{seed}'


# --------------------------------------------------------------------------- #
# Teacher                                                                      #
# --------------------------------------------------------------------------- #

def teacher_config(cfg, K=None, seed=None):
    return MCTSConfig(
        n_simulations=int(K or cfg.teacher_K), c_puct=cfg.c_puct, temperature=0.0,
        dirichlet_epsilon=0.0, leaf_eval='rollout', value_norm='bl', tree_reuse=True,
        fpu_mode='running_q', root_select='visits', return_root_visits=True,
        return_root_q=True, seed=int(cfg.data_seed if seed is None else seed))


def flatten_root_stats(solver, n_nodes):
    """Sparse (instance, step, action, N, Q) rows plus root estimates (B, N)."""
    visits_all = solver.root_visit_dists_per_instance
    q_all = solver.root_q_dists_per_instance
    values_all = solver.root_values_per_instance
    rows = []
    root_value = np.full((len(visits_all), n_nodes), np.nan, dtype=np.float32)
    for j, (visits, qs, values) in enumerate(zip(visits_all, q_all, values_all)):
        if not len(visits) == len(qs) == len(values) == n_nodes:
            raise RuntimeError('root statistics do not cover every tour step')
        for t in range(n_nodes):
            if visits[t].keys() != qs[t].keys():
                raise RuntimeError('visited-children sets differ between N and Q dumps')
            for a, count in visits[t].items():
                rows.append((j, t, a, count, qs[t][a]))
            root_value[j, t] = values[t]
    arr = np.array(rows, dtype=np.float64).reshape(-1, 5)
    return dict(st_inst=arr[:, 0].astype(np.int32), st_step=arr[:, 1].astype(np.int16),
                st_action=arr[:, 2].astype(np.int16), st_n=arr[:, 3].astype(np.int32),
                st_q=arr[:, 4].astype(np.float32), root_value=root_value)


def label_graphs(frozen, cfg, coords, device, K=None, batch=None):
    """Greedy and MCTS tours for a block of graphs, with root statistics."""
    n_nodes = coords.size(1)
    start = time.time()
    g_cost, g_tour = greedy_eval(frozen, coords, device, cfg.eval_batch_size)
    greedy_seconds = time.time() - start
    solver = cpp_solver.CppBatchMCTSSolver(frozen, teacher_config(cfg, K), device=device,
                                           mcts_batch_size=int(batch or cfg.teacher_batch))
    start = time.time()
    m_cost, m_tour = solver.solve_batch(coords.to(device), bl_vals=g_cost.to(device))
    mcts_seconds = time.time() - start
    record = dict(coords=coords.numpy().astype(np.float32),
                  greedy_cost=g_cost.numpy(), greedy_tour=g_tour.numpy().astype(np.int16),
                  mcts_cost=m_cost.double().cpu().numpy(),
                  mcts_tour=m_tour.cpu().numpy().astype(np.int16),
                  greedy_seconds=np.float64(greedy_seconds), mcts_seconds=np.float64(mcts_seconds))
    record.update(flatten_root_stats(solver, n_nodes))
    return record


def teacher_split(frozen, cfg, output, device, split, seed=0):
    coords = split_coordinates(cfg, split, seed)
    folder = teacher_dir(output, split, seed)
    n_shards = math.ceil(len(coords) / cfg.teacher_batch)
    started = time.time()
    for k in range(n_shards):
        path = folder / f'shard_{k:03d}.npz'
        if path.exists():
            continue
        block = coords[k * cfg.teacher_batch:(k + 1) * cfg.teacher_batch]
        record = label_graphs(frozen, cfg, block, device)
        save_npz(path, **record)
        per_graph = float(record['greedy_seconds'] + record['mcts_seconds']) / len(block)
        remaining = (len(coords) - (k + 1) * cfg.teacher_batch) * per_graph
        gain = float(np.mean(record['greedy_cost'] - record['mcts_cost']))
        print(f'teacher {folder.name} shard {k + 1}/{n_shards}: {per_graph:.3f} s/graph, '
              f'greedy - MCTS {gain:+.4f}, ETA for this split {max(remaining, 0) / 60:.1f} min',
              flush=True)
    data = load_teacher(output, cfg, split, seed)
    summary = dict(split=split, seed=seed, graphs=int(len(data['greedy_cost'])),
                   greedy_mean=float(data['greedy_cost'].mean()),
                   mcts_mean=float(data['mcts_cost'].mean()),
                   mcts_better_fraction=float(np.mean(data['mcts_cost'] < data['greedy_cost'] - TIE)),
                   tours_differ_fraction=float(np.mean((data['mcts_tour'] != data['greedy_tour']).any(1))),
                   wall_seconds=float(data['greedy_seconds'].sum() + data['mcts_seconds'].sum()),
                   session_seconds=time.time() - started)
    save_json(folder / 'summary.json', summary)
    return summary


def load_teacher(output, cfg, split, seed=0):
    folder = teacher_dir(output, split, seed)
    expected = {'train': cfg.train_instances, 'val': cfg.val_instances,
                'test': cfg.test_instances}[split]
    n_shards = math.ceil(expected / cfg.teacher_batch)
    paths = [folder / f'shard_{k:03d}.npz' for k in range(n_shards)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f'teacher data incomplete for {folder}: missing {missing[:3]}')
    parts = []
    for p in paths:
        with np.load(p) as z:
            parts.append({k: z[k] for k in z.files})
    offset, out = 0, {}
    for part in parts:
        part = dict(part)
        part['st_inst'] = part['st_inst'].astype(np.int64) + offset
        offset += len(part['greedy_cost'])
        part['greedy_seconds'] = np.atleast_1d(part['greedy_seconds'])
        part['mcts_seconds'] = np.atleast_1d(part['mcts_seconds'])
        for key, value in part.items():
            out.setdefault(key, []).append(value)
    data = {k: np.concatenate(v) for k, v in out.items()}
    if len(data['greedy_cost']) != expected:
        raise RuntimeError(f'{folder}: expected {expected} graphs, found {len(data["greedy_cost"])}')
    return data


# --------------------------------------------------------------------------- #
# Targets                                                                      #
# --------------------------------------------------------------------------- #

def dense_stats(data, index, n_nodes):
    """Dense visits (b, N, N) and Q (b, N, N; NaN when unvisited) for graphs `index`."""
    index = np.asarray(index)
    position = np.full(len(data['greedy_cost']), -1, dtype=np.int64)
    position[index] = np.arange(len(index))
    keep = position[data['st_inst']] >= 0
    rows = position[data['st_inst'][keep]]
    steps = data['st_step'][keep].astype(np.int64)
    actions = data['st_action'][keep].astype(np.int64)
    visits = np.zeros((len(index), n_nodes, n_nodes), dtype=np.float32)
    q = np.full((len(index), n_nodes, n_nodes), np.nan, dtype=np.float32)
    visits[rows, steps, actions] = data['st_n'][keep]
    q[rows, steps, actions] = data['st_q'][keep]
    return torch.from_numpy(visits), torch.from_numpy(q)


def visit_policy(visits, legal):
    total = visits.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise RuntimeError('a root step has no visits')
    return torch.where(legal, visits / total, torch.zeros_like(visits))


def gumbel_policy(logp, legal, visits, q, root_value=None, c_visit=50.0, c_scale=0.1,
                  tie_tol=1e-6):
    """Completed-Q improved policy (Danihelka et al., 2022; mctx defaults).

    pi'(a) = softmax(log pi(a) + (c_visit + max_b N(b)) * c_scale * q_hat(a)) over legal a,
    q_hat = completed Q min-max rescaled over legal children. Visited children keep
    their search Q; unvisited children get a completion value.

    root_value given: mctx's mixed value v_mix = (v_root + sum N * wq) / (1 + sum N),
    wq = prior-weighted mean Q of the visited children (faithful transcription).
    root_value=None (used for training): v_mix = wq. Reason, measured on Stage 1
    TSP-50 teacher data: PUCT at c_puct 0.05 visits ONE child at 78% of steps,
    and that child's averaged Q differs from the root's single greedy-rollout value
    by ~1e-3 in either direction. The min-max rescale then turns that noise into a
    5-9 nat swing toward every unvisited move. Valuing unvisited moves at wq keeps
    the target equal to the prior where the search compared nothing, and keeps the
    search's comparison wherever it visited two or more children (every step where
    the teacher overrode the prior's argmax).

    Computed in float64. A completed-Q spread at or below `tie_tol` (Q is stored in
    float32, resolution ~1e-7 near -1) is a tie: q_hat = 0, so pi' = pi there.
    Without this, rounding in the weighted mean recreates the swing above."""
    out_dtype = logp.dtype
    logp, q, visits = logp.double(), q.double(), visits.double()
    if root_value is not None:
        root_value = root_value.double()
    visited = visits > 0
    prior = torch.where(legal, logp.exp(), torch.zeros_like(logp))
    prior_vis = torch.where(visited, prior.clamp_min(1e-30), torch.zeros_like(prior))
    q_vis = torch.where(visited, q, torch.zeros_like(q))
    sum_n = visits.sum(-1)
    weighted_q = (prior_vis * q_vis).sum(-1) / prior_vis.sum(-1).clamp_min(1e-30)
    if root_value is None:
        v_mix = weighted_q
    else:
        root = torch.where(torch.isfinite(root_value), root_value, weighted_q)
        v_mix = (root + sum_n * weighted_q) / (sum_n + 1.0)
    completed = torch.where(visited, q_vis, v_mix.unsqueeze(-1))
    big = torch.finfo(completed.dtype).max
    low = torch.where(legal, completed, torch.full_like(completed, big)).amin(-1, keepdim=True)
    high = torch.where(legal, completed, torch.full_like(completed, -big)).amax(-1, keepdim=True)
    spread = high - low
    q_hat = torch.where(spread > tie_tol, (completed - low) / spread.clamp_min(1e-300),
                        torch.zeros_like(completed))
    sigma = (c_visit + visits.amax(-1, keepdim=True)) * c_scale * q_hat
    logits = torch.where(legal, logp + sigma, torch.full_like(logp, -math.inf))
    return torch.softmax(logits, -1).to(out_dtype)


def best_tours(data):
    use_mcts = data['mcts_cost'] < data['greedy_cost'] - TIE
    tours = np.where(use_mcts[:, None], data['mcts_tour'], data['greedy_tour'])
    return torch.from_numpy(tours.astype(np.int64)), use_mcts


def build_targets(frozen, cfg, data, target, device, chunk=1024):
    """Trajectories (n, N) and dense targets (n, N, N), or None for best_tour."""
    n_nodes = cfg.graph_size
    coords = torch.from_numpy(data['coords'])
    if target == 'best_tour':
        traj, _ = best_tours(data)
        return dict(coords=coords, traj=traj, target=None)
    traj = torch.from_numpy(data['mcts_tour'].astype(np.int64))
    out = torch.empty(len(traj), n_nodes, n_nodes, dtype=torch.float32)
    for start in range(0, len(traj), chunk):
        index = np.arange(start, min(start + chunk, len(traj)))
        visits, q = dense_stats(data, index, n_nodes)
        legal = legal_masks(traj[index])
        if target == 'visits':
            out[index] = visit_policy(visits, legal)
        else:
            logp = batched_logp(frozen, coords[index], traj[index], device)
            out[index] = gumbel_policy(logp, legal, visits, q, None,
                                       cfg.gumbel_c_visit, cfg.gumbel_c_scale)
    return dict(coords=coords, traj=traj, target=out)


# --------------------------------------------------------------------------- #
# Students                                                                     #
# --------------------------------------------------------------------------- #

def distillation_loss(model, coords, traj, target):
    """Mean teacher-forced cross-entropy over the N-1 non-forced steps."""
    logp = teacher_forced_logp(model, coords, traj)
    safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
    if target is None:
        ce = -safe.gather(-1, traj.unsqueeze(-1)).squeeze(-1)
    else:
        ce = -(target * safe).sum(-1)
    return ce[:, :-1].mean()


def train_student(cfg, ckpt, output, device, seed, target_name, lr, prepared, val_coords):
    run = student_dir(output, seed, target_name, lr)
    if (run / 'done.json').exists():
        return json.loads((run / 'done.json').read_text())
    model = build_model(cfg, ckpt[cfg.checkpoint_key], device, trainable=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(prepared['traj'])
    steps_per_epoch = math.ceil(n / cfg.batch_instances)
    resume = run / 'resume.pt'
    if resume.exists():
        state = load_pt(resume)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, step, history = state['epoch'], state['step'], state['history']
        best_val, best_step, best_state = state['best_val'], state['best_step'], state['best_state']
        wall = state['wall']
        print(f'resume {run.relative_to(output)} at epoch {start_epoch}', flush=True)
    else:
        start_epoch, step, wall = 0, 0, 0.0
        val0 = float(greedy_eval(model, val_coords, device, cfg.eval_batch_size)[0].mean())
        history = [dict(step=0, epoch=0, train_loss=float('nan'), val_cost=val0, seconds=0.0)]
        best_val, best_step = val0, 0
        best_state = copy.deepcopy(model.state_dict())
    session = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(
            cfg.data_seed + 100003 * int(seed) + epoch))
        model.train()
        running = []
        for b in range(steps_per_epoch):
            index = order[b * cfg.batch_instances:(b + 1) * cfg.batch_instances]
            coords = prepared['coords'][index].to(device)
            traj = prepared['traj'][index].to(device)
            target = None if prepared['target'] is None else prepared['target'][index].to(device)
            loss = distillation_loss(model, coords, traj, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            step += 1
            running.append(float(loss.detach()))
            if step % cfg.eval_every == 0 or step == cfg.epochs * steps_per_epoch:
                val = float(greedy_eval(model, val_coords, device, cfg.eval_batch_size)[0].mean())
                history.append(dict(step=step, epoch=epoch, train_loss=float(np.mean(running)),
                                    val_cost=val, seconds=wall + time.time() - session))
                running = []
                if val < best_val:
                    best_val, best_step = val, step
                    best_state = copy.deepcopy(model.state_dict())
                model.train()
        save_torch(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                epoch=epoch + 1, step=step, history=history, best_val=best_val,
                                best_step=best_step, best_state=best_state,
                                wall=wall + time.time() - session))
        print(f'{run.relative_to(output)} epoch {epoch + 1}/{cfg.epochs}: '
              f'val {history[-1]["val_cost"]:.5f} (best {best_val:.5f} @ step {best_step})',
              flush=True)
    wall = wall + time.time() - session
    save_torch(run / 'best.pt', dict(model=best_state, val_cost=best_val, step=best_step))
    write_csv(run / 'history.csv', history)
    done = dict(seed=seed, target=target_name, lr=lr, best_val=best_val, best_step=best_step,
                steps=step, wall_seconds=wall, initial_val=history[0]['val_cost'])
    save_json(run / 'done.json', clean(done))
    if resume.exists():
        resume.unlink()
    return done


def run_students(frozen, ckpt, cfg, output, device):
    val_coords = split_coordinates(cfg, 'val')
    for seed in cfg.seeds:
        data = load_teacher(output, cfg, 'train', seed)
        for target_name in cfg.targets:
            runs = [student_dir(output, seed, target_name, lr) for lr in cfg.learning_rates]
            if all((r / 'done.json').exists() for r in runs):
                continue
            prep_file = output / 'students' / f's{seed}' / target_name / 'prep.json'
            start = time.time()
            prepared = build_targets(frozen, cfg, data, target_name, device)
            prep_seconds = time.time() - start
            if not prep_file.exists():
                save_json(prep_file, dict(prep_seconds=prep_seconds))
            for lr in cfg.learning_rates:
                done = train_student(cfg, ckpt, output, device, seed, target_name, lr,
                                     prepared, val_coords)
                print(f'student s{seed} {target_name} lr={lr:g}: best val {done["best_val"]:.5f} '
                      f'(start {done["initial_val"]:.5f}), {done["wall_seconds"] / 60:.1f} min',
                      flush=True)
            del prepared


def select_student(output, cfg, seed, target_name):
    """Best learning rate by validation greedy cost (ties: smaller lr)."""
    done = []
    for lr in sorted(cfg.learning_rates):
        path = student_dir(output, seed, target_name, lr) / 'done.json'
        if path.exists():
            done.append(json.loads(path.read_text()))
    if len(done) != len(cfg.learning_rates):
        return None
    return min(done, key=lambda d: (d['best_val'], d['lr']))


# --------------------------------------------------------------------------- #
# Control: REINFORCE continued at matched wall time                            #
# --------------------------------------------------------------------------- #

class CoordinateSet(Dataset):
    def __init__(self, coords):
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        return self.coords[index]


def matched_budget(output, cfg, seed):
    """Teacher wall on this seed's training split + one target's full student cost."""
    summary = teacher_dir(output, 'train', seed) / 'summary.json'
    if not summary.exists():
        raise FileNotFoundError('run the teacher phase before the control')
    teacher_seconds = json.loads(summary.read_text())['wall_seconds']
    per_target = []
    for target_name in cfg.targets:
        walls = []
        for lr in cfg.learning_rates:
            path = student_dir(output, seed, target_name, lr) / 'done.json'
            if not path.exists():
                raise FileNotFoundError('run the students phase before the control')
            walls.append(json.loads(path.read_text())['wall_seconds'])
        prep = output / 'students' / f's{seed}' / target_name / 'prep.json'
        prep_seconds = json.loads(prep.read_text())['prep_seconds'] if prep.exists() else 0.0
        per_target.append(prep_seconds + sum(walls))
    return dict(teacher_seconds=teacher_seconds, student_seconds=float(np.mean(per_target)),
                budget_seconds=float(teacher_seconds + np.mean(per_target)))


def control_options(cfg, device):
    args = checkpoint_args(cfg)
    return SimpleNamespace(
        graph_size=cfg.graph_size, val_size=cfg.control_bl_val_size,
        eval_batch_size=cfg.eval_batch_size, bl_alpha=args.get('bl_alpha', 0.05),
        device=device, no_progress_bar=True, lambda_v=args.get('lambda_v', 1.0),
        value_target_norm=args.get('value_target_norm', 'bl'),
        max_grad_norm=args.get('max_grad_norm', 1.0), log_step=10 ** 12,
        batch_size=cfg.control_batch_size, epoch_size=cfg.control_epoch_size,
        use_cuda=device.type == 'cuda', run_name='student_isolation_control')


def run_control(cfg, ckpt, output, device, seed):
    from am_baseline.baseline.baselines import RolloutBaseline
    from am_baseline.training.trainer import rollout, train_batch

    run = control_dir(output, seed)
    if (run / 'done.json').exists():
        return json.loads((run / 'done.json').read_text())
    if cfg.control_budget_seconds > 0:
        budget = dict(budget_seconds=float(cfg.control_budget_seconds), teacher_seconds=None,
                      student_seconds=None)
    else:
        budget = matched_budget(output, cfg, seed)
    limit = budget['budget_seconds']
    opts = control_options(cfg, device)
    val_coords = split_coordinates(cfg, 'val')
    model = build_model(cfg, ckpt[cfg.checkpoint_key], device, trainable=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.control_lr)
    # The rollout baseline is used directly: warm-up ended long before the
    # checkpoint epoch (train.py --resume would leave WarmupBaseline at alpha=0).
    baseline = RolloutBaseline(model, TSP, opts, rollout)
    resume = run / 'resume.pt'
    if resume.exists():
        state = load_pt(resume)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        baseline.load_state_dict(state['baseline'])
        epoch, step, elapsed, history = state['epoch'], state['step'], state['elapsed'], state['history']
        best_val, best_state = state['best_val'], state['best_state']
        print(f'resume control s{seed} at epoch {epoch}, {elapsed / 60:.1f} of '
              f'{limit / 60:.1f} min used', flush=True)
    else:
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'baseline' in ckpt:
            baseline.load_state_dict(ckpt['baseline'])
        start_epoch = int(Path(cfg.checkpoint).stem.split('-')[-1]) + 1 \
            if Path(cfg.checkpoint).stem.startswith('epoch-') else 0
        epoch, step, elapsed = start_epoch, 0, 0.0
        val0 = float(greedy_eval(model, val_coords, device, cfg.eval_batch_size)[0].mean())
        history = [dict(step=0, epoch=epoch, val_cost=val0, seconds=0.0, baseline_updated='')]
        best_val, best_state = val0, copy.deepcopy(model.state_dict())
    for group in optimizer.param_groups:
        group['lr'] = cfg.control_lr
    session = time.time()

    def used():
        return elapsed + time.time() - session

    def evaluate(tag=''):
        nonlocal best_val, best_state
        val = float(greedy_eval(model, val_coords, device, cfg.eval_batch_size)[0].mean())
        history.append(dict(step=step, epoch=epoch, val_cost=val, seconds=used(),
                            baseline_updated=tag))
        if val < best_val:
            best_val, best_state = val, copy.deepcopy(model.state_dict())
        model.train()
        model.set_decode_type('sampling')

    finished = used() >= limit
    while not finished:
        gen = torch.Generator().manual_seed(cfg.data_seed + 7919 * int(seed) + epoch)
        data = CoordinateSet(torch.rand(cfg.control_epoch_size, cfg.graph_size, 2, generator=gen))
        loader = DataLoader(baseline.wrap_dataset(data), batch_size=cfg.control_batch_size,
                            shuffle=False, num_workers=0)
        model.train()
        model.set_decode_type('sampling')
        for batch_id, batch in enumerate(loader):
            train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, None, opts)
            step += 1
            if step % cfg.control_eval_every == 0:
                evaluate()
            if used() >= limit:
                finished = True
                break
        if finished:
            break
        updated = bool(baseline.epoch_callback(model, epoch))
        evaluate('yes' if updated else 'no')
        epoch += 1
        save_torch(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                baseline=baseline.state_dict(), epoch=epoch, step=step,
                                elapsed=used(), history=history, best_val=best_val,
                                best_state=best_state))
        print(f'control s{seed} epoch {epoch - 1} done: val {history[-1]["val_cost"]:.5f} '
              f'(best {best_val:.5f}), {used() / 60:.1f}/{limit / 60:.1f} min', flush=True)
    evaluate('final')
    save_torch(run / 'best.pt', dict(model=best_state, val_cost=best_val))
    write_csv(run / 'history.csv', history)
    done = dict(seed=seed, best_val=best_val, initial_val=history[0]['val_cost'], steps=step,
                instances=step * cfg.control_batch_size, elapsed_seconds=used(), **budget)
    save_json(run / 'done.json', clean(done))
    if resume.exists():
        resume.unlink()
    print(f'control s{seed}: best val {best_val:.5f} (start {done["initial_val"]:.5f}) after '
          f'{done["instances"]:,} graphs in {done["elapsed_seconds"] / 60:.1f} min', flush=True)
    return done


# --------------------------------------------------------------------------- #
# Evaluation, statistics and decision                                          #
# --------------------------------------------------------------------------- #

def paired(a, b):
    """Mean of a - b with per-graph SE and a normal 95% interval."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
    mean = float(d.mean())
    return dict(mean=mean, se=se, lo=mean - 1.96 * se, hi=mean + 1.96 * se)


def retention_ci(g0, teacher, student, reps=2000, seed=0):
    """Retention (mean(g0 - s) / mean(g0 - t)) with a paired bootstrap interval."""
    g0, teacher, student = (np.asarray(v, dtype=np.float64) for v in (g0, teacher, student))
    gain_t, gain_s = g0 - teacher, g0 - student
    point = float(gain_s.mean() / gain_t.mean()) if gain_t.mean() > 0 else float('nan')
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        idx = rng.integers(0, len(g0), len(g0))
        denom = gain_t[idx].mean()
        if denom > 0:
            draws.append(gain_s[idx].mean() / denom)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (float('nan'), float('nan')))
    return dict(retention=point, lo=float(lo), hi=float(hi))


@torch.no_grad()
def absorption(model, coords, traj, stage1_argmax, device):
    """Along the teacher's test trajectories: share of correction steps (teacher
    action != Stage 1 argmax) where this policy now picks the teacher's action,
    and the share of agreement steps it keeps."""
    logp = batched_logp(model, coords, traj, device)
    pick = logp.argmax(-1)[:, :-1]
    teacher = traj[:, :-1]
    correction = stage1_argmax[:, :-1] != teacher
    agree = pick == teacher
    return dict(correction_steps=int(correction.sum()),
                fix_rate=float(agree[correction].float().mean()) if correction.any() else float('nan'),
                keep_rate=float(agree[~correction].float().mean()))


def completed_seeds(output, cfg):
    """Seeds with any finished student or control run (Screen + Confirm together)."""
    seeds = set(cfg.seeds)
    for folder in list((output / 'students').glob('s*')) + list((output / 'control').glob('s*')):
        if folder.name[1:].isdigit():
            seeds.add(int(folder.name[1:]))
    return sorted(seeds)


def evaluate_all(frozen, ckpt, cfg, output, device):
    test = load_teacher(output, cfg, 'test')
    coords = torch.from_numpy(test['coords'])
    traj = torch.from_numpy(test['mcts_tour'].astype(np.int64))
    g0, _g0_tours = greedy_eval(frozen, coords, device, cfg.eval_batch_size)
    drift = np.abs(g0.numpy() - test['greedy_cost'])
    save_json(output / 'results' / 'stage1_recompute.json', clean(dict(
        max_abs_diff=float(drift.max()), graphs_differing=int((drift > 1e-4).sum()))))
    if (drift > 1e-4).any():
        print(f'WARNING: Stage 1 greedy differs from the teacher-phase record on '
              f'{int((drift > 1e-4).sum())} test graphs (device numerics); all comparisons '
              f'use the recomputed values.', flush=True)
    stage1_argmax = batched_logp(frozen, coords, traj, device).argmax(-1)
    canonical = split_coordinates(cfg, 'canonical') if cfg.canonical_val_instances else None
    costs = dict(stage1=g0.numpy(), teacher=test['mcts_cost'])
    rows = []

    def add_policy(name, kind, model, seed=None, target=None, info=None):
        cost = greedy_eval(model, coords, device, cfg.eval_batch_size)[0].numpy()
        costs[name] = cost
        row = dict(policy=name, kind=kind, seed=seed, target=target, test_mean=float(cost.mean()),
                   canonical_mean=(float(greedy_eval(model, canonical, device,
                                                     cfg.eval_batch_size)[0].mean())
                                   if canonical is not None else None))
        row.update(absorption(model, coords, traj, stage1_argmax, device))
        row.update(info or {})
        rows.append(row)

    add_policy('stage1', 'reference', frozen)
    rows[-1]['val_cost'] = float(greedy_eval(frozen, split_coordinates(cfg, 'val'), device,
                                             cfg.eval_batch_size)[0].mean())
    for seed in completed_seeds(output, cfg):
        for target_name in TARGETS:
            chosen = select_student(output, cfg, seed, target_name)
            if chosen is None:
                continue
            state = load_pt(student_dir(output, seed, target_name, chosen['lr']) / 'best.pt')
            model = build_model(cfg, state['model'], device, trainable=False)
            add_policy(f'{target_name}_s{seed}', 'student', model, seed, target_name,
                       dict(lr=chosen['lr'], val_cost=chosen['best_val'],
                            best_step=chosen['best_step'], train_seconds=chosen['wall_seconds']))
        path = control_dir(output, seed) / 'done.json'
        if path.exists():
            done = json.loads(path.read_text())
            state = load_pt(control_dir(output, seed) / 'best.pt')
            model = build_model(cfg, state['model'], device, trainable=False)
            add_policy(f'control_s{seed}', 'control', model, seed, None,
                       dict(val_cost=done['best_val'], train_seconds=done['elapsed_seconds'],
                            budget_seconds=done['budget_seconds'],
                            control_instances=done['instances']))
    teacher_row = dict(policy='teacher', kind='reference', test_mean=float(test['mcts_cost'].mean()))
    rows.insert(1, teacher_row)
    save_npz(output / 'results' / 'test_costs.npz', **costs)
    write_csv(output / 'results' / 'policies.csv', [clean(r) for r in rows])
    return rows, costs


def decide(costs, rows, min_retention):
    """Screen decision from per-graph test costs. Pure function (unit-tested)."""
    g0, teacher = costs['stage1'], costs['teacher']
    gain_t = paired(g0, teacher)
    records = []
    for row in rows:
        if row['kind'] != 'student':
            continue
        seed, name = row['seed'], row['policy']
        control = costs.get(f'control_s{seed}')
        vs_stage1 = paired(costs[name], g0)
        vs_control = paired(costs[name], control) if control is not None else None
        ret = retention_ci(g0, teacher, costs[name])
        beats_control = bool(vs_control is not None and vs_control['hi'] < 0)
        records.append(dict(policy=name, seed=seed, target=row['target'], lr=row.get('lr'),
                            val_cost=row.get('val_cost'),
                            student_minus_stage1=vs_stage1['mean'],
                            student_minus_stage1_se=vs_stage1['se'],
                            student_minus_control=None if vs_control is None else vs_control['mean'],
                            student_minus_control_lo=None if vs_control is None else vs_control['lo'],
                            student_minus_control_hi=None if vs_control is None else vs_control['hi'],
                            retention=ret['retention'], retention_lo=ret['lo'],
                            retention_hi=ret['hi'], beats_control=beats_control,
                            passes=beats_control and ret['retention'] >= min_retention))
    controls = {}
    for row in rows:
        if row['kind'] == 'control':
            c = paired(costs[row['policy']], g0)
            controls[row['policy']] = dict(control_minus_stage1=c['mean'], se=c['se'])
    screen = [r for r in records if r['seed'] == 0]
    passing = [r for r in screen if r['passes']]
    if not screen:
        outcome, chosen = 'INCOMPLETE', None
    elif passing:
        outcome = 'CONTINUE'
        chosen = min(passing, key=lambda r: r['val_cost'])['target']
    else:
        outcome, chosen = 'STOP', None
    confirm = {}
    if chosen is not None:
        for target_name in sorted({chosen, 'visits'}):
            later = [r for r in records if r['target'] == target_name and r['seed'] != 0]
            if later:
                confirm[target_name] = dict(
                    seeds=sorted(r['seed'] for r in later),
                    all_beat_control=all(r['beats_control'] for r in later),
                    mean_retention=float(np.mean([r['retention'] for r in later])))
    return dict(teacher_gain=gain_t['mean'], teacher_gain_se=gain_t['se'],
                min_retention=min_retention, screen_outcome=outcome, continue_with=chosen,
                students=records, controls=controls, confirm=confirm)


def report(cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with np.load(output / 'results' / 'test_costs.npz') as z:
        costs = {k: z[k] for k in z.files}
    with (output / 'results' / 'policies.csv').open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['seed'] = int(row['seed']) if row.get('seed') not in (None, '') else None
        for key in ('val_cost', 'lr'):
            if row.get(key) not in (None, ''):
                row[key] = float(row[key])
    decision = decide(costs, rows, cfg.min_retention)
    save_json(output / 'results' / 'decision.json', clean(decision))
    write_csv(output / 'results' / 'students.csv', [clean(r) for r in decision['students']])

    # Plot 1: validation curves (students by target/lr; control by wall time).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for history in sorted((output / 'students').glob('s*/*/lr*/history.csv')):
        with history.open() as f:
            hist = list(csv.DictReader(f))
        seed, target_name, lr = history.parts[-4], history.parts[-3], history.parts[-2]
        axes[0].plot([float(h['step']) for h in hist], [float(h['val_cost']) for h in hist],
                     label=f'{target_name} {lr} {seed}', alpha=.8)
    axes[0].set(xlabel='Optimizer step', ylabel='Validation greedy cost', title='Students')
    axes[0].legend(fontsize=7, ncol=2)
    for history in sorted((output / 'control').glob('s*/history.csv')):
        with history.open() as f:
            hist = list(csv.DictReader(f))
        axes[1].plot([float(h['seconds']) / 60 for h in hist], [float(h['val_cost']) for h in hist],
                     marker='.', label=f'control {history.parts[-2]}')
    axes[1].set(xlabel='Wall minutes', ylabel='Validation greedy cost',
                title='REINFORCE control (matched wall time)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'results' / 'training_curves.png', dpi=160)
    plt.close(fig)

    # Plot 2: test cost by policy with paired 95% intervals against Stage 1.
    names = [r['policy'] for r in rows if r['policy'] in costs]
    stats = [paired(costs[n], costs['stage1']) for n in names]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(names)), 4.5))
    ax.errorbar(range(len(names)), [s['mean'] for s in stats],
                yerr=[1.96 * (s['se'] if math.isfinite(s['se']) else 0) for s in stats],
                fmt='o', capsize=3)
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
    ax.set(ylabel='Test cost minus Stage 1 greedy (lower is better)',
           title='Paired per-graph differences, 95% intervals')
    fig.tight_layout()
    fig.savefig(output / 'results' / 'test_deltas.png', dpi=160)
    plt.close(fig)
    (output / 'INTERPRETATION.md').write_text(INTERPRETATION.format(outcome=outcome_summary(decision)))
    return decision


def outcome_summary(decision):
    """Markdown lines naming the outcome and which criterion each target met."""
    floor = decision['min_retention']
    chosen = decision['continue_with']
    lines = [f"**Screen outcome: {decision['screen_outcome']}**"
             + (f" (continue with `{chosen}`)." if chosen else '.'),
             f"Teacher gain over Stage 1 greedy: {decision['teacher_gain']:+.4f} "
             f"(SE {decision['teacher_gain_se']:.4f}); retention floor {floor:g}.", '']
    for r in decision['students']:
        if r['student_minus_control'] is None:
            vs = 'no control for this seed'
        else:
            vs = (f"student - control {r['student_minus_control']:+.4f} "
                  f"[{r['student_minus_control_lo']:+.4f}, {r['student_minus_control_hi']:+.4f}], "
                  + ('beats the control' if r['beats_control'] else 'does not beat the control'))
        lines.append(f"- seed {r['seed']} `{r['target']}`: {vs}; retention {r['retention']:.3f} "
                     f"[{r['retention_lo']:.3f}, {r['retention_hi']:.3f}], "
                     + ('meets the floor.' if r['retention'] >= floor else 'below the floor.'))
    for name, c in decision['confirm'].items():
        lines.append(f"- confirm `{name}` seeds {c['seeds']}: all beat their control = "
                     f"{c['all_beat_control']}; mean retention {c['mean_retention']:.3f}.")
    return '\n'.join(lines)


INTERPRETATION = """# Student isolation (Stage 5 §I Step 2): how to read the results

{outcome}

`results/decision.json` holds the screen outcome. A target PASSES when its test
greedy cost beats the matched-time REINFORCE control with a paired 95% interval
below zero AND it keeps at least `min_retention` of the teacher's gain over
Stage 1 greedy (retention = (Stage 1 - student) / (Stage 1 - teacher)).
CONTINUE names the passing target with the best validation cost; run the
`confirm` profile next (seeds 1 and 2: new training graphs, same val/test).
STOP means no screen target met both criteria; the lines above (and
`beats_control` / `retention` in `results/decision.json`) say which one failed.

`results/policies.csv` lists every policy on the test graphs (and on the
canonical 10K seed-42 set when enabled). `fix_rate` is the share of the
teacher's corrections to Stage 1 (steps where the MCTS action differs from the
Stage 1 argmax, along the teacher's test trajectories) that the policy now
makes; `keep_rate` is its agreement with the teacher elsewhere. Read the two
together: correction steps are near-ties for Stage 1 (it puts ~0.23 on the
teacher's move there), so any perturbed policy flips some of them; a real
absorption raises fix_rate while keeping keep_rate near 1. Students and the
control were selected on the validation graphs, never on test; validation
numbers are therefore optimistic, test numbers are not.

Limits: one fixed teacher and one round of distillation; the iterated loop can
compound gains this test does not see. Wall times are Colab-runtime specific.
Intervals are per seed; check that seeds agree before claiming anything.
"""


# --------------------------------------------------------------------------- #
# Correctness checks                                                           #
# --------------------------------------------------------------------------- #

def check_invariants(frozen, cfg, output, device):
    report_ = {}
    n_small = min(cfg.graph_size, 10)
    x = torch.rand(3, n_small, 2, generator=torch.Generator().manual_seed(cfg.data_seed + 5))
    # 1. C++ root statistics exist and match the Python reference search. Both run
    #    on a CPU copy: this checks the C++ code path, not GPU batching numerics.
    cpu = torch.device('cpu')
    cpu_model = copy.deepcopy(frozen).to(cpu)
    mcfg = teacher_config(cfg, K=12)
    batch = cpp_solver.CppBatchMCTSSolver(cpu_model, mcfg, device=cpu, mcts_batch_size=2)
    bl = greedy_eval(cpu_model, x, cpu)[0]
    _costs, tours = batch.solve_batch(x, bl_vals=bl)
    worst = 0.0
    for i in range(len(x)):
        ref = MCTSSolver(cpu_model, mcfg, device=cpu)
        _c, tour = ref.solve_instance(x[i:i + 1], bl_val=float(bl[i]))
        if tour.tolist() != tours[i].tolist():
            raise AssertionError('C++ batched teacher tour differs from the Python reference')
        if ref.root_visit_dists != batch.root_visit_dists_per_instance[i]:
            raise AssertionError('C++ batched visit counts differ from the Python reference')
        for d0, d1 in zip(ref.root_q_dists, batch.root_q_dists_per_instance[i]):
            worst = max(worst, max(abs(d0[a] - d1[a]) for a in d0))
        worst = max(worst, max(abs(a - b) for a, b in zip(ref.root_values,
                                                         batch.root_values_per_instance[i])))
    if worst > 1e-4:
        raise AssertionError(f'root Q / value mismatch {worst:.2e}')
    report_['cpp_python_root_q_max_abs_diff'] = worst
    # 2. Teacher forcing reproduces the policy's own greedy decisions/likelihood.
    with torch.no_grad():
        frozen.set_decode_type('greedy')
        _cost, ll, pi = frozen(x.to(device), return_pi=True)
        logp = teacher_forced_logp(frozen, x.to(device), pi)
    chosen = logp.gather(-1, pi.unsqueeze(-1)).squeeze(-1).sum(-1)
    if not torch.equal(logp.argmax(-1), pi) or (chosen - ll).abs().max() > 1e-3:
        raise AssertionError('teacher forcing does not reproduce the greedy decode')
    report_['teacher_forcing_ll_max_abs_diff'] = float((chosen - ll).abs().max())
    # 3. Targets on a tiny labelled block are proper distributions on legal moves.
    record = label_graphs(frozen, cfg, x, device, K=8, batch=3)
    data = dict(record, st_inst=record['st_inst'].astype(np.int64),
                greedy_seconds=np.atleast_1d(record['greedy_seconds']),
                mcts_seconds=np.atleast_1d(record['mcts_seconds']))
    small = SimpleNamespace(**dict(asdict(cfg), graph_size=n_small))
    for name in TARGETS:
        built = build_targets(frozen, small, data, name, device)
        legal = legal_masks(built['traj'])
        if built['target'] is not None:
            t = built['target']
            if not torch.allclose(t.sum(-1), torch.ones(t.shape[:2]), atol=1e-5):
                raise AssertionError(f'{name} targets do not sum to one')
            if (t[~legal] != 0).any() or (t < 0).any() or not torch.isfinite(t).all():
                raise AssertionError(f'{name} targets put mass on illegal moves')
        elif not torch.equal(built['traj'].sort(-1).values,
                             torch.arange(n_small).expand(len(x), -1)):
            raise AssertionError('best tours are not permutations')
    tours_, used = best_tours(data)
    expect = np.minimum(data['greedy_cost'], data['mcts_cost'])
    got = np.where(used, data['mcts_cost'], data['greedy_cost'])
    if not np.allclose(expect, got):
        raise AssertionError('best_tour does not pick the cheaper tour')
    report_['targets_valid'] = True
    # 4. Canonical set reproduces the TSPDataset draw.
    if cfg.canonical_val_instances:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.canonical_val_seed)
            first = TSP.make_dataset(size=cfg.graph_size, num_samples=2).data[0]
        if not torch.equal(first, split_coordinates(
                SimpleNamespace(**dict(asdict(cfg), canonical_val_instances=1)), 'canonical')[0]):
            raise AssertionError('canonical set does not reproduce TSP.make_dataset')
        report_['canonical_set_matches_make_dataset'] = True
    save_json(output / 'invariants.json', clean(report_))
    print(json.dumps(clean(report_), indent=2), flush=True)
    return report_


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

PHASES = ('check', 'teacher', 'students', 'control', 'evaluate', 'report', 'all')


def run_phase(cfg, phase):
    """Notebook/CLI entry point. Every phase resumes from artifacts on disk."""
    if phase not in PHASES:
        raise ValueError(f'unknown phase {phase}; choose from {PHASES}')
    frozen, ckpt, output, device = open_run(cfg)
    before = policy_digest(frozen)
    result = None
    try:
        if phase in ('check', 'all'):
            result = check_invariants(frozen, cfg, output, device)
        if phase in ('teacher', 'all'):
            for split in ('test',):
                result = teacher_split(frozen, cfg, output, device, split)
            for seed in cfg.seeds:
                result = teacher_split(frozen, cfg, output, device, 'train', seed)
        if phase in ('students', 'all'):
            run_students(frozen, ckpt, cfg, output, device)
        if phase in ('control', 'all') and cfg.control:
            for seed in cfg.seeds:
                result = run_control(cfg, ckpt, output, device, seed)
        if phase in ('evaluate', 'all'):
            result = evaluate_all(frozen, ckpt, cfg, output, device)
        if phase in ('report', 'all'):
            result = report(cfg, output)
    finally:
        if policy_digest(frozen) != before or any(p.requires_grad for p in frozen.parameters()):
            raise AssertionError('The frozen teacher policy changed!')
        with (output / 'freeze_checks.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=phase, unchanged=True, time=time.time())) + '\n')
    print(f'{phase} complete -> {output}', flush=True)
    return result
