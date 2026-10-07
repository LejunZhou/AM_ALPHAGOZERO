"""Frozen-policy value-input ablation used by colab_value_evaluator_repair.ipynb.

All refitted heads predict RAW remaining greedy-policy cost, including closing
the tour. Exact optimal completions are evaluation oracles, never train labels.
Production decoder, value head, and search implementations are not modified.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
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
from torch import nn

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.state import StateTSP
from am_baseline.search.mcts import MCTSConfig, MCTSSolver

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(**kwargs):
        return lambda f: f


SOURCE_FILES = (
    'model/attention_model.py', 'model/encoder.py', 'model/decoder.py',
    'model/value_head.py', 'problem/state.py', 'problem/tsp.py',
    'utils/tensor_ops.py', 'search/mcts.py', 'search/puct.py', 'search/tree.py',
    'experiments/value_repair.py',
)
VARIANTS = ('original', 'original_wide', 'repaired', 'repaired_geo')
GEO_FEATURES = 14  # see geometry_features()


def base_feature_dim(embedding_dim):
    """Columns used by the 'repaired' head: glimpse + first/current/pooled
    embeddings + 8 scalars (coordinates, centroid, remaining fraction, start flag)."""
    return 4 * embedding_dim + 8


@dataclass
class ExperimentConfig:
    checkpoint: str
    output_dir: str
    original_value_norm: str  # explicitly 'none' for F.6.1.6; no silent default
    checkpoint_key: str = 'auto'
    graph_size: int = 20
    train_instances: int = 512
    val_instances: int = 64
    probe_instances: int = 64
    calibration_instances: int = 4
    search_instances: int = 48
    data_seed: int = 20260926
    head_seeds: tuple = (0, 1, 2)
    epochs: int = 30
    batch_size: int = 512
    feature_batch_size: int = 32
    train_parent_steps: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    prefix_temperature: float = 3.0
    prefix_fraction: float = 0.5
    oracle_max_remaining: int = 20
    search_K: int = 40
    c_puct: float = 0.05
    timing_repeats: int = 2
    max_time_K_multiplier: int = 4
    device: str = 'auto'
    search_device: str = 'cpu'
    cpu_threads: int = 2

    def validate(self):
        if self.original_value_norm not in ('none', 'bl', 'sqrt_n'):
            raise ValueError('Specify the ORIGINAL checkpoint output units: none/bl/sqrt_n.')
        if not 4 <= self.graph_size <= 50 or not 1 <= self.oracle_max_remaining <= 20:
            raise ValueError('Supported graph sizes: 4..50; exact DP cap: 1..20.')
        for name in ('train_instances', 'val_instances', 'probe_instances',
                     'calibration_instances', 'search_instances', 'epochs',
                     'batch_size', 'feature_batch_size', 'train_parent_steps',
                     'hidden_dim', 'search_K', 'timing_repeats', 'cpu_threads'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if not self.head_seeds or len(set(self.head_seeds)) != len(self.head_seeds):
            raise ValueError('head_seeds must be nonempty and distinct')
        if self.prefix_temperature <= 0 or not 0 <= self.prefix_fraction <= 1:
            raise ValueError('Invalid prefix sampling settings')
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_time_K_multiplier < 1:
            raise ValueError('Invalid optimizer or time-budget settings')


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
    """Includes frozen parameters AND batch-normalization buffers."""
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


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_pt(path):
    # These are the user's own model/cache artifacts, not third-party uploads.
    return torch.load(path, map_location='cpu', weights_only=False)


def load_policy(cfg, device):
    path = Path(cfg.checkpoint)
    args_path = path.parent / 'args.json'
    args = json.loads(args_path.read_text()) if args_path.exists() else {}
    if args.get('value_target_norm', cfg.original_value_norm) != cfg.original_value_norm:
        raise ValueError('original_value_norm disagrees with checkpoint args.json')
    arch = dict(embedding_dim=128, n_encode_layers=3, n_heads=8,
                tanh_clipping=10., normalization='batch', feed_forward_hidden=512,
                value_enabled=True, value_hidden_dim=128)
    arch.update({k: args[k] for k in arch if k in args})
    ckpt = load_pt(path)
    key = cfg.checkpoint_key
    if key == 'auto':
        key = 'best_model' if 'best_model' in ckpt else 'model'
    if key not in ckpt:
        raise ValueError(f'Checkpoint does not contain {key!r}; specify checkpoint_key.')
    model = AttentionModel(SimpleNamespace(**arch))
    model.load_state_dict(ckpt[key], strict=True)
    if model.value_head is None:
        raise ValueError('This comparison requires a checkpoint with an existing value head.')
    model.to(device).eval().requires_grad_(False)
    model.set_decode_type('greedy')
    return model, dict(architecture=arch, key=key,
                       args_json_present=args_path.exists(),
                       original_value_norm=cfg.original_value_norm)


def open_run(cfg):
    cfg.validate()
    torch.set_num_threads(cfg.cpu_threads)
    device = torch.device('cuda' if cfg.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if cfg.device == 'auto' else cfg.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    model, info = load_policy(cfg, device)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = asdict(cfg)
    protocol.pop('checkpoint')
    protocol.pop('output_dir')
    # JSON round-trip normalizes tuple/list representations on resume.
    protocol = json.loads(json.dumps(protocol))
    signature = dict(protocol=protocol, checkpoint_sha256=sha256(cfg.checkpoint),
                     source_sha256=source_digest(), policy_sha256=policy_digest(model),
                     checkpoint_info=info)
    manifest = output / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['signature'] != signature:
            raise ValueError('Run manifest differs. Use a NEW output_dir for a new protocol, '
                             'checkpoint, or source version; existing results are preserved.')
    else:
        save_json(manifest, dict(signature=signature, checkpoint_path=str(cfg.checkpoint),
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    with (output / 'sessions.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), device=str(device), torch=torch.__version__,
                numpy=np.__version__, python=platform.python_version(), platform=platform.platform(),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)) + '\n')
    return model, output, device


@njit(cache=True)
def _geometry_rows(loc, legal, first, current, has_start, out):
    """Per-row tour geometry of the remaining sub-problem (float64):
    0  MST(unvisited) + nearest link from current + nearest link to first
       (equals the `mst` control evaluator; closing edge when nothing remains)
    1-3  three smallest distances current -> unvisited (repeat last if fewer)
    4-6  three smallest distances first -> unvisited
    7  distance current -> first
    8-9  std of unvisited x / y;  10-11  bounding-box extent x / y
    12-13  mean distance current -> unvisited, first -> unvisited"""
    B, N = legal.shape
    for b in range(B):
        m = 0
        for j in range(N):
            if legal[b, j]:
                m += 1
        c = current[b]
        f = first[b]
        cx = loc[b, c, 0]
        cy = loc[b, c, 1]
        fx = loc[b, f, 0]
        fy = loc[b, f, 1]
        dcf = math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2) if has_start[b] else 0.0
        for q in range(out.shape[1]):
            out[b, q] = 0.0
        if m == 0:
            out[b, 0] = dcf
            out[b, 7] = dcf
            continue
        idx = np.empty(m, np.int64)
        k = 0
        for j in range(N):
            if legal[b, j]:
                idx[k] = j
                k += 1
        dc = np.empty(m)
        df = np.empty(m)
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        minx = 1e300
        maxx = -1e300
        miny = 1e300
        maxy = -1e300
        for k in range(m):
            x = loc[b, idx[k], 0]
            y = loc[b, idx[k], 1]
            dc[k] = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            df[k] = math.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            sx += x
            sy += y
            sxx += x * x
            syy += y * y
            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)
        in_tree = np.zeros(m, np.bool_)
        key = np.full(m, np.inf)
        key[0] = 0.0
        mst = 0.0
        for _ in range(m):
            j = -1
            best = np.inf
            for k in range(m):
                if not in_tree[k] and key[k] < best:
                    best = key[k]
                    j = k
            in_tree[j] = True
            mst += best
            jx = loc[b, idx[j], 0]
            jy = loc[b, idx[j], 1]
            for k in range(m):
                if not in_tree[k]:
                    d = math.sqrt((loc[b, idx[k], 0] - jx) ** 2 + (loc[b, idx[k], 1] - jy) ** 2)
                    if d < key[k]:
                        key[k] = d
        dc_sorted = np.sort(dc)
        df_sorted = np.sort(df)
        bound = mst
        if has_start[b]:
            bound += dc_sorted[0] + df_sorted[0]
        out[b, 0] = bound
        for q in range(3):
            out[b, 1 + q] = dc_sorted[min(q, m - 1)]
            out[b, 4 + q] = df_sorted[min(q, m - 1)]
        out[b, 7] = dcf
        mx = sx / m
        my = sy / m
        out[b, 8] = math.sqrt(max(sxx / m - mx * mx, 0.0))
        out[b, 9] = math.sqrt(max(syy / m - my * my, 0.0))
        out[b, 10] = maxx - minx
        out[b, 11] = maxy - miny
        out[b, 12] = dc.sum() / m
        out[b, 13] = df.sum() / m


@torch.no_grad()
def geometry_features(state):
    """(B, GEO_FEATURES) float64->model dtype tensor of remaining-tour geometry."""
    ids = state.ids.view(-1)
    b = ids.numel()
    loc = state.loc[ids].detach().cpu().double().numpy()
    legal = (~state.get_mask().squeeze(1)).cpu().numpy()
    first = state.first_a.view(-1).cpu().numpy().astype(np.int64)
    current = state.prev_a.view(-1).cpu().numpy().astype(np.int64)
    has_start = (state.i.reshape(-1) > 0).expand(b).cpu().numpy()
    out = np.zeros((b, GEO_FEATURES), dtype=np.float64)
    _geometry_rows(np.ascontiguousarray(loc), np.ascontiguousarray(legal), first, current,
                   np.ascontiguousarray(has_start), out)
    return torch.from_numpy(out)


@torch.no_grad()
def state_features(fixed, state, glimpse):
    """[glimpse | first/current/pooled embeddings + 8 scalars | 14 geometry features].

    Columns [0, d) feed the 'original' heads, [0, 4d+8) the 'repaired' head
    (explicit endpoint coordinates plus remaining centroid remove the known
    one-city alias even if encoder embeddings happen to collide), and all
    columns the 'repaired_geo' head, which also predicts a residual over the
    MST bound in column 4d+8. Pooling is NOT claimed to be an injective
    representation of arbitrary remaining sets.
    """
    emb = fixed.node_embeddings
    b, n, d = emb.shape
    rows = torch.arange(b, device=emb.device)
    legal = ~state.get_mask().squeeze(1)
    count = legal.sum(1, keepdim=True)
    has_start = (state.i.reshape(-1, 1) > 0).expand(b, 1).to(emb.dtype)
    first = state.first_a.view(-1)
    current = state.prev_a.view(-1)
    loc = state.loc[state.ids.view(-1)]
    pool = (emb * legal.unsqueeze(-1)).sum(1) / count.clamp_min(1)
    centroid = (loc * legal.unsqueeze(-1)).sum(1) / count.clamp_min(1)
    geometry = geometry_features(state).to(device=emb.device, dtype=emb.dtype)
    return torch.cat((glimpse.reshape(b, d), emb[rows, first] * has_start,
                      emb[rows, current] * has_start, pool,
                      loc[rows, first] * has_start, loc[rows, current] * has_start,
                      centroid, count / float(n), has_start, geometry), dim=1)


@torch.no_grad()
def greedy_remaining(model, fixed, state):
    start = state.lengths.view(-1)
    cur = state
    while not cur.all_finished():
        logp, _ = model.decode_step(fixed, cur)
        cur = cur.update(logp.squeeze(1).argmax(-1))
    return cur.get_final_cost().view(-1) - start


@njit(cache=True)
def _dp_children(dist, m):
    table = np.full((1 << m, m), np.inf)
    for j in range(m):
        table[1 << j, j] = dist[j, m]
    for mask in range(1, 1 << m):
        for j in range(m):
            if mask & (1 << j) == 0 or mask == 1 << j:
                continue
            rest = mask ^ (1 << j)
            best = np.inf
            for k in range(m):
                if rest & (1 << k):
                    best = min(best, dist[j, k] + table[rest, k])
            table[mask, j] = best
    return table[(1 << m) - 1].copy()


def exact_child_remaining(loc, legal, first):
    """Cost after choosing each child: visit other unvisited cities, close to first."""
    m = len(legal)
    if m > 20 or (not HAVE_NUMBA and m > 10):
        raise RuntimeError('Exact oracle requires numba for >10 remaining cities; install numba.')
    points = np.asarray(loc, dtype=np.float64)[list(legal) + [int(first)]]
    dist = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return _dp_children(np.ascontiguousarray(dist), m)


def split_coordinates(cfg, split):
    offsets = dict(train=11, val=23, probe=37, calibration=53, search=71)
    count = getattr(cfg, split + '_instances')
    return torch.rand(count, cfg.graph_size, 2,
                      generator=torch.Generator().manual_seed(cfg.data_seed + offsets[split]))


@torch.no_grad()
def collect_batch(model, coords, cfg, split, offset):
    """Enumerate ALL children at each chosen parent; frozen batched rollouts."""
    device = next(model.parameters()).device
    coords = coords.to(device)
    b, n, _ = coords.shape
    fixed = model.precompute_decoder(model.encode(coords))
    baseline = greedy_remaining(model, fixed, StateTSP.initialize(coords))
    if split == 'probe':
        steps = set(range(1, n - 1))  # exclude symmetric empty root and forced final move
    else:
        steps = set(np.linspace(0, n - 2, min(cfg.train_parent_steps, n - 1), dtype=int))
        steps.add(1)
    result = {}
    def append(key, val):
        result.setdefault(key, []).append(val.detach().cpu())
    for source in (0, 1):
        state = StateTSP.initialize(coords)
        generator = torch.Generator().manual_seed(cfg.data_seed + 101 * offset +
                                                  997 * source + {'train': 1, 'val': 2, 'probe': 3}[split])
        for step in range(n - 1):
            logp, _ = model.decode_step(fixed, state)
            if step in steps:
                parent_ids, actions = (~state.get_mask().squeeze(1)).nonzero(as_tuple=True)
                children = state[parent_ids].update(actions)
                child_fixed = fixed[parent_ids]
                _, _, glimpse = model.decode_step(child_fixed, children, return_glimpse=True)
                features = state_features(child_fixed, children, glimpse)
                targets = greedy_remaining(model, child_fixed, children)
                size = len(actions)
                append('x', features)
                append('y', targets)
                append('weight', torch.full((size,), 1.0 / (n - step), device=device))
                append('instance', parent_ids + offset)
                append('source', torch.full_like(actions, source))
                append('step', torch.full_like(actions, step))
                append('action', actions)
                append('prior', logp.squeeze(1)[parent_ids, actions].exp())
                append('path', children.lengths.view(-1))
                append('baseline', baseline[parent_ids])
                if split == 'probe':
                    oracle = np.full(size, np.nan, dtype=np.float64)
                    if n - step <= cfg.oracle_max_remaining:
                        loc_cpu = coords.cpu().numpy()
                        mask_cpu = state.get_mask().squeeze(1).cpu().numpy()
                        firsts = state.first_a.view(-1).cpu().numpy()
                        currents = state.prev_a.view(-1).cpu().numpy()
                        paths = state.lengths.view(-1).cpu().numpy()
                        for row in range(b):
                            legal = np.flatnonzero(~mask_cpu[row])
                            rem = exact_child_remaining(loc_cpu[row], legal, firsts[row])
                            edge = np.linalg.norm(loc_cpu[row, legal].astype(np.float64) -
                                                  loc_cpu[row, currents[row]].astype(np.float64), axis=-1)
                            oracle[row * len(legal):(row + 1) * len(legal)] = paths[row] + edge + rem
                    append('oracle', torch.from_numpy(oracle))
            if source == 1 and step < math.ceil(cfg.prefix_fraction * n):
                probs = (logp.squeeze(1).cpu() / cfg.prefix_temperature).softmax(-1)
                action = torch.multinomial(probs, 1, generator=generator).squeeze(1).to(device)
            else:
                action = logp.squeeze(1).argmax(-1)
            state = state.update(action)
    return {k: torch.cat(v) for k, v in result.items()}


def cached_dataset(model, cfg, split, output):
    coords = split_coordinates(cfg, split)
    directory = output / 'data' / split
    directory.mkdir(parents=True, exist_ok=True)
    save_torch(directory / 'coordinates.pt', coords)
    shards = []
    for start in range(0, len(coords), cfg.feature_batch_size):
        path = directory / f'features_{start:06d}.pt'
        if not path.exists():
            t0 = time.perf_counter()
            batch = collect_batch(model, coords[start:start + cfg.feature_batch_size], cfg, split, start)
            save_torch(path, batch)
            print(f'{split}: {min(start + cfg.feature_batch_size, len(coords))}/{len(coords)} '
                  f'instances, {len(batch["y"])} children, {time.perf_counter()-t0:.1f}s', flush=True)
        shards.append(path)
    batches = [load_pt(p) for p in shards]
    return {key: torch.cat([part[key] for part in batches]) for key in batches[0]}


class RefitHead(nn.Module):
    """original / original_wide: glimpse only. repaired: glimpse + endpoint/pool
    features. repaired_geo: everything, predicting a residual over the MST bound
    (column base_dim), so its output is bound + MLP(standardized features)."""
    def __init__(self, variant, mean, std, embedding_dim, hidden_dim):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'unknown variant {variant!r}')
        self.variant = variant
        base_dim = base_feature_dim(embedding_dim)
        if variant in ('original', 'original_wide'):
            self.input_dim = embedding_dim
        elif variant == 'repaired':
            self.input_dim = base_dim
        else:
            self.input_dim = len(mean)
        if variant == 'original_wide':  # parameter count matched to 'repaired'
            hidden_dim = round(hidden_dim * (base_dim + 2) / (embedding_dim + 2))
        self.residual_col = base_dim if variant == 'repaired_geo' else -1
        self.register_buffer('mean', mean[:self.input_dim].clone())
        self.register_buffer('std', std[:self.input_dim].clone())
        self.mlp = nn.Sequential(nn.Linear(self.input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        z = (x[..., :self.input_dim] - self.mean) / self.std
        out = self.mlp(z).squeeze(-1)
        if self.residual_col >= 0:
            out = out + x[..., self.residual_col]
        return out


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def predict_batched(head, x, device, batch=4096):
    head.eval()
    return torch.cat([head(part.to(device)).cpu() for part in x.split(batch)])


def train_heads(model, cfg, output, device):
    train = cached_dataset(model, cfg, 'train', output)
    val = cached_dataset(model, cfg, 'val', output)
    mean = train['x'].mean(0)
    std = train['x'].std(0, unbiased=False).clamp_min(1e-4)
    weights = train['weight'] / train['weight'].mean()
    metadata = dict(mean=mean, std=std, embedding_dim=model.embedding_dim,
                    hidden_dim=cfg.hidden_dim, value_target_norm='none',
                    target='frozen_greedy_remaining_including_closing_edge')
    save_torch(output / 'feature_statistics.pt', metadata)
    for seed in cfg.head_seeds:
        for variant in VARIANTS:
            name = f'{variant}_s{seed}'
            path = output / 'heads' / f'{name}.pt'
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            head = RefitHead(variant, mean, std, model.embedding_dim, cfg.hidden_dim).to(device)
            with torch.no_grad():
                if head.residual_col >= 0:
                    head.mlp[-1].bias.fill_(float((train['y'] - train['x'][:, head.residual_col]).mean()))
                else:
                    head.mlp[-1].bias.fill_(float(train['y'].mean()))
            opt = torch.optim.Adam(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            start_epoch, best_loss, history, best_state = 0, float('inf'), [], None
            if path.exists():
                saved = load_pt(path)
                start_epoch, best_loss, history = saved['epoch'], saved['best_val_mse'], saved['history']
                head.load_state_dict(saved['last_state'])
                opt.load_state_dict(saved['optimizer'])
                best_state = saved['best_state']
            count = sum(p.numel() for p in head.parameters())
            print(f'{name}: {count:,} parameters; resume epoch {start_epoch}/{cfg.epochs}', flush=True)
            for epoch in range(start_epoch, cfg.epochs):
                start = time.perf_counter()
                head.train()
                # Same permutation for every variant at this (seed, epoch), including resume.
                order = torch.randperm(len(train['y']), generator=torch.Generator().manual_seed(
                    cfg.data_seed + 100003 * seed + epoch))
                running = 0.0
                for ids in order.split(cfg.batch_size):
                    pred = head(train['x'][ids].to(device))
                    error = pred - train['y'][ids].to(device)
                    loss = (error.square() * weights[ids].to(device)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()
                    running += float(loss.detach()) * len(ids)
                prediction = predict_batched(head, val['x'], device)
                val_loss = float(((prediction - val['y']).square() * val['weight']).sum() / val['weight'].sum())
                if not math.isfinite(val_loss):
                    raise RuntimeError(f'Non-finite loss in {name}; stop rather than save invalid results.')
                if val_loss < best_loss:
                    best_loss, best_state = val_loss, cpu_state(head)
                history.append(dict(variant=variant, seed=seed, epoch=epoch + 1,
                    train_weighted_mse=running / len(order), val_weighted_mse=val_loss,
                    seconds=time.perf_counter() - start, parameters=count))
                save_torch(path, dict(**metadata, variant=variant, seed=seed,
                    epoch=epoch + 1, best_val_mse=best_loss, best_state=best_state,
                    last_state=cpu_state(head), optimizer=opt.state_dict(), history=history))
                print(f'{name} epoch {epoch+1}: val MSE={val_loss:.6f}', flush=True)
    histories = []
    for seed in cfg.head_seeds:
        for variant in VARIANTS:
            histories.extend(load_pt(output / 'heads' / f'{variant}_s{seed}.pt')['history'])
    write_csv(output / 'training_history.csv', histories)


def load_heads(cfg, output, device):
    heads = {}
    for seed in cfg.head_seeds:
        for variant in VARIANTS:
            name = f'{variant}_s{seed}'
            path = output / 'heads' / f'{name}.pt'
            saved = load_pt(path)
            if saved['epoch'] != cfg.epochs:
                raise ValueError(f'{name} training incomplete; run train before evaluating.')
            head = RefitHead(variant, saved['mean'], saved['std'], saved['embedding_dim'], saved['hidden_dim'])
            head.load_state_dict(saved['best_state'])
            heads[name] = head.to(device).eval().requires_grad_(False)
    return heads


def original_raw(pred, baseline, norm, n):
    if norm == 'bl':
        return pred * baseline
    if norm == 'sqrt_n':
        return pred * math.sqrt(n)
    return pred


def _rankcorr(a, b):
    # Stable tie-aware ranks without scipy/pandas dependencies.
    def rank(x):
        order = np.argsort(x, kind='stable')
        ranks = np.empty(len(x), dtype=float)
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and x[order[j]] == x[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + j - 1) / 2
            i = j
        return ranks
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def horizon_bucket(remaining):
    return '01-05' if remaining <= 5 else '06-10' if remaining <= 10 else '11-20' if remaining <= 20 else '21+'


@torch.no_grad()
def probe_heads(model, cfg, output, device):
    if (output / 'sibling_summary.csv').exists():
        print('Sibling probe already complete; reusing saved outputs.', flush=True)
        return
    data = cached_dataset(model, cfg, 'probe', output)
    heads = load_heads(cfg, output, device)
    predictions = {name: predict_batched(head, data['x'], device) for name, head in heads.items()}
    old = predict_batched(model.value_head, data['x'][:, :model.embedding_dim], device)
    predictions['checkpoint_head'] = original_raw(old, data['baseline'], cfg.original_value_norm, cfg.graph_size)
    predictions['rollout'] = data['y']
    scores = {name: (data['path'] + pred).numpy() for name, pred in predictions.items()}
    scores['prior'] = -data['prior'].numpy()
    np.savez_compressed(output / 'sibling_child_scores.npz',
        **{k: v.numpy() for k, v in data.items() if k != 'x'},
        **{f'score_{k}': v for k, v in scores.items()})
    keys = np.stack([data[k].numpy() for k in ('instance', 'source', 'step')], axis=1)
    groups = {}
    for row, key in enumerate(map(tuple, keys.tolist())):
        groups.setdefault(key, []).append(row)
    rows, calibration = [], []
    oracle = data['oracle'].numpy()
    for (instance, source, step), ids in groups.items():
        truth = oracle[ids]
        remaining = cfg.graph_size - step
        bucket = horizon_bucket(remaining)
        prior_pick = int(np.argmin(scores['prior'][ids]))
        source_name = 'greedy' if source == 0 else 'sample_tau3' if cfg.prefix_temperature == 3 else 'sample'
        for name, score in scores.items():
            values = score[ids]
            pick = int(np.argmin(values))
            exact = bool(np.isfinite(truth).all())
            regret = max(0., float(truth[pick] - truth.min())) if exact else float('nan')
            prior_regret = max(0., float(truth[prior_pick] - truth.min())) if exact else float('nan')
            rows.append(dict(instance=instance, source=source_name, step=step, remaining=remaining,
                bucket=bucket, method=name, oracle='exact_dp' if exact else 'unavailable',
                chosen_action=int(data['action'][ids[pick]]),
                regret=regret, optimal=float(regret <= 1e-6) if exact else float('nan'),
                prior_wrong=float(prior_regret > 1e-6) if exact else float('nan'),
                fixes_prior=float(prior_regret > 1e-6 and regret <= 1e-6) if exact else float('nan'),
                harmful_override=float(truth[pick] > truth[prior_pick] + 1e-6) if exact else float('nan'),
                spearman=_rankcorr(values, truth) if exact else float('nan'),
                agrees_prior=float(pick == prior_pick)))
        for name, pred in predictions.items():
            error = (pred[ids] - data['y'][ids]).numpy()
            calibration.append(dict(instance=instance, source=source_name, bucket=bucket,
                method=name, mse=float(np.mean(error ** 2)), bias=float(error.mean()),
                sibling_centered_rmse=float(np.sqrt(np.mean((error-error.mean()) ** 2)))))
    write_csv(output / 'sibling_decisions.csv', rows)
    write_csv(output / 'rollout_target_errors.csv', calibration)
    summary = []
    for name in scores:
        for source in sorted({r['source'] for r in rows}):
            for bucket in ['all', '01-05', '06-10', '11-20', '21+']:
                selected = [r for r in rows if r['method'] == name and r['source'] == source
                            and (bucket == 'all' or r['bucket'] == bucket)]
                if not selected:
                    continue
                exact = [r for r in selected if r['oracle'] == 'exact_dp']
                item = dict(method=name, source=source, bucket=bucket, states=len(selected),
                            exact_states=len(exact))
                for key in ('regret', 'optimal', 'harmful_override', 'spearman', 'agrees_prior'):
                    values = [r[key] for r in exact if np.isfinite(r[key])]
                    item[key] = float(np.mean(values)) if values else float('nan')
                wrong = sum(r['prior_wrong'] for r in exact)
                item['prior_wrong_states'] = int(wrong)
                item['fix_rate_when_prior_wrong'] = sum(r['fixes_prior'] for r in exact)/wrong if wrong else float('nan')
                # Cluster by graph, not by sibling/decision, for instance uncertainty.
                per_instance = {}
                for r in exact:
                    per_instance.setdefault(r['instance'], []).append(r['regret'])
                means = np.array([np.mean(v) for v in per_instance.values()])
                item['regret_instance_se'] = float(means.std(ddof=1)/math.sqrt(len(means))) if len(means)>1 else float('nan')
                summary.append(item)
    write_csv(output / 'sibling_summary.csv', summary)


def mst_remaining(state):
    """Admissible geometric bound for a started tour: MST(U)+connections to ends."""
    loc = state.loc[state.ids.view(-1)][0].detach().cpu().numpy().astype(np.float64)
    legal = np.flatnonzero(~state.get_mask().view(-1).cpu().numpy())
    points = loc[legal]
    if not len(legal):
        return float(np.linalg.norm(loc[int(state.prev_a.item())] - loc[int(state.first_a.item())]))
    d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    visited = np.zeros(len(legal), dtype=bool)
    nearest = np.full(len(legal), np.inf)
    nearest[0] = 0.0
    total = 0.0
    for _ in legal:
        j = int(np.argmin(np.where(visited, np.inf, nearest)))
        total += nearest[j]
        visited[j] = True
        nearest = np.minimum(nearest, d[j])
    if int(state.i.item()) > 0:
        total += np.linalg.norm(points - loc[int(state.prev_a.item())], axis=1).min()
        total += np.linalg.norm(points - loc[int(state.first_a.item())], axis=1).min()
    return float(total)


class EvaluatorSolver(MCTSSolver):
    """Only replaces leaf evaluation; selection, priors, backup, reuse unchanged."""
    def __init__(self, model, cfg, method, head=None, original_norm='none', device=None):
        super().__init__(model, cfg, device=device)
        self.method, self.head, self.original_norm = method, head, original_norm

    def _leaf(self, node, fixed, bl_val):
        logp, mask, glimpse = self.model.decode_step(fixed, node.state, return_glimpse=True)
        self.fwd_count_decode += 1
        self._fill_priors_from_logp(node, logp, mask)
        if self.method == 'rollout':
            return self._rollout_remaining_real(node.state, fixed) / bl_val
        if self.method == 'prior_only':
            # Constant total estimate: no learned/geometric discrimination at
            # nonterminal leaves. Terminal exact costs still enter backups.
            return 1.0 - float(node.state.lengths.item()) / bl_val
        if self.method == 'mst':
            return mst_remaining(node.state) / bl_val
        self.fwd_count_value += 1
        if self.method == 'checkpoint_head':
            value = float(self.model.value_head(glimpse).item())
            raw = original_raw(value, bl_val, self.original_norm, node.state.loc.shape[1])
        else:
            features = (state_features(fixed, node.state, glimpse)
                        if self.head.variant in ('repaired', 'repaired_geo') else glimpse)
            raw = float(self.head(features).item())
        return raw / bl_val  # every custom head is trained in raw tour-cost units

    def _populate_priors(self, node, fixed, bl_val):
        node.v_estimate = self._leaf(node, fixed, bl_val)

    def _expand(self, node, fixed, bl_val):
        return self._leaf(node, fixed, bl_val)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def search_batch(model, cfg, method, head, coords, baseline, k, device):
    solver = EvaluatorSolver(model, MCTSConfig(n_simulations=k, c_puct=cfg.c_puct,
        leaf_eval='value_head', value_norm='bl', value_target_norm='none',
        dirichlet_epsilon=0., temperature=0., tree_reuse=True, seed=cfg.data_seed),
        method, head, cfg.original_value_norm, device)
    records = []
    for i, loc in enumerate(coords):
        loc = loc.unsqueeze(0).to(device)
        sync(device)
        start = time.perf_counter()
        cost, tour = solver.solve_instance(loc, bl_val=float(baseline[i]))
        sync(device)
        seconds = time.perf_counter() - start
        if sorted(tour.cpu().tolist()) != list(range(cfg.graph_size)):
            raise AssertionError('Search returned an infeasible tour')
        records.append(dict(instance=i, cost=float(cost), seconds=seconds,
            decode_calls=solver.fwd_count_decode, value_calls=solver.fwd_count_value,
            rollout_calls=solver.fwd_count_rollout, tour=tour.cpu().tolist()))
    return records


@torch.no_grad()
def timed_greedy(model, coords, device):
    costs, seconds = [], []
    for loc in coords:
        loc = loc.unsqueeze(0).to(device)
        sync(device)
        start = time.perf_counter()
        c, _ = model(loc)
        sync(device)
        seconds.append(time.perf_counter() - start)
        costs.append(float(c))
    return np.asarray(costs), np.asarray(seconds)


@torch.no_grad()
def evaluate_search(model, cfg, output):
    device = torch.device(cfg.search_device if cfg.search_device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device).eval()
    heads = load_heads(cfg, output, device)
    methods = dict(rollout=None, prior_only=None, mst=None, checkpoint_head=None, **heads)
    calibration_coords = split_coordinates(cfg, 'calibration')
    test_coords = split_coordinates(cfg, 'search')
    save_torch(output / 'search_instances.pt', dict(calibration=calibration_coords, test=test_coords))
    # Warm up before any timing; all timing includes encode+search and adds the
    # measured single-instance greedy normalizer pass, shared across methods.
    model(calibration_coords[:1].to(device))
    for name, head in methods.items():
        search_batch(model, cfg, name, head, calibration_coords[:1], [1.0], 1, device)
    greedy_path = output / 'search_greedy_measurements.pt'
    if greedy_path.exists():
        measured = load_pt(greedy_path)
    else:
        cal_base, cal_greedy_time = timed_greedy(model, calibration_coords, device)
        test_base, test_greedy_time = timed_greedy(model, test_coords, device)
        measured = dict(cal_base=cal_base, cal_seconds=cal_greedy_time,
                        test_base=test_base, test_seconds=test_greedy_time)
        save_torch(greedy_path, measured)
    cal_base, cal_greedy_time = measured['cal_base'], measured['cal_seconds']
    test_base, test_greedy_time = measured['test_base'], measured['test_seconds']
    calibration_path = output / 'timing_calibration.json'
    if calibration_path.exists():
        timing = json.loads(calibration_path.read_text())
    else:
        timing = dict(device=str(device), cpu_threads=cfg.cpu_threads, methods={})
        def measure(name, k):
            totals = []
            for _ in range(cfg.timing_repeats):
                records = search_batch(model, cfg, name, methods[name], calibration_coords, cal_base, k, device)
                totals.append(np.mean([r['seconds'] for r in records]) + cal_greedy_time.mean())
            return float(np.median(totals))
        target = measure('rollout', cfg.search_K)
        timing['target_seconds_per_instance'] = target
        for name in methods:
            base_time = target if name == 'rollout' else measure(name, cfg.search_K)
            candidates = {cfg.search_K: base_time}
            if name != 'rollout':
                # Calibration instances only. Actual held-out times are always
                # reported: this is approximate time matching, not a deadline.
                overhead = float(cal_greedy_time.mean())
                guess = int(np.clip(round(cfg.search_K * max(target-overhead, 1e-6) /
                                    max(base_time-overhead, 1e-6)), 1,
                                    cfg.search_K * cfg.max_time_K_multiplier))
                if guess not in candidates:
                    candidates[guess] = measure(name, guess)
            chosen = min(candidates, key=lambda k: abs(math.log(candidates[k] / target)))
            timing['methods'][name] = dict(K=chosen, measured_seconds=candidates[chosen],
                                          candidates=candidates)
            print(f'calibration {name}: K={chosen}, {candidates[chosen]:.4f}s '
                  f'(target {target:.4f}s)', flush=True)
        save_json(calibration_path, timing)
    rows = []
    for i, (cost, seconds) in enumerate(zip(test_base, test_greedy_time)):
        rows.append(dict(mode='greedy', method='greedy', instance=i, K=0, cost=float(cost),
                         delta_greedy=0., search_seconds=0., total_seconds=float(seconds),
                         decode_calls=cfg.graph_size, value_calls=0, rollout_calls=0))
    for name, head in methods.items():
        budgets = dict(equal_K=cfg.search_K, calibrated_time=timing['methods'][name]['K'])
        for mode, k in budgets.items():
            path = output / 'search_raw' / f'{name}_K{k}.json'
            if path.exists():
                records = json.loads(path.read_text())
            else:
                records = search_batch(model, cfg, name, head, test_coords, test_base, k, device)
                save_json(path, records)
            for rec in records:
                i = rec['instance']
                rows.append(dict(mode=mode, method=name, instance=i, K=k, cost=rec['cost'],
                    delta_greedy=rec['cost']-float(test_base[i]), search_seconds=rec['seconds'],
                    total_seconds=rec['seconds']+float(test_greedy_time[i]),
                    decode_calls=rec['decode_calls'], value_calls=rec['value_calls'], rollout_calls=rec['rollout_calls']))
            print(f'{mode} {name}: K={k}, mean cost={np.mean([r["cost"] for r in records]):.6f}', flush=True)
    write_csv(output / 'search_per_instance.csv', rows)
    summary = []
    for mode, name in sorted({(r['mode'], r['method']) for r in rows}):
        chosen = [r for r in rows if r['mode'] == mode and r['method'] == name]
        delta = np.array([r['delta_greedy'] for r in chosen])
        total = float(np.mean([r['total_seconds'] for r in chosen]))
        summary.append(dict(mode=mode, method=name, K=chosen[0]['K'], instances=len(chosen),
            mean_cost=float(np.mean([r['cost'] for r in chosen])), delta_greedy=float(delta.mean()),
            paired_instance_se=float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta)>1 else float('nan'),
            seconds_per_instance=total, time_ratio_to_target=total/timing['target_seconds_per_instance']))
    write_csv(output / 'search_summary.csv', summary)


def paired_report(cfg, output):
    """Paired instance contrasts; never treat siblings as independent samples."""
    def read(name):
        with (output / name).open() as f:
            return list(csv.DictReader(f))
    contrasts = []
    def compare(kind, scope, seed, reference, a, b, variant='repaired'):
        ids = sorted(set(a) & set(b))
        if not ids:
            return
        delta = np.array([np.mean(a[i])-np.mean(b[i]) for i in ids])
        se = float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta)>1 else float('nan')
        contrasts.append(dict(evaluation=kind, scope=scope, seed=seed, variant=variant,
            comparison=f'{variant}_s{seed} minus {reference}', instances=len(ids),
            mean_difference=float(delta.mean()), paired_instance_se=se,
            approximate_ci95_low=float(delta.mean())-1.96*se,
            approximate_ci95_high=float(delta.mean())+1.96*se))
    if (output / 'sibling_decisions.csv').exists():
        rows = read('sibling_decisions.csv')
        for source in sorted({r['source'] for r in rows}):
            for bucket in ('all', '01-05', '06-10', '11-20', '21+'):
                maps = {}
                for r in rows:
                    if r['oracle'] == 'exact_dp' and r['source'] == source and (bucket == 'all' or r['bucket'] == bucket):
                        maps.setdefault(r['method'], {}).setdefault(r['instance'], []).append(float(r['regret']))
                for seed in cfg.head_seeds:
                    for variant in ('repaired', 'repaired_geo'):
                        refs = [f'original_s{seed}', f'original_wide_s{seed}', 'checkpoint_head', 'prior', 'rollout']
                        if variant == 'repaired_geo':
                            refs.append(f'repaired_s{seed}')
                        for ref in refs:
                            compare('sibling_regret', f'{source}/{bucket}', seed, ref,
                                    maps.get(f'{variant}_s{seed}', {}), maps.get(ref, {}), variant)
    if (output / 'search_per_instance.csv').exists():
        rows = read('search_per_instance.csv')
        for mode in ('equal_K', 'calibrated_time'):
            maps = {}
            for r in rows:
                if r['mode'] == mode:
                    maps.setdefault(r['method'], {}).setdefault(r['instance'], []).append(float(r['cost']))
            for seed in cfg.head_seeds:
                for variant in ('repaired', 'repaired_geo'):
                    refs = [f'original_s{seed}', f'original_wide_s{seed}', 'checkpoint_head', 'prior_only', 'mst', 'rollout']
                    if variant == 'repaired_geo':
                        refs.append(f'repaired_s{seed}')
                    for ref in refs:
                        compare('search_cost', mode, seed, ref, maps.get(f'{variant}_s{seed}', {}), maps.get(ref, {}), variant)
    write_csv(output / 'paired_contrasts.csv', contrasts)
    notes = '''# Value evaluator diagnostic: interpretation

Primary contrasts: repaired vs original_wide (information vs capacity) and
repaired_geo vs repaired (does explicit tour geometry help beyond endpoints?)
on held-out sibling regret and search cost. original_wide approximately matches
the repaired head's parameter count; repaired_geo has 14 extra inputs (about
1.8K more parameters at d=128) and predicts a residual over the MST bound, so
the `mst` control is its zero-MLP baseline. original is the unchanged small
architecture refitted on the same data. checkpoint_head is an historical
reference with a different training history.

All refits use identical raw greedy-policy completion targets, state-balanced
MSE, fixed train-instance feature statistics, optimizer settings, and seeded
minibatch orders. Best checkpoints are selected by validation rollout-target
MSE; test sibling/search results never select weights. Exact DP is an evaluation
oracle, not a training target. Features and targets are cached; the policy and
its normalization buffers remain frozen.

Read sibling_summary.csv by state source and remaining horizon. exact_states
is the denominator for exact regret/accuracy: horizons above the configured DP
cap have no certified ranking result. rollout_target_errors.csv measures a
different quantity: prediction of the frozen completion policy.

paired_contrasts.csv averages paired decision differences within each graph
before computing an approximate 95% interval across graphs. Negative differences
favor the repaired head. Intervals are conditional on a head seed and frozen
checkpoint; they do not describe variation across training seeds. Inspect all
seeds and their sample standard deviation in the notebook.

Search uses the Python reference implementation. prior_only assigns a constant
total estimate at nonterminal leaves while still backing up exact terminal
costs. mst uses an admissible geometric remaining-cost bound. Both retain the
same policy prior and search machinery. equal_K is a simulation-budget control.
calibrated_time picks K on separate instances to approximate rollout search's
time at the configured K. This is not a strict deadline; inspect actual test
seconds and time_ratio_to_target before claiming time-matched superiority.
Times include a measured greedy normalizer pass plus encoding/search, exclude
model loading and data transfer to the search device, and are hardware/backend
specific. Refitted feature construction is included. Resumed runs can span
sessions; rerun search in a fresh output directory for publication timings.

Before proceeding: check validation curves for unfinished fitting, all head
seeds, the parameter-count control, off-policy and early-horizon decisions,
and the actual quality/time tradeoff. Improvement only at short tails does not
establish an early-search solution. Successful evaluator repair would justify
a separate short policy-distillation experiment; it does not establish that
self-play will beat the REINFORCE baseline.
'''
    (output / 'INTERPRETATION.md').write_text(notes)


@torch.no_grad()
def check_invariants(model, cfg, output):
    device = next(model.parameters()).device
    loc = torch.rand(1, 6, 2, generator=torch.Generator().manual_seed(cfg.data_seed)).to(device)
    pairs = list(itertools.permutations(range(5), 2))
    prefixes = torch.tensor([[a] + [v for v in range(5) if v not in (a, b)] + [b]
                             for a, b in pairs], device=device)
    state = StateTSP.initialize(loc.expand(len(pairs), -1, -1))
    for t in range(5):
        state = state.update(prefixes[:, t])
    fixed = model.precompute_decoder(model.encode(loc))[torch.zeros(len(pairs), dtype=torch.long, device=device)]
    _, _, glimpse = model.decode_step(fixed, state, return_glimpse=True)
    enriched = state_features(fixed, state, glimpse)
    truth = (loc[0, prefixes[:, -1]] - loc[0, 5]).norm(dim=-1) + (loc[0, 5] - loc[0, prefixes[:, 0]]).norm(dim=-1)
    terminal = state.update(torch.full((len(pairs),), 5, device=device, dtype=torch.long))
    assert torch.allclose(truth, terminal.get_final_cost().view(-1)-state.lengths.view(-1), atol=2e-6)
    assert float((glimpse-glimpse[:1]).abs().max()) < 1e-5
    assert float((enriched-enriched[:1]).abs().max()) > 1e-4
    assert float(truth.max()-truth.min()) > 1e-4
    # Geometry column 0 (MST bound) is exact for a one-city tail and must equal
    # the `mst` control evaluator on a random started state.
    geo = enriched[:, base_feature_dim(model.embedding_dim):]
    assert geo.shape[1] == GEO_FEATURES
    assert torch.allclose(geo[:, 0].double().cpu(), truth.double().cpu(), atol=1e-5)
    loc8 = torch.rand(1, 8, 2, generator=torch.Generator().manual_seed(cfg.data_seed + 7)).to(device)
    state8 = StateTSP.initialize(loc8).update(torch.tensor([3], device=device)).update(torch.tensor([6], device=device))
    fixed8 = model.precompute_decoder(model.encode(loc8))
    _, _, glimpse8 = model.decode_step(fixed8, state8, return_glimpse=True)
    feat8 = state_features(fixed8, state8, glimpse8)
    mst8 = float(feat8[0, base_feature_dim(model.embedding_dim)])
    assert abs(mst8 - mst_remaining(state8)) < 1e-5, (mst8, mst_remaining(state8))
    # DP independently checked against exhaustive path enumeration.
    loc_np = loc[0].cpu().numpy().astype(np.float64)
    legal = [2, 3, 4, 5]
    dp = exact_child_remaining(loc_np, legal, 0)
    brute = []
    for action in legal:
        costs = []
        for perm in itertools.permutations([x for x in legal if x != action]):
            path = loc_np[[action, *perm, 0]]
            costs.append(np.linalg.norm(path[1:]-path[:-1], axis=-1).sum())
        brute.append(min(costs))
    assert np.allclose(dp, brute, atol=1e-10)
    # Check normalized wrapper against the untouched reference MCTS, including
    # both original-head unit conversion and rollout target accounting.
    baseline = float(model(loc)[0].item())
    parity = {}
    for mode in ('value_head', 'rollout'):
        conf = MCTSConfig(n_simulations=5, c_puct=cfg.c_puct, leaf_eval=mode,
                          value_target_norm=cfg.original_value_norm, seed=cfg.data_seed)
        reference = MCTSSolver(model, conf, device)
        adapter = EvaluatorSolver(model, conf, 'checkpoint_head' if mode == 'value_head' else 'rollout',
                                  original_norm=cfg.original_value_norm, device=device)
        a, ta = reference.solve_instance(loc, bl_val=baseline)
        b, tb = adapter.solve_instance(loc, bl_val=baseline)
        assert torch.equal(ta, tb) and torch.allclose(a, b, atol=1e-6)
        parity[mode] = True
    report = dict(alias_glimpse_max_difference=float((glimpse-glimpse[:1]).abs().max()),
        repaired_features_max_difference=float((enriched-enriched[:1]).abs().max()),
        geometry_mst_bound_matches_control=True, geometry_one_city_tail_exact=True,
        true_cost_range=[float(truth.min()), float(truth.max())],
        exact_oracle_matches_brute_force=True, closing_edge_accounting=True,
        reference_search_parity=parity)
    save_json(output / 'invariants.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_phase(cfg, phase):
    """Notebook/CLI entry point. Stages resume from manifest-checked artifacts."""
    if phase not in ('check', 'data', 'train', 'probe', 'search', 'report', 'all'):
        raise ValueError(f'Unknown phase {phase}')
    model, output, device = open_run(cfg)
    before = policy_digest(model)
    try:
        if phase in ('check', 'all'):
            check_invariants(model, cfg, output)
        if phase == 'data':
            for split in ('train', 'val'):
                cached_dataset(model, cfg, split, output)
        if phase in ('train', 'all'):
            train_heads(model, cfg, output, device)
        if phase in ('probe', 'all'):
            probe_heads(model, cfg, output, device)
        if phase in ('search', 'all'):
            evaluate_search(model, cfg, output)
        if phase in ('probe', 'search', 'report', 'all'):
            paired_report(cfg, output)
    finally:
        after = policy_digest(model)
        if before != after or any(p.requires_grad for p in model.parameters()):
            raise AssertionError('Frozen policy parameters/buffers changed!')
        with (output / 'freeze_checks.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=phase, unchanged=True, sha256=after, time=time.time()))+'\n')
    print(f'{phase} complete → {output}', flush=True)
    return output
