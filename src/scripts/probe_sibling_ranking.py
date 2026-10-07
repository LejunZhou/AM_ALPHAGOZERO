"""Sibling-ranking diagnostic: value head vs greedy rollout vs exact optimum.

The question this answers: at a partial-tour state s, MCTS must RANK the legal
children s' = s + a. How often does each leaf evaluator pick the child that the
exact optimal completion says is best, and how much tour length does a wrong
pick cost? R^2 on cost-to-go does not answer this; this probe does.

For every probed state s we enumerate all legal children and score each child in
TOTAL tour-cost units (path so far + completion from the child):

    vh  : lengths(s') + value_head(s')            (de-normalized per --value_target_norm)
    ro  : lengths(s') + greedy rollout from s'    (what leaf_eval='rollout' uses in MCTS)
    opt : lengths(s') + exact optimal completion  (path-TSP solved by LKH via elkai,
                                                   dummy-node trick; brute-force for k<=3)
    pol : the parent's policy prior over children (argmax = greedy action)

Per state we record which child each scorer picks, the optimal-completion regret
of that pick, pairwise pick agreement, Spearman rank correlations, and the
error-vs-signal spread across siblings. Aggregates are printed by step bucket.

States come from the model's own greedy trajectory (--state_source greedy, the
on-policy states MCTS roots sit on) or from a trajectory whose first
ceil(sample_frac*N) actions are sampled at tau=1 (--state_source sample, mimicking
step30 self-play exploration).

Usage:
    PYTHONPATH=src python src/scripts/probe_sibling_ranking.py \
        --ckpt outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
        --value_target_norm bl --n_instances 100 --out _progress/eval_logs/sib_tsp20_stage1.csv
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.state import StateTSP
from am_baseline.training.coach import reconstruct_state


# ---------------------------------------------------------------------------
# Model loading (Stage 1 epoch-*.pt with sibling args.json, or Stage 4 iter-*.pt)
# ---------------------------------------------------------------------------

class _Cfg:
    embedding_dim = 128
    hidden_dim = 128
    n_encode_layers = 3
    n_heads = 8
    tanh_clipping = 10.0
    normalization = 'batch'
    feed_forward_hidden = 512
    value_enabled = True
    value_hidden_dim = 128


def load_any(ckpt_path, key):
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'args.json')
    train_args = json.load(open(args_path)) if os.path.exists(args_path) else {}
    cfg = _Cfg()
    for k in ('embedding_dim', 'hidden_dim', 'n_encode_layers', 'n_heads', 'tanh_clipping',
              'normalization', 'feed_forward_hidden', 'value_hidden_dim'):
        if k in train_args:
            setattr(cfg, k, train_args[k])
    model = AttentionModel(cfg)
    if key == 'auto':
        key = 'best_model' if 'best_model' in data else 'model'
    state = data[key] if isinstance(data, dict) and key in data else data
    model.load_state_dict(state)
    model.eval()
    return model, train_args, key


# ---------------------------------------------------------------------------
# Exact completion costs for ALL children of a parent state
#   m <= DP_MAX : Held-Karp bitmask DP (numba), exact, one DP per parent state
#   m >  DP_MAX : LKH (elkai) with a dummy node per child; heuristic, verified
# ---------------------------------------------------------------------------

DP_MAX = 20
# LKH multiplies int costs by PRECISION=100 internally (int32): keep scaled
# costs below ~2e7. 1e5 on unit-square instances leaves ample headroom.
_SCALE = 100_000

from numba import njit


@njit(cache=True)
def _dp_all_children(D, m):
    """D: (m+1, m+1) distances among [u_0..u_{m-1}, first]. Returns g[j] = min
    cost of a Hamiltonian path that starts at u_j, visits every other u, and
    ends at `first`."""
    INF = 1e18
    full = (1 << m) - 1
    f = np.full((1 << m, m), INF)
    for j in range(m):
        f[1 << j, j] = D[j, m]
    for S in range(1, 1 << m):
        for j in range(m):
            if ((S >> j) & 1) == 0 or S == (1 << j):
                continue
            R = S & ~(1 << j)
            best = INF
            for k in range(m):
                if ((R >> k) & 1) == 1:
                    v = D[j, k] + f[R, k]
                    if v < best:
                        best = v
            f[S, j] = best
    out = np.empty(m)
    for j in range(m):
        out[j] = f[full, j]
    return out


def dp_children_costs(loc, u_list, first):
    nodes = list(u_list) + [first]
    P = loc[nodes]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return _dp_all_children(np.ascontiguousarray(D), len(u_list))


def _elkai_tour(mat_int, runs=10):
    import elkai
    if hasattr(elkai, 'solve_int_matrix'):
        return list(elkai.solve_int_matrix(mat_int, runs))
    return list(elkai.DistanceMatrix(mat_int).solve_tsp(runs=runs))


def path_tsp_cost_lkh(loc, start, end, inner, runs=10):
    """Heuristic min-cost Hamiltonian path start -> (all of inner) -> end via LKH
    with a dummy node adjacent (cost 0) to both endpoints. Raises if the dummy
    constraint is violated (caller retries with more runs)."""
    nodes = [start, *inner, end]
    n = len(nodes)
    P = loc[nodes]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    big = float(D.max()) * n + 1.0          # exceeds any Hamiltonian path cost
    M = np.full((n + 1, n + 1), big, dtype=np.float64)
    M[:n, :n] = D
    M[n, n] = 0.0
    M[n, 0] = M[0, n] = 0.0
    M[n, n - 1] = M[n - 1, n] = 0.0
    Mi = np.rint(M * _SCALE).astype(np.int64).tolist()
    tour = _elkai_tour(Mi, runs)
    if len(tour) == n + 2:
        tour = tour[:-1]
    assert sorted(tour) == list(range(n + 1)), f'bad LKH tour {tour}'
    pos = tour.index(n)
    nb = {tour[(pos - 1) % (n + 1)], tour[(pos + 1) % (n + 1)]}
    if nb != {0, n - 1}:
        raise RuntimeError(f'dummy node not between start/end: neighbors {nb}')
    return float(sum(M[tour[i], tour[(i + 1) % (n + 1)]] for i in range(n + 1)))


def children_completion_costs(loc, legal, first, lkh_runs=10):
    """Exact (DP) or near-exact (LKH) completion cost from each child a in legal:
    path a -> (legal minus a) -> first. Returns (costs (m,), n_lkh_used)."""
    m = len(legal)
    if m <= DP_MAX:
        return dp_children_costs(loc, list(legal), first), 0
    out = np.empty(m)
    others = set(int(x) for x in legal)
    for j, a in enumerate(legal):
        inner = sorted(others - {int(a)})
        for runs in (lkh_runs, 3 * lkh_runs):
            try:
                out[j] = path_tsp_cost_lkh(loc, int(a), first, inner, runs=runs)
                break
            except RuntimeError:
                continue
        else:
            out[j] = np.nan
    return out, m


def selftest_solvers(seed=0):
    rng = np.random.default_rng(seed)
    worst_dp = 0.0
    for _ in range(30):                       # DP vs brute force, 3..6 inner nodes
        m = int(rng.integers(3, 7))
        loc = rng.random((m + 1, 2))
        first = m
        g = dp_children_costs(loc, list(range(m)), first)
        for j in range(m):
            inner = [x for x in range(m) if x != j]
            best = math.inf
            for perm in itertools.permutations(inner):
                seq = [j, *perm, first]
                best = min(best, sum(float(np.linalg.norm(loc[seq[i]] - loc[seq[i + 1]]))
                                     for i in range(len(seq) - 1)))
            worst_dp = max(worst_dp, abs(g[j] - best))
    worst_lkh = 0.0
    for _ in range(10):                       # LKH vs DP, 18 inner nodes
        loc = rng.random((20, 2))
        first = 19
        g = dp_children_costs(loc, list(range(19)), first)
        for j in (0, 7):
            inner = [x for x in range(19) if x != j]
            c = path_tsp_cost_lkh(loc, j, first, inner)
            worst_lkh = max(worst_lkh, c - g[j])
    return worst_dp, worst_lkh


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

@torch.no_grad()
def trajectory(model, fixed, coords, n_sample_steps, gen, tau=1.0):
    """Roll one tour. First `n_sample_steps` actions sampled at temperature `tau`,
    rest greedy. Returns per-step snapshots + final tour cost."""
    N = coords.size(1)
    st = StateTSP.initialize(coords)
    snaps = []
    for t in range(N):
        log_p, mask = model.decode_step(fixed, st)
        probs = log_p.exp().view(-1)
        if t < n_sample_steps:
            lp = log_p.view(-1) / tau
            lp = torch.where(mask.view(-1), torch.full_like(lp, -float('inf')), lp)
            p_tau = torch.softmax(lp, dim=-1)
            a = int(torch.multinomial(p_tau, 1, generator=gen).item())
        else:
            a = int(probs.argmax().item())
        snaps.append(dict(
            t=t,
            first_a=int(st.first_a.view(-1)[0].item()),
            prev_a=int(st.prev_a.view(-1)[0].item()),
            visited=st.visited_.view(-1).clone(),
            lengths=float(st.lengths.view(-1)[0].item()),
            probs=probs.clone(),
            action=a,
        ))
        st = st.update(torch.tensor([a], dtype=torch.long))
    return snaps, float(st.get_final_cost().view(-1)[0].item())


@torch.no_grad()
def score_children(model, fixed, coords, snap, bl_val, value_norm, loc_np, do_opt=True, lkh_runs=10):
    """Score every legal child of the state in `snap`. Returns dict of (m,) arrays."""
    N = coords.size(1)
    t = snap['t']
    visited = snap['visited']
    legal = (~visited).nonzero().view(-1)
    m = int(legal.numel())
    prev = snap['prev_a']
    first = snap['first_a']
    dist_row = torch.from_numpy(np.linalg.norm(loc_np - loc_np[prev], axis=-1)).float()
    child_lengths = snap['lengths'] + dist_row[legal]                          # (m,)
    child_visited = visited.view(1, N).expand(m, N).clone()
    child_visited[torch.arange(m), legal] = True
    batch = dict(
        state_i=t + 1,
        coords=coords.expand(m, N, 2).contiguous(),
        visited=child_visited,
        first_a=torch.full((m,), first, dtype=torch.long),
        prev_a=legal.clone(),
        lengths=child_lengths,
    )
    cs = reconstruct_state(batch, device=torch.device('cpu'))
    fixed_m = fixed[torch.zeros(m, dtype=torch.long)]

    # value head on each child
    _, _, glimpse = model.decode_step(fixed_m, cs, return_glimpse=True)
    v = model.value_head(glimpse).view(-1).double()
    if value_norm == 'bl':
        v_raw = v * bl_val
    elif value_norm == 'none':
        v_raw = v
    elif value_norm == 'sqrt_n':
        v_raw = v * math.sqrt(N)
    else:
        raise ValueError(value_norm)
    vh_total = child_lengths.double() + v_raw

    # greedy rollout from each child (batched)
    cur = cs
    while not cur.all_finished():
        lp, _ = model.decode_step(fixed_m, cur)
        a = lp.view(m, -1).argmax(-1)
        cur = cur.update(a)
    ro_total = cur.get_final_cost().view(-1).double()

    # exact / near-exact optimal completion from each child
    opt_total = np.full(m, np.nan)
    n_lkh = 0
    if do_opt:
        comp, n_lkh = children_completion_costs(loc_np, legal.numpy(), first, lkh_runs=lkh_runs)
        opt_total = child_lengths.double().numpy() + comp

    return dict(
        m=m, legal=legal.numpy(), n_lkh=n_lkh,
        pol=snap['probs'][legal].double().numpy(),
        vh=vh_total.numpy(), ro=ro_total.numpy(), opt=opt_total,
    )


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).correlation)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ckpt_key', default='auto', help='model | best_model | auto')
    p.add_argument('--value_target_norm', default='auto', help='bl | none | sqrt_n | auto')
    p.add_argument('--graph_size', type=int, default=None)
    p.add_argument('--n_instances', type=int, default=100)
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--state_source', choices=['greedy', 'sample'], default='greedy')
    p.add_argument('--sample_frac', type=float, default=0.3)
    p.add_argument('--sample_tau', type=float, default=1.0,
                   help='softmax temperature for the sampled prefix (1 = policy, >1 = flatter)')
    p.add_argument('--step_stride', type=int, default=1)
    p.add_argument('--step_offset', type=int, default=1)
    p.add_argument('--no_opt', action='store_true', help='skip exact LKH completion')
    p.add_argument('--lkh_runs', type=int, default=10, help='LKH runs per child when m > DP_MAX')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--tag', default='')
    args = p.parse_args()

    if args.selftest:
        w_dp, w_lkh = selftest_solvers()
        print(f'[selftest] DP vs brute force: max |diff| = {w_dp:.2e}; '
              f'LKH(dummy) minus DP: max = {w_lkh:.2e}')
        assert w_dp < 1e-9
    torch.set_num_threads(max(1, torch.get_num_threads()))

    model, train_args, used_key = load_any(args.ckpt, args.ckpt_key)
    assert model.value_head is not None, 'checkpoint has no value head'
    vnorm = args.value_target_norm
    if vnorm == 'auto':
        vnorm = train_args.get('value_target_norm', 'none')
    N = args.graph_size or train_args.get('graph_size')
    assert N, 'pass --graph_size'
    print(f'[*] ckpt={args.ckpt} key={used_key} value_target_norm={vnorm} N={N} '
          f'source={args.state_source} n_inst={args.n_instances} seed={args.seed}')

    gen = torch.Generator().manual_seed(args.seed)
    coords_all = torch.rand(args.n_instances, N, 2, generator=gen)
    n_sample = int(math.ceil(args.sample_frac * N)) if args.state_source == 'sample' else 0

    rows = []
    t0 = time.time()
    n_lkh = 0
    n_lkh_used = 0
    n_lkh_above_ro = 0
    with torch.no_grad():
        for idx in range(args.n_instances):
            coords = coords_all[idx:idx + 1]
            loc_np = coords[0].double().numpy()
            enc = model.encode(coords)
            fixed = model.precompute_decoder(enc)
            g_snaps, g_cost = trajectory(model, fixed, coords, 0, gen)
            bl_val = g_cost
            if args.state_source == 'sample':
                snaps, traj_cost = trajectory(model, fixed, coords, n_sample, gen, tau=args.sample_tau)
            else:
                snaps, traj_cost = g_snaps, g_cost

            for t in range(args.step_offset, N - 1, args.step_stride):
                snap = snaps[t]
                sc = score_children(model, fixed, coords, snap, bl_val, vnorm, loc_np,
                                    do_opt=not args.no_opt, lkh_runs=args.lkh_runs)
                m = sc['m']
                n_lkh += m
                legal = sc['legal']
                pick = {k: int(np.argmin(sc[k])) for k in ('vh', 'ro', 'opt')}
                pick['pol'] = int(np.argmax(sc['pol']))
                # sanity: greedy-source states -> rollout through policy pick == trajectory cost
                if args.state_source == 'greedy':
                    assert abs(sc['ro'][pick['pol']] - traj_cost) < 1e-3, \
                        (sc['ro'][pick['pol']], traj_cost)
                opt = sc['opt']
                ro = sc['ro']
                vh = sc['vh']
                n_lkh_used += sc['n_lkh']
                if sc['n_lkh']:
                    # LKH is heuristic: a completion above the greedy rollout is a
                    # solver miss, not a property of the state. Clamp and count.
                    over = opt > ro + 1e-9
                    n_lkh_above_ro += int(np.nansum(over))
                    opt = np.where(np.isnan(opt) | over, ro, opt)
                min_opt = np.nanmin(opt)
                min_ro = ro.min()
                srt = np.sort(opt)
                margin = float(srt[1] - srt[0]) if m >= 2 else np.nan
                row = dict(
                    inst=idx, t=t, m=m, bl_val=bl_val, traj_cost=traj_cost, min_opt=min_opt,
                    min_ro=min_ro, margin_opt=margin,
                    pol_entropy=float(-(sc['pol'] * np.log(sc['pol'] + 1e-12)).sum()),
                )
                for k in ('vh', 'ro', 'pol'):
                    row[f'ropt_{k}'] = float(opt[pick[k]] - min_opt)      # regret under optimal play
                    row[f'isopt_{k}'] = int(opt[pick[k]] - min_opt <= 1e-6)
                for k in ('vh', 'pol'):
                    row[f'rro_{k}'] = float(ro[pick[k]] - min_ro)         # regret measured by rollout
                    row[f'isbestro_{k}'] = int(ro[pick[k]] - min_ro <= 1e-6)
                # MCTS operational regime: PUCT with c_puct=0.05 only explores the
                # top-prior children, so rank quality among the top-3 by prior is
                # what the search actually consumes.
                top = np.argsort(-sc['pol'])[:3]
                best_top = opt[top].min()
                srt_top = np.sort(opt[top])
                row['margin_top3'] = float(srt_top[1] - srt_top[0]) if len(top) >= 2 else np.nan
                for k, arr in (('vh', vh), ('ro', ro)):
                    j = top[int(np.argmin(arr[top]))]
                    row[f'ropt_top3_{k}'] = float(opt[j] - best_top)
                    row[f'isopt_top3_{k}'] = int(opt[j] - best_top <= 1e-6)
                row['ropt_top3_pol'] = float(opt[top[0]] - best_top)
                row['isopt_top3_pol'] = int(opt[top[0]] - best_top <= 1e-6)
                # calibration slope across siblings: d(estimate)/d(opt). 1 = fully
                # resolves sibling differences; 0 = flat (no discrimination).
                def _slope(y, x):
                    if len(x) < 3 or np.std(x) < 1e-12:
                        return np.nan
                    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
                row['slope_vh'] = _slope(vh, opt)
                row['slope_ro'] = _slope(ro, opt)
                # value head prefers an off-policy child that is actually worse
                row['vh_offpol_worse'] = int(pick['vh'] != pick['pol'] and
                                             opt[pick['vh']] > opt[pick['pol']] + 1e-6)
                row['agree_vh_ro'] = int(pick['vh'] == pick['ro'])
                row['agree_vh_pol'] = int(pick['vh'] == pick['pol'])
                row['agree_ro_pol'] = int(pick['ro'] == pick['pol'])
                row['sp_vh_opt'] = spearman(vh, opt)
                row['sp_ro_opt'] = spearman(ro, opt)
                row['sp_vh_ro'] = spearman(vh, ro)
                row['sig_std'] = float(np.std(opt))
                row['err_std_vh'] = float(np.std(vh - opt))
                row['err_std_ro'] = float(np.std(ro - opt))
                row['bias_vh'] = float(np.mean(vh - opt))
                row['bias_ro'] = float(np.mean(ro - opt))
                rows.append(row)
            if (idx + 1) % 10 == 0:
                print(f'  inst {idx + 1}/{args.n_instances}  states={len(rows)}  '
                      f'child-evals={n_lkh}  wall={time.time() - t0:.0f}s', flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f'[*] wrote {len(df)} state rows -> {args.out}')

    # ---- summary -----------------------------------------------------------
    def bucket(t):
        q = N // 4
        return 'early' if t < q else ('late' if t >= N - q else 'mid')
    df['bucket'] = df['t'].map(bucket)

    def summarize(d):
        out = dict(n=len(d), m=d['m'].mean())
        for k in ('pol', 'ro', 'vh'):
            out[f'P(opt)_{k}'] = d[f'isopt_{k}'].mean()
            out[f'regret_{k}'] = d[f'ropt_{k}'].mean()
        out['agree_vh_ro'] = d['agree_vh_ro'].mean()
        out['agree_vh_pol'] = d['agree_vh_pol'].mean()
        out['agree_ro_pol'] = d['agree_ro_pol'].mean()
        out['sp_vh_opt'] = d['sp_vh_opt'].mean()
        out['sp_ro_opt'] = d['sp_ro_opt'].mean()
        out['err/sig_vh'] = (d['err_std_vh'] / d['sig_std'].clip(lower=1e-9)).median()
        out['err/sig_ro'] = (d['err_std_ro'] / d['sig_std'].clip(lower=1e-9)).median()
        out['bias_vh'] = d['bias_vh'].mean()
        out['bias_ro'] = d['bias_ro'].mean()
        # policy-improvement view: states where the greedy action is NOT optimal
        bad = d[d['isopt_pol'] == 0]
        out['n_pol_wrong'] = len(bad)
        out['fix_by_ro'] = bad['isopt_ro'].mean() if len(bad) else np.nan
        out['fix_by_vh'] = bad['isopt_vh'].mean() if len(bad) else np.nan
        out['vh_worse_than_pol'] = (d['ropt_vh'] > d['ropt_pol'] + 1e-9).mean()
        out['vh_better_than_pol'] = (d['ropt_vh'] < d['ropt_pol'] - 1e-9).mean()
        for k in ('pol', 'ro', 'vh'):
            out[f'P(opt)top3_{k}'] = d[f'isopt_top3_{k}'].mean()
            out[f'regret_top3_{k}'] = d[f'ropt_top3_{k}'].mean()
        out['slope_ro_med'] = d['slope_ro'].median()
        out['slope_vh_med'] = d['slope_vh'].median()
        out['margin_opt_med'] = d['margin_opt'].median()
        out['margin_top3_med'] = d['margin_top3'].median()
        out['sig_std_med'] = d['sig_std'].median()
        out['err_std_vh_med'] = d['err_std_vh'].median()
        out['err_std_ro_med'] = d['err_std_ro'].median()
        out['ropt_vh_p50'] = d['ropt_vh'].median()
        out['ropt_vh_p90'] = d['ropt_vh'].quantile(0.9)
        out['P(ropt_vh>0.05)'] = (d['ropt_vh'] > 0.05).mean()
        out['vh_offpol_worse'] = d['vh_offpol_worse'].mean()
        return out

    print('\n' + '=' * 96)
    print(f'SIBLING-RANKING SUMMARY  {args.tag}  ckpt={os.path.basename(os.path.dirname(args.ckpt))}/'
          f'{os.path.basename(args.ckpt)} [{used_key}]  N={N}  source={args.state_source}  '
          f'value_target_norm={vnorm}')
    print('=' * 96)
    groups = [('ALL', df)] + [(b, df[df['bucket'] == b]) for b in ('early', 'mid', 'late')]
    keys = ['n', 'm', 'P(opt)_pol', 'P(opt)_ro', 'P(opt)_vh', 'regret_pol', 'regret_ro', 'regret_vh',
            'agree_vh_ro', 'agree_vh_pol', 'agree_ro_pol', 'sp_ro_opt', 'sp_vh_opt',
            'err/sig_ro', 'err/sig_vh', 'bias_ro', 'bias_vh',
            'n_pol_wrong', 'fix_by_ro', 'fix_by_vh', 'vh_better_than_pol', 'vh_worse_than_pol',
            'vh_offpol_worse',
            'P(opt)top3_pol', 'P(opt)top3_ro', 'P(opt)top3_vh',
            'regret_top3_pol', 'regret_top3_ro', 'regret_top3_vh',
            'slope_ro_med', 'slope_vh_med', 'margin_opt_med', 'margin_top3_med', 'sig_std_med',
            'err_std_ro_med', 'err_std_vh_med', 'ropt_vh_p50', 'ropt_vh_p90', 'P(ropt_vh>0.05)']
    header = f'{"metric":>22s}' + ''.join(f'{g:>12s}' for g, _ in groups)
    print(header)
    sums = [summarize(d) for _, d in groups]
    for k in keys:
        vals = []
        for s in sums:
            v = s[k]
            vals.append(f'{v:12d}' if isinstance(v, (int, np.integer)) else f'{v:12.4f}')
        print(f'{k:>22s}' + ''.join(vals))
    print(f'\nwall {time.time() - t0:.0f}s, child evaluations {n_lkh} '
          f'(exact DP: {n_lkh - n_lkh_used}, LKH: {n_lkh_used}, LKH above rollout (clamped): {n_lkh_above_ro})')


if __name__ == '__main__':
    main()
