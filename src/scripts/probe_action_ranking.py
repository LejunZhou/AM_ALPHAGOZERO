"""Stage 5 §H.7 — action-RANKING probe for the value head.

The decision-relevant question (not absolute bias): at a node, does the value
head rank the legal next actions the same way the ground truth does? A constant
bias is invisible to PUCT's argmax; what matters is the *relative* ordering of
the children.

Per sampled decision node s, over legal actions a -> child s'_a:
  v_a  = value head cost-to-go estimate for s'_a
  g_a  = E[cost-to-go | s'_a] via R STOCHASTIC (sampled-policy) rollouts (mean),
         with SE reported so we can confirm R is high enough to rank reliably.
  edge_a = dist(current, a)   (0 at step 0)
Action value = edge_a + {v or g}_a  (what MCTS effectively compares).

Metrics per node: Spearman/Kendall (edge+v vs edge+g = decision-relevant; and
v vs g = the head's intrinsic cost-to-go ranking), top-1 action match, and
DECISION REGRET = g-cost of the value-head's pick minus the true best (raw+bl).

`g` uses the policy (encoder+decoder), which is shared across the glimpse /
shared-trunk / own-encoder checkpoints, so it is cached and reused.

Repro:
    PYTHONPATH=src python -m scripts.probe_action_ranking \
        --ckpt outputs/.../iter-99.pt --buffer outputs/.../buffer.pt --which best \
        --num_nodes 50 --rollouts_per_child 16 --min_actions 4 --seed 1234
"""
import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from am_baseline.training.coach import MCTSReplayBuffer, reconstruct_state
from scripts.probe_value_aleatoric import (
    _load_stage4_model, _read_json_next_to, _sample_slots, _batch_from_slot,
)


# ---- rank-correlation helpers (no scipy dependency) -----------------------
def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    starts = csum - counts
    avg = (starts + csum - 1) / 2.0
    return avg[inv]


def _pearson(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def _kendall(x, y):
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return float((c - d) / (c + d)) if (c + d) > 0 else float("nan")


@torch.no_grad()
def _rollout_ctg(model, fixed, child_state, R, gen, device, mode="sampled"):
    """Rollout cost-to-go samples from a child.
    mode='sampled': R stochastic (multinomial) rollouts.
    mode='greedy' : 1 deterministic argmax rollout (= mix's leaf-eval rollout)."""
    reps = 1 if mode == "greedy" else R
    out = np.empty(reps, dtype=np.float64)
    start_len = float(child_state.lengths.view(-1)[0].item())
    for r in range(reps):
        state = child_state
        while not state.all_finished():
            log_p, _mask = model.decode_step(fixed, state)
            if mode == "greedy":
                a = log_p.view(1, -1).argmax(dim=-1)
            else:
                a = torch.multinomial(log_p.exp().view(1, -1), 1, generator=gen).view(1)
            state = state.update(a)
        total = float(state.get_final_cost().view(-1)[0].item())
        out[r] = total - start_len
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Value-head action-ranking probe.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--which", choices=["best", "working"], default="best")
    p.add_argument("--value_head_type", choices=["glimpse_mlp", "separate_trunk"], default="glimpse_mlp")
    p.add_argument("--value_own_encoder", action="store_true")
    p.add_argument("--value_target_norm", choices=["bl", "sqrt_n", "none"], default=None)
    p.add_argument("--num_nodes", type=int, default=50)
    p.add_argument("--rollouts_per_child", type=int, default=16)
    p.add_argument("--gt_mode", choices=["sampled", "greedy"], default="sampled",
                   help="'greedy' = deterministic argmax rollout (= mix's leaf-eval); 'sampled' = τ=1 rollouts.")
    p.add_argument("--min_actions", type=int, default=4, help="Skip nodes with fewer legal actions.")
    p.add_argument("--root_hop", type=int, default=0,
                   help="Push each probe node h RANDOM legal actions off the buffer state "
                        "before ranking (depth-(h+1) off-policy generalization probe; §V0 gate G2).")
    p.add_argument("--holdout_k", type=int, default=0)
    p.add_argument("--inst_split", choices=["all", "train", "eval"], default="all")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gt_cache", default=None, help="npz path to cache/reuse the ground-truth g.")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def _gt_cache_path(args, graph_size, explicit):
    if explicit:
        return explicit
    key = f"{os.path.abspath(args.buffer)}|{args.seed}|{args.num_nodes}|{args.rollouts_per_child}|" \
          f"{args.min_actions}|{args.holdout_k}|{args.inst_split}|{graph_size}|{args.gt_mode}"
    if args.root_hop > 0:
        # Appended only when hopping so pre-hop cache hashes stay valid.
        key += f"|hop{args.root_hop}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"rank_gt_{h}.npz")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_args = _read_json_next_to(args.ckpt)
    graph_size = int(args.__dict__.get("graph_size") or (train_args or {}).get("graph_size", 20))
    vtn = args.value_target_norm or (train_args or {}).get("value_target_norm", "none")
    capacity = int((train_args or {}).get("buffer_capacity", 5000))

    model = _load_stage4_model(
        args.ckpt, args.which, train_args, graph_size, vtn, device,
        value_head_type=args.value_head_type, value_own_encoder=args.value_own_encoder,
    )
    own_trunk = getattr(model, "value_trunk", None) is not None
    buf = MCTSReplayBuffer(graph_size=graph_size, capacity_instances=capacity, device="cpu")
    buf.load(args.buffer)

    # With --root_hop h, the ranked node sits at step t+h, so cap parent steps
    # at graph_size - min_actions - h to keep >= min_actions legal children.
    steps_filter = list(range(0, graph_size - args.min_actions + 1 - args.root_hop))
    slots = _sample_slots(buf, args.num_nodes, np_rng, steps_filter=steps_filter,
                          holdout_k=args.holdout_k, inst_split=args.inst_split)
    hop_rng = np.random.default_rng(args.seed + 4242)  # separate stream: slot sampling unchanged

    cache_path = _gt_cache_path(args, graph_size, args.gt_cache)
    gt = None
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if list(z["slots"]) == list(slots):
            gt = {k: z[k] for k in z.files}
            print(f"[gt] loaded cached ground truth from {cache_path}")

    print(f"device={device} own_trunk={own_trunk} value_head_type={args.value_head_type} "
          f"own_encoder={args.value_own_encoder} vtn={vtn}")
    print(f"nodes={args.num_nodes} R={args.rollouts_per_child} min_actions={args.min_actions} "
          f"split={args.inst_split} holdout_k={args.holdout_k} root_hop={args.root_hop}")

    gen = torch.Generator(device=device).manual_seed(args.seed + 999)
    rows = []
    t0 = time.time()

    # per-node cached ground truth accumulators (built if gt is None)
    gt_actions, gt_edge, gt_gmean, gt_gse, gt_bl, gt_step = {}, {}, {}, {}, {}, {}

    for node_id, slot in enumerate(slots):
        batch = _batch_from_slot(buf, slot)
        coords = batch["coords"].to(device=device, dtype=torch.float32)
        state = reconstruct_state(
            {
                "state_i": batch["state_i"],
                "coords": coords,
                "visited": batch["visited"].to(device=device),
                "first_a": batch["first_a"].to(device=device),
                "prev_a": batch["prev_a"].to(device=device),
                "lengths": batch["lengths"].to(device=device),
            },
            device=device,
        )
        fixed = model.precompute_decoder(model.encode(coords))
        step = int(batch["state_i"])
        bl = float(batch["bl_val"].view(-1)[0].item())

        # Depth-(h+1) probe: walk h seeded-random legal actions off the buffer
        # state. Deterministic given (seed, slots), so cached gt stays valid.
        for _hop in range(args.root_hop):
            hop_legal = torch.nonzero(~state.get_mask().view(-1)).view(-1).tolist()
            a_hop = int(hop_rng.choice(hop_legal))
            state = state.update(torch.tensor([a_hop], dtype=torch.long, device=device))
        step_eff = step + args.root_hop
        current = int(state.prev_a.view(-1)[0].item())

        mask = state.get_mask().view(-1)  # (N,) True=visited
        legal = torch.nonzero(~mask).view(-1).tolist()
        if len(legal) < 2:
            continue

        v_arr, g_arr, se_arr, edge_arr = [], [], [], []
        for a in legal:
            child = state.update(torch.tensor([a], dtype=torch.long, device=device))
            if own_trunk:
                v = float(model.value_from_state(fixed, child).view(-1)[0].item())
            else:
                _lp, _m, gl = model.decode_step(fixed, child, return_glimpse=True)
                v = float(model.value_head(gl).view(-1)[0].item())
            edge = 0.0 if step_eff == 0 else float(state.dist[0, current, a].item())
            if gt is None:
                samples = _rollout_ctg(model, fixed, child, args.rollouts_per_child, gen, device, mode=args.gt_mode)
                g = float(samples.mean())
                se = float(samples.std(ddof=0) / np.sqrt(len(samples)))
            else:
                mrow = (gt["node_id"] == node_id) & (gt["action"] == a)
                g = float(gt["gmean"][mrow][0]); se = float(gt["gse"][mrow][0])
            v_arr.append(v); g_arr.append(g); se_arr.append(se); edge_arr.append(edge)

        v_arr = np.array(v_arr); g_arr = np.array(g_arr)
        se_arr = np.array(se_arr); edge_arr = np.array(edge_arr)
        av_v = edge_arr + v_arr          # action value by value head
        av_g = edge_arr + g_arr          # action value by ground truth
        i_head = int(np.argmin(av_v)); i_true = int(np.argmin(av_g))
        rows.append({
            "node_id": node_id, "slot": int(slot), "step": step_eff, "hop": args.root_hop,
            "n_actions": len(legal), "bl_val": bl,
            "spearman_action": _spearman(av_v, av_g),
            "spearman_ctg": _spearman(v_arr, g_arr),
            "kendall_action": _kendall(av_v, av_g),
            "top1": int(i_head == i_true),
            "regret_raw": float(av_g[i_head] - av_g.min()),
            "regret_bl": float((av_g[i_head] - av_g.min()) / max(bl, 1e-6)),
            "rand_regret_raw": float(av_g.mean() - av_g.min()),
            "sep_raw": float(av_g.max() - av_g.min()),
            "mean_gse": float(se_arr.mean()),
        })
        if gt is None:
            gt_actions[node_id] = legal
            gt_edge[node_id] = edge_arr; gt_gmean[node_id] = g_arr
            gt_gse[node_id] = se_arr; gt_bl[node_id] = bl; gt_step[node_id] = step_eff
        if (node_id + 1) % 10 == 0:
            print(f"  {node_id + 1}/{len(slots)} nodes ({time.time() - t0:.1f}s)")

    # save gt cache
    if gt is None:
        flat_nid, flat_a, flat_e, flat_g, flat_se = [], [], [], [], []
        for nid, acts in gt_actions.items():
            for j, a in enumerate(acts):
                flat_nid.append(nid); flat_a.append(a)
                flat_e.append(gt_edge[nid][j]); flat_g.append(gt_gmean[nid][j]); flat_se.append(gt_gse[nid][j])
        np.savez(cache_path, slots=np.array(list(slots)),
                 node_id=np.array(flat_nid), action=np.array(flat_a),
                 edge=np.array(flat_e), gmean=np.array(flat_g), gse=np.array(flat_se))
        print(f"[gt] cached ground truth -> {cache_path}")

    # ---- summary ----
    def m(k):
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n=== ACTION-RANKING SUMMARY  ({args.ckpt}) ===")
    print(f"  nodes used            : {len(rows)}")
    print(f"  mean n_actions        : {m('n_actions'):.1f}")
    print(f"  Spearman (edge+v vs edge+g)  : {m('spearman_action'):.4f}   <- decision-relevant")
    print(f"  Spearman (v vs g, cost-to-go): {m('spearman_ctg'):.4f}")
    print(f"  Kendall  (edge+v vs edge+g)  : {m('kendall_action'):.4f}")
    print(f"  top-1 action match           : {m('top1'):.3f}")
    print(f"  decision regret (raw)        : {m('regret_raw'):.5f}   (bl-norm {m('regret_bl'):.5f})")
    print(f"  random-pick regret (raw)     : {m('rand_regret_raw'):.5f}   <- baseline for scale")
    print(f"  mean action separation (raw) : {m('sep_raw'):.5f}")
    print(f"  mean ground-truth SE (raw)   : {m('mean_gse'):.5f}   (want << separation)")

    out_csv = args.out_csv
    if out_csv is None:
        tag = "own" if args.value_own_encoder else ("trunk" if own_trunk else "glimpse")
        out_csv = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), f"rank_{tag}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
