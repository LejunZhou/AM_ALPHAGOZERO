"""Stage 5 §V0 — greedy-rollout ranking benchmark from two probe gt caches.

Computes how well the GREEDY ROLLOUT (the production leaf eval) ranks sibling
actions against the sampled-policy E[z|s'] ground truth, purely from two
`probe_action_ranking.py` gt caches over the SAME slots:

    --gt_greedy   cache built with --gt_mode greedy   (1 deterministic rollout)
    --gt_sampled  cache built with --gt_mode sampled  (R tau=1 rollouts, mean)

This is the anchor every value head must be compared against (§V0 gate):
TSP-20 depth-1 anchors are regret 0.047 (all instances) / 0.056 (held-out).
Zero model forwards — pure cache arithmetic.

Repro:
    PYTHONPATH=src python -m scripts.rank_rollout_benchmark \
        --gt_greedy .../rank_gt_<greedy>.npz --gt_sampled .../rank_gt_<sampled>.npz
"""
import argparse

import numpy as np


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    starts = csum - counts
    return ((starts + csum - 1) / 2.0)[inv]


def _pearson(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def main():
    p = argparse.ArgumentParser(description="Greedy-rollout ranking anchor from paired gt caches.")
    p.add_argument("--gt_greedy", required=True)
    p.add_argument("--gt_sampled", required=True)
    args = p.parse_args()

    zg = np.load(args.gt_greedy, allow_pickle=True)
    zs = np.load(args.gt_sampled, allow_pickle=True)
    assert list(zg["slots"]) == list(zs["slots"]), \
        "caches were built over different slots — regenerate with matching probe args"

    sp, top1, regret, rrand, sep, gse = [], [], [], [], [], []
    for nid in np.unique(zg["node_id"]):
        mg = zg["node_id"] == nid
        ms = zs["node_id"] == nid
        assert list(zg["action"][mg]) == list(zs["action"][ms])
        av_greedy = zg["edge"][mg] + zg["gmean"][mg]   # action value per greedy rollout
        av_ez = zs["edge"][ms] + zs["gmean"][ms]       # action value per E[z|s'] (sampled)
        i_g, i_t = int(np.argmin(av_greedy)), int(np.argmin(av_ez))
        sp.append(_spearman(av_greedy, av_ez))
        top1.append(int(i_g == i_t))
        regret.append(float(av_ez[i_g] - av_ez.min()))
        rrand.append(float(av_ez.mean() - av_ez.min()))
        sep.append(float(av_ez.max() - av_ez.min()))
        gse.append(float(zs["gse"][ms].mean()))

    print(f"=== GREEDY ROLLOUT vs E[z|s'] ranking anchor  ({len(sp)} nodes) ===")
    print(f"  Spearman (action values) : {np.nanmean(sp):.4f}")
    print(f"  top-1 match              : {np.mean(top1):.3f}")
    print(f"  decision regret (raw)    : {np.mean(regret):.5f}")
    print(f"  random-pick regret (raw) : {np.mean(rrand):.5f}")
    print(f"  action separation (raw)  : {np.mean(sep):.5f}")
    print(f"  ground-truth SE (raw)    : {np.mean(gse):.5f}   (want << separation)")


if __name__ == "__main__":
    main()
