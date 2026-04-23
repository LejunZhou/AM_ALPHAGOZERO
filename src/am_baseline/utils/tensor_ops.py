import torch
import torch.nn.functional as F


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches.
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size
    if n_batches == 1:
        return f(*args)

    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            return None
        return torch.cat(chunks, dim)

    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)
    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def cost_to_go(edge_costs):
    """
    Reverse cumulative sum along dim 1.

    Given per-edge costs of shape (batch, N) where entry [b, t] is the cost
    incurred at decoding step t+1, returns (batch, N) where entry [b, t] is
    the total remaining cost from step t+1 onward (inclusive).

    entry[:, 0]  == edge_costs.sum(dim=1)
    entry[:, -1] == edge_costs[:, -1]
    """
    return torch.flip(torch.cumsum(torch.flip(edge_costs, [1]), dim=1), [1])


def value_targets_from_edges(edge_costs):
    """
    Per-state V_CURRENT targets aligned to the decoder's glimpse indexing.

    glimpse[i] represents state s_i = partial tour with i nodes visited and
    max(0, i-1) edges traversed. The value of s_i (realized) is the cost of
    every edge still to be traversed, INCLUDING the edge selected at step i.

    Concretely, with edge_costs[k] = edge traversed at decoder step k+1
    (with edge_costs[N-1] = closing):
        target[0]         = total cost
        target[1]         = total cost     (s_1 has no traversed edges yet)
        target[i], i >= 2 = sum edge_costs[i-1:]

    Implementation: prepend a 0 (step 0 has no edge) to edge_costs, then take
    the reverse cumsum over the (N+1)-length tensor and drop the trailing
    entry (which would correspond to after-closing = 0).

    edge_costs: (B, N)  ->  targets: (B, N)
    """
    pad_edges = torch.nn.functional.pad(edge_costs, (1, 0))      # (B, N+1)
    return cost_to_go(pad_edges)[:, :-1]                         # (B, N)


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        cost, mask = get_cost_func(input, pi)
        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )
    costs = torch.cat(costs, 1)

    mincosts, argmincosts = costs.min(-1)
    minpis = pis[torch.arange(pis.size(0), device=argmincosts.device), argmincosts]

    return minpis, mincosts
