"""PUCT action selection — pure function, kept standalone for testability."""
import math

from am_baseline.search.tree import MCTSNode


def select_action(node: MCTSNode, c_puct: float, fpu_value: float) -> int:
    """AlphaGo-Zero-style PUCT selection:

        a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(Σ_b N(s,b)) / (1 + N(s,a)) ]

    `fpu_value` is the Q-init used for actions with N(s,a) = 0. The caller
    decides the strategy — see `MCTSSolver._fpu_value_for(node)`.

    node must be expanded (P populated). Only actions in node.P are considered,
    which are exactly the legal actions by construction of _expand().
    """
    assert node.is_expanded(), "PUCT called on unexpanded node"
    total_N = node.total_visits()
    sqrt_total = math.sqrt(max(total_N, 1))

    best_action = -1
    best_score = -math.inf
    for a, p_a in node.P.items():
        n_sa = node.N.get(a, 0)
        q_sa = node.Q[a] if n_sa > 0 else fpu_value
        u_sa = c_puct * p_a * sqrt_total / (1 + n_sa)
        score = q_sa + u_sa
        if score > best_score:
            best_score = score
            best_action = a
    assert best_action != -1, "PUCT: no legal action found"
    return best_action
