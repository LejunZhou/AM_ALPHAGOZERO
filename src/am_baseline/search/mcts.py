"""MCTSSolver — AlphaGo-Zero-style MCTS on a trained AM + value head.

Correctness invariants (verified in unit tests):
  (1) No edge double-counting: at a non-terminal leaf we use
          total_norm = state.lengths / bl_val + v(state)
      where v(state) is the V_CURRENT target the value head was trained on
      (`utils.tensor_ops.value_targets_from_edges`). V_CURRENT at state s_k = cost
      of ALL edges still to be traversed from s_k, including the upcoming edge and
      the closing edge. state.lengths = cost of edges already traversed (which is 0
      at s_0 and s_1, and grows as the tour is built).
  (2) The closing edge is included exactly once: either through
      state.get_final_cost() at terminal leaves (which adds the first↔last edge)
      or through the value head prediction at non-terminal leaves (V_CURRENT
      includes the closing edge by construction).
  (3) Single-agent minimization: Q = -total_normalized_cost (no sign flip per
      depth).
  (4) All priors are renormalized over LEGAL actions only — any Dirichlet mix
      or zero/NaN defense preserves Σ_legal P(a) = 1.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.state import StateTSP
from am_baseline.problem.tsp import TSP
from am_baseline.search.puct import select_action
from am_baseline.search.tree import MCTSNode


@dataclass
class MCTSConfig:
    # --- search budget / exploration ---
    n_simulations: int = 200
    c_puct: float = 0.05                      # routing sweet spot; AlphaGo's 1.0 is wrong here
    temperature: float = 0.0                  # 0 = argmax, >0 = sample from N^(1/τ)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0            # 0 = noise off (Stage 2 default)

    # --- leaf evaluation ---
    leaf_eval: str = 'value_head'             # 'value_head' | 'rollout'
    value_norm: str = 'bl'                    # 'bl' (per-instance greedy) | 'sqrt_n'

    # --- FPU (first-play urgency): Q_init for unvisited actions ---
    # 'fallback'   : constant `fpu_fallback` everywhere (useful for sweeping)
    # 'running_q'  : node's own running mean sum(W)/sum(N); falls back when N=0
    # 'node_value' : -v_estimate of this node (needs expansion value cached)
    fpu_mode: str = 'running_q'
    fpu_fallback: float = -1.0                # typical Q on TSP is ≈ -1 (normalized)

    # --- final-action selection at root (diagnostic) ---
    # 'visits' : argmax N (AlphaGo default; robust when K is large)
    # 'q'      : argmax Q among visited actions (diagnostic; sensitive to Q noise)
    root_select: str = 'visits'

    # --- tree reuse across tour-steps ---
    tree_reuse: bool = False                  # Phase A.5 optimization

    seed: Optional[int] = None


class MCTSSolver:
    """Given a trained AttentionModel, solve TSP instances by MCTS.

    Scope matches `_plans/stage2_plan.md`:
      - One tree per instance, sequential across instances.
      - K simulations per tour-step.
      - Leaf eval: value head by default; greedy rollout as optional fallback.
    """

    VALID_LEAF_EVAL = {'value_head', 'rollout'}
    VALID_VALUE_NORM = {'bl', 'sqrt_n'}
    VALID_FPU_MODE = {'fallback', 'running_q', 'node_value'}
    VALID_ROOT_SELECT = {'visits', 'q'}

    def __init__(self,
                 model: AttentionModel,
                 cfg: MCTSConfig,
                 device: Optional[torch.device] = None):
        assert cfg.leaf_eval in self.VALID_LEAF_EVAL, cfg.leaf_eval
        assert cfg.value_norm in self.VALID_VALUE_NORM, cfg.value_norm
        assert cfg.fpu_mode in self.VALID_FPU_MODE, cfg.fpu_mode
        assert cfg.root_select in self.VALID_ROOT_SELECT, cfg.root_select

        self.model = model
        self.cfg = cfg
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()
        if cfg.leaf_eval == 'value_head' and model.value_head is None:
            raise ValueError(
                "cfg.leaf_eval='value_head' but model has no value_head. "
                "Either pass a checkpoint trained with value_enabled=True, "
                "or use cfg.leaf_eval='rollout'."
            )
        self.rng = np.random.default_rng(cfg.seed)

    @torch.no_grad()
    def solve_batch(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """inputs: (B, N, 2). Returns (costs: (B,), tours: (B, N) long)."""
        inputs = inputs.to(self.device)
        B, N, _ = inputs.shape

        bl_vals = self._compute_bl_val_batch(inputs)  # (B,) real units

        costs = torch.empty(B, device=self.device)
        tours = torch.empty(B, N, dtype=torch.long, device=self.device)
        for i in range(B):
            cost_i, tour_i = self.solve_instance(inputs[i:i+1], bl_val=float(bl_vals[i].item()))
            costs[i] = cost_i
            tours[i] = tour_i
        return costs, tours

    def _compute_bl_val_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Per-instance `bl_val` used to normalize cost-to-go. One batched pass."""
        B, N, _ = inputs.shape
        if self.cfg.value_norm == 'sqrt_n':
            return torch.full((B,), math.sqrt(N), device=self.device)
        prev = self.model.decoder.decode_type
        self.model.set_decode_type('greedy')
        cost, _ = self.model(inputs)
        if prev is not None:
            self.model.set_decode_type(prev)
        return cost.detach()

    @torch.no_grad()
    def solve_instance(self,
                       input_1: torch.Tensor,
                       bl_val: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """input_1: (1, N, 2). Returns (cost: 0-d tensor, tour: (N,) long)."""
        assert input_1.dim() == 3 and input_1.size(0) == 1, "solve_instance expects (1, N, 2)"

        if bl_val is None:
            bl_val = float(self._compute_bl_val_batch(input_1).item())

        embeddings = self.model.encode(input_1)
        fixed = self.model.precompute_decoder(embeddings)

        state = TSP.make_state(input_1)
        tour_actions = []
        root: Optional[MCTSNode] = None  # reused across tour-steps when cfg.tree_reuse

        while not state.all_finished():
            # Obtain root for this tour-step: either reuse prior subtree or build fresh.
            if root is None or not self.cfg.tree_reuse:
                root = MCTSNode(state=state, parent=None, action_into_me=None)
            else:
                # Tree reuse: root was set to the chosen child at end of prior step.
                # Detach from former parent chain.
                root.parent = None
                root.action_into_me = None

            if not root.is_expanded() and not root.is_terminal():
                self._populate_priors(root, fixed)

            if self.cfg.dirichlet_epsilon > 0 and not root.is_terminal():
                self._apply_dirichlet(root)

            for _ in range(self.cfg.n_simulations):
                self._simulate(root, fixed, bl_val)

            a = self._pick_root_action(root)
            tour_actions.append(a)
            state = state.update(torch.tensor([a], dtype=torch.long, device=self.device))

            if self.cfg.tree_reuse and a in root.children:
                # Advance root; retain subtree statistics below.
                root = root.children[a]
            else:
                root = None  # fresh tree next iteration

        cost = state.get_final_cost().view(-1)[0]
        tour = torch.tensor(tour_actions, dtype=torch.long, device=self.device)
        return cost, tour

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _simulate(self, root: MCTSNode, fixed, bl_val: float) -> None:
        """One MCTS simulation: select → expand/evaluate → backup."""
        path = []  # list of (parent_node, action_taken)
        node = root

        # --- Selection: descend through expanded non-terminal nodes. --- #
        while node.is_expanded() and not node.is_terminal():
            fpu = self._fpu_value_for(node)
            a = select_action(node, self.cfg.c_puct, fpu)
            path.append((node, a))
            child = node.children.get(a)
            if child is None:
                new_state = node.state.update(
                    torch.tensor([a], dtype=torch.long, device=self.device)
                )
                child = MCTSNode(state=new_state, parent=node, action_into_me=a)
                node.children[a] = child
            node = child

        # --- Evaluation at the leaf ---
        # Invariant: use ONE of the two cost-accounting paths, never both.
        if node.is_terminal():
            # Terminal leaf: exact realized cost = lengths + closing edge.
            total_real = float(node.state.get_final_cost().view(-1)[0].item())
            total_norm = total_real / bl_val
            # Still cache v_estimate at a terminal leaf = 0 remaining cost (nothing
            # left to traverse). Useful if FPU ever queries this.
            node.v_estimate = 0.0
        else:
            # Non-terminal leaf: expand (populate priors + cache v_estimate).
            v_remaining_norm = self._expand(node, fixed, bl_val)
            node.v_estimate = v_remaining_norm
            c_path_real = float(node.state.lengths.view(-1)[0].item())
            total_norm = c_path_real / bl_val + v_remaining_norm

        # Backup: higher Q == lower cost → negate.
        value_for_backup = -total_norm
        for parent, a in path:
            parent.N[a] = parent.N.get(a, 0) + 1
            parent.W[a] = parent.W.get(a, 0.0) + value_for_backup
            parent.Q[a] = parent.W[a] / parent.N[a]

    def _fpu_value_for(self, node: MCTSNode) -> float:
        """Choose the Q_init for unvisited actions at `node` per cfg.fpu_mode."""
        mode = self.cfg.fpu_mode
        if mode == 'fallback':
            return self.cfg.fpu_fallback
        if mode == 'running_q':
            total_N = node.total_visits()
            if total_N > 0:
                return sum(node.W.values()) / total_N
            return self.cfg.fpu_fallback
        if mode == 'node_value':
            # FPU = -v(node): unvisited actions inherit the node's own cost-to-go
            # estimate (negated because Q orientation is "higher=better = -cost").
            if math.isfinite(node.v_estimate):
                # Add current path cost (normalized) so unvisited Q is on the same
                # scale as visited actions' backed-up values at this node.
                c_path_real = float(node.state.lengths.view(-1)[0].item())
                # bl_val is instance-constant; rely on caller to pass the same
                # via self state. We read from the last call's bl_val via the
                # path cost + v_estimate (already normalized).
                # Actually v_estimate alone represents remaining norm; total norm
                # from this node's subtree = c_path_norm + v_estimate. But we
                # need bl_val for c_path. Callers use running_q or fallback for
                # correctness; node_value mode is best-effort and uses
                # running_q as a safe fallback when c_path normalization isn't
                # trivially available at this call site.
                total_N = node.total_visits()
                if total_N > 0:
                    return sum(node.W.values()) / total_N
                # Unvisited node: use -v_estimate as optimistic estimate. This
                # treats the node's intrinsic "remaining quality" as the FPU.
                return -node.v_estimate
            return self.cfg.fpu_fallback
        raise ValueError(f"unknown fpu_mode: {mode}")

    def _populate_priors(self, node: MCTSNode, fixed) -> None:
        """Populate `node.P` for all legal actions, renormalized safely."""
        assert not node.is_terminal(), "_populate_priors called on terminal node"
        log_p, mask = self.model.decoder.decode_step(fixed, node.state, return_glimpse=False)
        self._fill_priors_from_logp(node, log_p, mask)

    def _expand(self, node: MCTSNode, fixed, bl_val: float) -> float:
        """Populate `node.P` AND return estimated NORMALIZED cost-to-go from node.state.

        `bl_val` is only used by the rollout branch to normalize realized
        remaining cost; the value_head branch is already in normalized space.
        """
        assert not node.is_terminal(), "_expand called on terminal node"
        log_p, mask, glimpse = self.model.decoder.decode_step(
            fixed, node.state, return_glimpse=True
        )
        self._fill_priors_from_logp(node, log_p, mask)

        if self.cfg.leaf_eval == 'value_head':
            return float(self.model.value_head(glimpse).view(-1)[0].item())
        if self.cfg.leaf_eval == 'rollout':
            remaining_real = self._rollout_remaining_real(node.state, fixed)
            return remaining_real / bl_val
        raise ValueError(f"Unknown leaf_eval: {self.cfg.leaf_eval}")

    def _fill_priors_from_logp(self, node: MCTSNode, log_p, mask) -> None:
        """Extract legal-action priors, renormalize, defend against NaN / zero-sum.

        Invariants after this call:
            - node.P contains exactly the legal actions (where mask is False).
            - Σ_a node.P[a] == 1 (up to fp).
            - No NaN, no negatives.
        """
        probs = log_p.exp().view(-1)
        mask_vec = mask.view(-1)

        legal: list[int] = []
        raw: list[float] = []
        for a in range(probs.size(0)):
            if not bool(mask_vec[a].item()):
                p = float(probs[a].item())
                if not math.isfinite(p):
                    p = 0.0
                elif p < 0.0:
                    p = 0.0
                legal.append(a)
                raw.append(p)

        assert legal, "_fill_priors_from_logp: no legal actions but node not terminal"

        total = sum(raw)
        if total > 0 and math.isfinite(total):
            for a, p in zip(legal, raw):
                node.P[a] = p / total
        else:
            # Fallback: uniform over legal (rare; happens if softmax underflowed).
            u = 1.0 / len(legal)
            for a in legal:
                node.P[a] = u

    def _rollout_remaining_real(self, state: StateTSP, fixed) -> float:
        """Greedy rollout from `state` to terminal. Returns remaining REAL cost."""
        c_path_start = float(state.lengths.view(-1)[0].item())
        cur = state
        while not cur.all_finished():
            log_p, mask = self.model.decoder.decode_step(fixed, cur, return_glimpse=False)
            a = int(log_p.view(-1).argmax().item())
            cur = cur.update(torch.tensor([a], dtype=torch.long, device=self.device))
        total_real = float(cur.get_final_cost().view(-1)[0].item())
        return total_real - c_path_start

    def _pick_root_action(self, root: MCTSNode) -> int:
        """Final action from the root. If K=0 (no sims), falls back to argmax prior.

        Defined behavior (explicit):
            - cfg.n_simulations == 0  →  argmax P(root, a) over legal actions.
              This makes MCTS(K=0, τ=0) match model greedy decode exactly.
            - cfg.root_select == 'visits' and τ==0  →  argmax N (ties broken by action order).
            - cfg.root_select == 'visits' and τ>0   →  sample ∝ N^(1/τ).
            - cfg.root_select == 'q'                →  argmax Q among visited actions.
        """
        if not root.N:
            # No simulation ran — fall back to argmax prior.
            assert self.cfg.n_simulations == 0 or root.is_terminal(), \
                "no visits at root but K>0 and not terminal — bug?"
            legal = sorted(root.P.keys())
            assert legal, "root has no legal actions and is not terminal"
            return max(legal, key=lambda a: root.P[a])

        if self.cfg.root_select == 'q':
            # argmax Q among visited actions.
            return max(root.N.keys(), key=lambda a: root.Q[a])

        # 'visits'
        actions = sorted(root.N.keys())
        counts = np.array([root.N[a] for a in actions], dtype=np.float64)
        if self.cfg.temperature == 0.0 or counts.max() == 0:
            return actions[int(counts.argmax())]
        counts_pow = counts ** (1.0 / self.cfg.temperature)
        probs = counts_pow / counts_pow.sum()
        return int(self.rng.choice(actions, p=probs))

    def _apply_dirichlet(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into root priors; renormalize for safety."""
        actions = sorted(root.P.keys())
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(actions))
        eps = self.cfg.dirichlet_epsilon
        mixed = {}
        total = 0.0
        for a, eta in zip(actions, noise):
            mixed[a] = (1.0 - eps) * root.P[a] + eps * float(eta)
            total += mixed[a]
        if total > 0 and math.isfinite(total):
            for a in actions:
                root.P[a] = mixed[a] / total
        else:
            u = 1.0 / len(actions)
            for a in actions:
                root.P[a] = u
