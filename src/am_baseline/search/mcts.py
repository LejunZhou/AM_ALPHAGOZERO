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
    simulation_batch_size: int = 1            # C++ backend only; 1 = sequential reference
    virtual_loss_weight: float = 3.0          # C++ batched mode only; 0 = virtual visits only
    virtual_loss_margin: float = 0.5          # temporary Q penalty per pending edge
    c_puct: float = 0.05                      # routing sweet spot; AlphaGo's 1.0 is wrong here
    temperature: float = 0.0                  # 0 = argmax, >0 = sample from N^(1/τ)
    # Per-tour-step temperature schedule (Stage 4 Phase E).
    # Accepts {None, 'const', 'step30', 'step50'}. Semantics:
    #   None | 'const' : τ = cfg.temperature for all steps (preserves Stage 2/3 behavior).
    #   'step30'       : τ = cfg.temperature for first ⌈0.3·N⌉ steps, τ = 0 thereafter.
    #   'step50'       : τ = cfg.temperature for first ⌈0.5·N⌉ steps, τ = 0 thereafter.
    # Affects ONLY action-selection σ_t at the root; the stored visit dist π_t
    # (when Phase A is wired) remains raw τ=1.
    temperature_schedule: Optional[str] = None
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0            # 0 = noise off (Stage 2 default)

    # --- leaf evaluation ---
    # Default `rollout` per Stage 2 leaf-eval ablation (uniformly +12-22pp
    # gap reduction over `value_head` at every matched K on TSP-20/TSP-50);
    # `value_head` remains available for diagnostics and is required by
    # Stage 4 training-loop semantics.
    leaf_eval: str = 'rollout'                # 'value_head' | 'rollout'
    value_norm: str = 'bl'                    # 'bl' (per-instance greedy) | 'sqrt_n'

    # --- FPU (first-play urgency): Q_init for unvisited actions ---
    # 'fallback'   : constant `fpu_fallback` everywhere (useful for sweeping)
    # 'running_q'  : node's own running mean sum(W)/sum(N); falls back when N=0
    # 'node_value' : -(c_path_norm + v_estimate) of this node — total-from-root scale
    fpu_mode: str = 'running_q'
    fpu_fallback: float = -1.0                # typical Q on TSP is ≈ -1 (normalized)

    # --- final-action selection at root (diagnostic) ---
    # 'visits' : argmax N (AlphaGo default; robust when K is large)
    # 'q'      : argmax Q among visited actions (diagnostic; sensitive to Q noise)
    root_select: str = 'visits'

    # --- tree reuse across tour-steps ---
    # Default True per Stage 2 tree-reuse diagnostic (47/100 wins, +0.149%
    # quality, 17% wall-clock saved on TSP-20). Strictly Pareto-better.
    tree_reuse: bool = True

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
    # None and 'const' are treated identically (constant τ = cfg.temperature).
    VALID_TEMPERATURE_SCHEDULE = {None, 'const', 'step30', 'step50'}

    def __init__(self,
                 model: AttentionModel,
                 cfg: MCTSConfig,
                 device: Optional[torch.device] = None):
        self._validate_config(cfg, model)
        if cfg.simulation_batch_size != 1:
            raise ValueError(
                "simulation_batch_size > 1 is only supported by the C++ MCTS backend. "
                "Use --backend cpp, or leave simulation_batch_size=1 for the Python backend."
            )

        self.model = model
        self.cfg = cfg
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()
        self.rng = np.random.default_rng(cfg.seed)

        # Stage 3 forward-pass instrumentation (search-efficiency metric).
        # Reset at the start of every `solve_instance` call; readable by the
        # caller after the call returns. Counts only MCTS-internal work —
        # the greedy bl_val pass is excluded (it's the baseline normalizer,
        # not part of the search budget).
        #   fwd_count_decode  : total decode_step calls (priors + expansions + rollout steps)
        #   fwd_count_value   : total value_head MLP calls (only when leaf_eval='value_head')
        #   fwd_count_rollout : decode_step calls inside rollouts (subset of fwd_count_decode)
        self.fwd_count_decode = 0
        self.fwd_count_value = 0
        self.fwd_count_rollout = 0

    @classmethod
    def _validate_config(cls, cfg: MCTSConfig, model: AttentionModel) -> None:
        """Reject invalid or scale-incompatible configs at construction time.

        Raises ValueError (not assert) so misuse surfaces under `python -O` too.
        """
        if cfg.leaf_eval not in cls.VALID_LEAF_EVAL:
            raise ValueError(f"cfg.leaf_eval={cfg.leaf_eval!r} not in {cls.VALID_LEAF_EVAL}")
        if cfg.simulation_batch_size < 1:
            raise ValueError(
                f"cfg.simulation_batch_size={cfg.simulation_batch_size!r} must be >= 1"
            )
        if cfg.virtual_loss_weight < 0:
            raise ValueError(
                f"cfg.virtual_loss_weight={cfg.virtual_loss_weight!r} must be >= 0"
            )
        if cfg.virtual_loss_margin < 0:
            raise ValueError(
                f"cfg.virtual_loss_margin={cfg.virtual_loss_margin!r} must be >= 0"
            )
        if cfg.value_norm not in cls.VALID_VALUE_NORM:
            raise ValueError(f"cfg.value_norm={cfg.value_norm!r} not in {cls.VALID_VALUE_NORM}")
        if cfg.fpu_mode not in cls.VALID_FPU_MODE:
            raise ValueError(f"cfg.fpu_mode={cfg.fpu_mode!r} not in {cls.VALID_FPU_MODE}")
        if cfg.root_select not in cls.VALID_ROOT_SELECT:
            raise ValueError(f"cfg.root_select={cfg.root_select!r} not in {cls.VALID_ROOT_SELECT}")
        if cfg.temperature_schedule not in cls.VALID_TEMPERATURE_SCHEDULE:
            raise ValueError(
                f"cfg.temperature_schedule={cfg.temperature_schedule!r} "
                f"not in {cls.VALID_TEMPERATURE_SCHEDULE}"
            )
        if cfg.leaf_eval == 'value_head' and model.value_head is None:
            raise ValueError(
                "cfg.leaf_eval='value_head' but model has no value_head. "
                "Either pass a checkpoint trained with value_enabled=True, "
                "or use cfg.leaf_eval='rollout'."
            )
        # Scale-compatibility check: the value head was trained against
        # `bl_val_training`-normalized targets (≈ realized cost / greedy cost
        # ≈ 1.0). Combining its raw output with `bl_val = sqrt(N)` path
        # normalization mixes incompatible scales inside `total_norm` in
        # `_simulate`. Rollout returns `remaining_real / bl_val` and stays
        # internally consistent under any value_norm.
        if cfg.value_norm == 'sqrt_n' and cfg.leaf_eval == 'value_head':
            raise ValueError(
                "value_norm='sqrt_n' is incompatible with leaf_eval='value_head': "
                "the value head was trained in bl-normalized units, so its raw output "
                "would be combined with sqrt(N)-scaled path costs, mixing units in PUCT. "
                "Use leaf_eval='rollout' with value_norm='sqrt_n'."
            )

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

        # Reset forward-pass counters for this instance.
        self.fwd_count_decode = 0
        self.fwd_count_value = 0
        self.fwd_count_rollout = 0

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
                self._populate_priors(root, fixed, bl_val)

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
            fpu = self._fpu_value_for(node, bl_val)
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

    def _fpu_value_for(self, node: MCTSNode, bl_val: float) -> float:
        """Choose the Q_init for unvisited actions at `node` per cfg.fpu_mode.

        All three modes return values on the same "higher Q = better tour"
        scale that backed-up Q uses (Q = -total_normalized_cost from root).
        `bl_val` is needed by `node_value` to normalize the path-cost term.
        """
        mode = self.cfg.fpu_mode
        if mode == 'fallback':
            return self.cfg.fpu_fallback
        if mode == 'running_q':
            total_N = node.total_visits()
            if total_N > 0:
                return sum(node.W.values()) / total_N
            return self.cfg.fpu_fallback
        if mode == 'node_value':
            # FPU = -(c_path_norm + v_estimate): unvisited actions inherit the
            # estimated total-from-root cost through this node, on the same
            # scale as Q values backed up via PUCT.
            if math.isfinite(node.v_estimate):
                c_path_norm = float(node.state.lengths.view(-1)[0].item()) / bl_val
                return -(c_path_norm + node.v_estimate)
            return self.cfg.fpu_fallback
        raise ValueError(f"unknown fpu_mode: {mode}")

    def _populate_priors(self, node: MCTSNode, fixed, bl_val: float) -> None:
        """Populate `node.P` for all legal actions AND cache `node.v_estimate`.

        `v_estimate` is computed in the same way `_expand` does (matching
        `cfg.leaf_eval`) so the root's FPU values are on the same scale
        regardless of leaf-eval mode. Without this caching the root would
        carry `v_estimate=NaN` and `fpu_mode='node_value'` would silently
        fall back to `running_q` / `fpu_fallback`. `bl_val` is needed only
        by the rollout branch (value_head returns normalized space directly).
        """
        assert not node.is_terminal(), "_populate_priors called on terminal node"
        log_p, mask, glimpse = self.model.decoder.decode_step(
            fixed, node.state, return_glimpse=True
        )
        self.fwd_count_decode += 1
        self._fill_priors_from_logp(node, log_p, mask)

        if self.cfg.leaf_eval == 'value_head':
            node.v_estimate = float(self.model.value_head(glimpse).view(-1)[0].item())
            self.fwd_count_value += 1
        elif self.cfg.leaf_eval == 'rollout':
            remaining_real = self._rollout_remaining_real(node.state, fixed)
            node.v_estimate = remaining_real / bl_val
        else:
            raise ValueError(f"Unknown leaf_eval: {self.cfg.leaf_eval}")

    def _expand(self, node: MCTSNode, fixed, bl_val: float) -> float:
        """Populate `node.P` AND return estimated NORMALIZED cost-to-go from node.state.

        `bl_val` is only used by the rollout branch to normalize realized
        remaining cost; the value_head branch is already in normalized space.
        """
        assert not node.is_terminal(), "_expand called on terminal node"
        log_p, mask, glimpse = self.model.decoder.decode_step(
            fixed, node.state, return_glimpse=True
        )
        self.fwd_count_decode += 1
        self._fill_priors_from_logp(node, log_p, mask)

        if self.cfg.leaf_eval == 'value_head':
            v = float(self.model.value_head(glimpse).view(-1)[0].item())
            self.fwd_count_value += 1
            return v
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
            self.fwd_count_decode += 1
            self.fwd_count_rollout += 1
            a = int(log_p.view(-1).argmax().item())
            cur = cur.update(torch.tensor([a], dtype=torch.long, device=self.device))
        total_real = float(cur.get_final_cost().view(-1)[0].item())
        return total_real - c_path_start

    @staticmethod
    def _resolve_tau(cfg: 'MCTSConfig', step: int, n: int) -> float:
        """Look up the per-tour-step temperature τ under cfg.temperature_schedule.

        Behavior (mirrored exactly in the C++ backend):
            - None or 'const' : τ = cfg.temperature for all steps (Stage 2/3 default).
            - 'step30'        : τ = cfg.temperature for step < ⌈0.3·N⌉, else τ = 0.
            - 'step50'        : τ = cfg.temperature for step < ⌈0.5·N⌉, else τ = 0.

        `step` is the 0-indexed tour-step (state.i at the root); `n` is the graph
        size. The schedule affects ONLY the σ_t action-selection draw at the root;
        the raw visit counts N(s_t, ·) used by Phase C as the π_t target are not
        modified here.
        """
        sched = cfg.temperature_schedule
        if sched is None or sched == 'const':
            return float(cfg.temperature)
        if sched == 'step30':
            cutoff = math.ceil(0.3 * n)
            return float(cfg.temperature) if step < cutoff else 0.0
        if sched == 'step50':
            cutoff = math.ceil(0.5 * n)
            return float(cfg.temperature) if step < cutoff else 0.0
        raise ValueError(f"unknown temperature_schedule: {sched!r}")

    def _pick_root_action(self, root: MCTSNode) -> int:
        """Final action from the root. If K=0 (no sims), falls back to argmax prior.

        Defined behavior (explicit):
            - cfg.n_simulations == 0  →  argmax P(root, a) over legal actions.
              This makes MCTS(K=0, τ=0) match model greedy decode exactly.
            - cfg.root_select == 'visits' and τ==0  →  argmax N (ties broken by action order).
            - cfg.root_select == 'visits' and τ>0   →  sample ∝ N^(1/τ).
            - cfg.root_select == 'q'                →  argmax Q among visited actions.

        τ is looked up via `_resolve_tau` from `cfg.temperature_schedule`; the
        scalar `cfg.temperature` is recovered when the schedule is None/'const'.
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
        step = int(root.state.i.view(-1)[0].item())
        n = int(root.state.loc.size(1))
        tau = self._resolve_tau(self.cfg, step, n)

        actions = sorted(root.N.keys())
        counts = np.array([root.N[a] for a in actions], dtype=np.float64)
        if tau == 0.0 or counts.max() == 0:
            return actions[int(counts.argmax())]
        counts_pow = counts ** (1.0 / tau)
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
