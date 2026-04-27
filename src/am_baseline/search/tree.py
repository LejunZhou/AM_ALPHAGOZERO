"""MCTSNode — one node per visited partial-tour state in the search tree."""
import math
from typing import Dict, List, Optional

from am_baseline.problem.state import StateTSP


class MCTSNode:
    """A node in the MCTS tree for a single TSP instance.

    Holds a StateTSP (partial tour + feasibility mask) plus per-action statistics
    for the actions leaving this node:
      N[a] : visit count of edge (s, a)
      W[a] : total backed-up value along edge (s, a)  (in "higher=better" sign)
      Q[a] : running mean = W[a] / N[a]
      P[a] : policy prior for action a at this state  (renormalized over legal actions)
      children[a] : child MCTSNode reached by taking action a

    An entry exists in P/N/W/Q only for LEGAL (non-masked) actions — populated in
    _populate_priors() / _expand(). children[a] is created lazily on first descent.

    Node-level cache (populated at expansion, used by FPU and diagnostics):
      v_estimate : normalized remaining cost-to-go estimated for THIS state
                   (the value returned by _expand when this node was expanded).
                   math.nan until expanded.
    """

    __slots__ = ("state", "parent", "action_into_me",
                 "N", "W", "Q", "P", "children",
                 "v_estimate")

    def __init__(self,
                 state: StateTSP,
                 parent: Optional["MCTSNode"] = None,
                 action_into_me: Optional[int] = None):
        self.state: StateTSP = state
        self.parent: Optional["MCTSNode"] = parent
        self.action_into_me: Optional[int] = action_into_me
        self.N: Dict[int, int] = {}
        self.W: Dict[int, float] = {}
        self.Q: Dict[int, float] = {}
        self.P: Dict[int, float] = {}
        self.children: Dict[int, "MCTSNode"] = {}
        self.v_estimate: float = math.nan  # set when this node is expanded

    def is_terminal(self) -> bool:
        return bool(self.state.all_finished())

    def is_expanded(self) -> bool:
        """True if priors have been populated (expansion has occurred)."""
        return len(self.P) > 0

    def legal_actions(self) -> List[int]:
        """Unvisited node indices, read from the state's feasibility mask."""
        visited = self.state.visited_.view(-1).tolist()
        return [i for i, v in enumerate(visited) if not v]

    def total_visits(self) -> int:
        return sum(self.N.values())

    def running_value(self) -> float:
        """Running value estimate at this node: sum(W)/sum(N). math.nan if unvisited."""
        total_N = self.total_visits()
        if total_N == 0:
            return math.nan
        return sum(self.W.values()) / total_N
