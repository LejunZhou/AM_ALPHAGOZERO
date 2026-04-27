# AM AlphaGoZero

This project combines a neural Attention Model for the Traveling Salesman Problem
with Monte Carlo Tree Search. The Attention Model proposes likely next cities in a
tour, and MCTS searches around those proposals to find lower-cost tours than
greedy decoding alone.

## How MCTS Improves Decoding

The search treats TSP construction as a sequence of small decisions. At any point,
the current state is a partial tour, and the next action is choosing one unvisited
city.

For each step of the real tour, MCTS builds a search tree from the current partial
tour:

1. A tree node represents a partial tour.
2. An action represents choosing the next unvisited city.
3. The Attention Model gives prior probabilities over legal next cities.
4. MCTS runs simulations through the tree before committing to the next city.
5. During each simulation, PUCT balances two goals: follow cities the model thinks
   are promising, and explore alternatives that may lead to better complete tours.
6. When a simulation reaches a leaf, the remaining tour is scored by completing it
   with greedy rollout. A learned value head can also be used as a faster diagnostic
   leaf evaluator.
7. The completed-tour score is backed up through the simulated path, so choices
   that repeatedly lead to shorter tours receive stronger search statistics.
8. After the simulations finish, the next real city is chosen by the most visited
   root action.
9. The chosen subtree is reused for the next tour step, and the process repeats
   until every city has been visited.

Internally, lower tour cost is stored as a higher search value, so the tree can use
standard "larger is better" selection logic while still solving a minimization
problem.

## Default Search Behavior

The routing search uses a small PUCT constant because the trained model is already
strong and cost differences between good actions are subtle. The default test-time
behavior is:

- `c_puct=0.05` for routing-scale exploration.
- `root_select=visits` to choose the most searched action.
- `tree_reuse=True` to keep useful statistics after committing to a city.
- `leaf_eval=rollout` for test-time search, because greedy rollout gave stronger
  tour quality than value-head leaf evaluation in the experiments.

The value head remains useful for diagnostics and for future training loops where
the model learns from MCTS-generated states.

## Running MCTS

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m scripts.run_mcts `
  --model outputs/tsp_20/your_checkpoint/epoch-99.pt `
  --graph_size 20 `
  --val_size 1000 `
  --seed 1234 `
  --n_simulations 200 `
  --c_puct 0.05 `
  --temperature 0.0 `
  --leaf_eval rollout `
  --fpu_mode running_q `
  --fpu_fallback -1.0 `
  --root_select visits `
  --tree_reuse `
  --output_csv outputs/stage2/tsp20_mcts_rollout.csv
```

The main search implementation is in `src/am_baseline/search/`, and the runnable
CLI is `src/scripts/run_mcts.py`.

## Current Status

The MCTS implementation produces valid TSP tours and improves over greedy decoding
in the project experiments on TSP-20 and TSP-50. For test-time search, greedy
rollout leaf evaluation produced better tour quality than value-head leaf
evaluation, so rollout is the recommended setting when compute allows.
