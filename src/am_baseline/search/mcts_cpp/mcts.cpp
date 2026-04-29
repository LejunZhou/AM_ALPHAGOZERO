#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace am_mcts_cpp {

namespace {

template <typename T>
T get_or(py::dict dict, const char* key, T fallback) {
  if (!dict.contains(py::str(key))) {
    return fallback;
  }
  return py::cast<T>(dict[py::str(key)]);
}

std::vector<double> sequence_to_doubles(py::handle handle) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(handle);
  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(py::len(seq)));
  for (py::handle item : seq) {
    values.push_back(py::cast<double>(item));
  }
  return values;
}

std::vector<unsigned char> sequence_to_bools(py::handle handle) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(handle);
  std::vector<unsigned char> values;
  values.reserve(static_cast<std::size_t>(py::len(seq)));
  for (py::handle item : seq) {
    values.push_back(py::cast<bool>(item) ? 1 : 0);
  }
  return values;
}

}  // namespace

Config Config::from_python(py::dict cfg) {
  Config out;
  out.n_simulations = get_or<int>(cfg, "n_simulations", out.n_simulations);
  out.simulation_batch_size = get_or<int>(cfg, "simulation_batch_size", out.simulation_batch_size);
  out.virtual_loss_weight = get_or<double>(cfg, "virtual_loss_weight", out.virtual_loss_weight);
  out.virtual_loss_margin = get_or<double>(cfg, "virtual_loss_margin", out.virtual_loss_margin);
  out.c_puct = get_or<double>(cfg, "c_puct", out.c_puct);
  out.temperature = get_or<double>(cfg, "temperature", out.temperature);
  out.dirichlet_alpha = get_or<double>(cfg, "dirichlet_alpha", out.dirichlet_alpha);
  out.dirichlet_epsilon = get_or<double>(cfg, "dirichlet_epsilon", out.dirichlet_epsilon);
  out.leaf_eval = get_or<std::string>(cfg, "leaf_eval", out.leaf_eval);
  out.value_norm = get_or<std::string>(cfg, "value_norm", out.value_norm);
  out.fpu_mode = get_or<std::string>(cfg, "fpu_mode", out.fpu_mode);
  out.fpu_fallback = get_or<double>(cfg, "fpu_fallback", out.fpu_fallback);
  out.root_select = get_or<std::string>(cfg, "root_select", out.root_select);
  out.tree_reuse = get_or<bool>(cfg, "tree_reuse", out.tree_reuse);
  out.seed = get_or<std::uint64_t>(cfg, "seed", out.seed);
  if (out.simulation_batch_size < 1) {
    throw std::runtime_error("simulation_batch_size must be >= 1");
  }
  if (out.virtual_loss_weight < 0.0) {
    throw std::runtime_error("virtual_loss_weight must be >= 0");
  }
  if (out.virtual_loss_margin < 0.0) {
    throw std::runtime_error("virtual_loss_margin must be >= 0");
  }
  return out;
}

TspState TspState::initial(std::shared_ptr<const std::vector<double>> coords, int n) {
  TspState state;
  state.n = n;
  state.coords = std::move(coords);
  state.visited.assign(static_cast<std::size_t>(n), 0);
  state.tour.reserve(static_cast<std::size_t>(n));
  return state;
}

bool TspState::all_finished() const {
  return step >= n;
}

double TspState::distance(int a, int b) const {
  const double dx = (*coords)[static_cast<std::size_t>(2 * a)] - (*coords)[static_cast<std::size_t>(2 * b)];
  const double dy = (*coords)[static_cast<std::size_t>(2 * a + 1)] - (*coords)[static_cast<std::size_t>(2 * b + 1)];
  return std::sqrt(dx * dx + dy * dy);
}

double TspState::final_cost() const {
  if (!all_finished()) {
    throw std::runtime_error("final_cost called on a non-terminal TSP state");
  }
  if (n <= 1) {
    return length;
  }
  return length + distance(prev, first);
}

TspState TspState::updated(int action) const {
  TspState next = *this;
  next.update_in_place(action);
  return next;
}

void TspState::update_in_place(int action) {
  if (action < 0 || action >= n) {
    throw std::runtime_error("TSP action out of range");
  }
  if (visited[static_cast<std::size_t>(action)] != 0) {
    throw std::runtime_error("TSP action revisits an already visited city");
  }
  if (step > 0) {
    length += distance(prev, action);
  } else {
    first = action;
  }
  prev = action;
  visited[static_cast<std::size_t>(action)] = 1;
  tour.push_back(action);
  step += 1;
}

py::dict TspState::to_python() const {
  py::dict out;
  out["step"] = step;
  out["first"] = first;
  out["prev"] = prev;
  out["length"] = length;

  py::list visited_list;
  for (unsigned char v : visited) {
    visited_list.append(py::bool_(v != 0));
  }
  out["visited"] = visited_list;

  py::list tour_list;
  for (int action : tour) {
    tour_list.append(action);
  }
  out["tour"] = tour_list;
  return out;
}

Node::Node(const TspState& state_)
    : state(state_),
      n_visits(static_cast<std::size_t>(state_.n), 0),
      virtual_n(static_cast<std::size_t>(state_.n), 0),
      w_total(static_cast<std::size_t>(state_.n), 0.0),
      q_value(static_cast<std::size_t>(state_.n), 0.0),
      prior(static_cast<std::size_t>(state_.n), 0.0),
      has_prior(static_cast<std::size_t>(state_.n), 0),
      children(static_cast<std::size_t>(state_.n)),
      v_estimate(std::numeric_limits<double>::quiet_NaN()),
      pending_eval(false) {}

bool Node::is_terminal() const {
  return state.all_finished();
}

bool Node::is_expanded() const {
  return std::any_of(has_prior.begin(), has_prior.end(), [](unsigned char v) { return v != 0; });
}

int Node::total_visits() const {
  return std::accumulate(n_visits.begin(), n_visits.end(), 0);
}

int Node::total_effective_visits() const {
  int total = 0;
  for (std::size_t i = 0; i < n_visits.size(); ++i) {
    total += n_visits[i] + virtual_n[i];
  }
  return total;
}

double Node::sum_w() const {
  return std::accumulate(w_total.begin(), w_total.end(), 0.0);
}

Solver::Solver(Config cfg, py::function evaluator, py::object rollout_evaluator)
    : cfg_(std::move(cfg)),
      evaluator_(std::move(evaluator)),
      rollout_evaluator_(std::move(rollout_evaluator)),
      rng_(cfg_.seed) {}

py::dict Solver::solve_instance(std::shared_ptr<const std::vector<double>> coords,
                                int n,
                                double bl_val) {
  counters_ = Counters{};
  TspState state = TspState::initial(std::move(coords), n);
  std::unique_ptr<Node> root;
  std::vector<int> tour;
  tour.reserve(static_cast<std::size_t>(n));

  while (!state.all_finished()) {
    if (!root || !cfg_.tree_reuse) {
      root = std::make_unique<Node>(state);
    }

    if (!root->is_expanded() && !root->is_terminal()) {
      populate_priors(*root, bl_val);
    }

    if (cfg_.dirichlet_epsilon > 0.0 && !root->is_terminal()) {
      apply_dirichlet(*root);
    }

    if (cfg_.simulation_batch_size == 1) {
      for (int i = 0; i < cfg_.n_simulations; ++i) {
        simulate(*root, bl_val);
      }
    } else {
      simulate_batched(*root, bl_val);
    }

    counters_.max_virtual_visits_remaining =
        std::max<long long>(counters_.max_virtual_visits_remaining, max_virtual_visits(*root));

    const int action = pick_root_action(*root);
    tour.push_back(action);
    state.update_in_place(action);

    if (cfg_.tree_reuse && root->children[static_cast<std::size_t>(action)]) {
      root = std::move(root->children[static_cast<std::size_t>(action)]);
    } else {
      root.reset();
    }
  }

  py::list py_tour;
  for (int action : tour) {
    py_tour.append(action);
  }

  py::dict out;
  out["cost"] = state.final_cost();
  out["tour"] = py_tour;
  out["decode_steps"] = counters_.decode;
  out["rollout_steps"] = counters_.rollout;
  out["value_calls"] = counters_.value;
  out["batch_eval_calls"] = counters_.batch_eval_calls;
  out["batch_eval_rows"] = counters_.batch_eval_rows;
  out["pending_batch_calls"] = counters_.pending_batch_calls;
  out["pending_batch_rows"] = counters_.pending_batch_rows;
  out["pending_collection_attempts"] = counters_.pending_collection_attempts;
  out["pending_collection_successes"] = counters_.pending_collection_successes;
  out["virtual_collision_count"] = counters_.virtual_collision_count;
  out["max_virtual_visits_remaining"] = counters_.max_virtual_visits_remaining;
  return out;
}

EvalResult Solver::evaluate(const TspState& state, bool need_value) {
  py::tuple result = evaluator_(state.to_python(), need_value).cast<py::tuple>();
  if (result.size() != 3) {
    throw std::runtime_error("C++ MCTS evaluator must return (probs, mask, value)");
  }

  EvalResult eval;
  eval.probs = sequence_to_doubles(result[0]);
  eval.mask = sequence_to_bools(result[1]);
  eval.value = py::cast<double>(result[2]);
  if (static_cast<int>(eval.probs.size()) != state.n || static_cast<int>(eval.mask.size()) != state.n) {
    throw std::runtime_error("C++ MCTS evaluator returned vectors with the wrong graph size");
  }
  counters_.decode += 1;
  if (need_value) {
    counters_.value += 1;
  }
  return eval;
}

std::vector<EvalResult> Solver::evaluate_many(const std::vector<TspState>& states, bool need_value) {
  std::vector<EvalResult> evals;
  if (states.empty()) {
    return evals;
  }

  py::list snapshots;
  for (const TspState& state : states) {
    snapshots.append(state.to_python());
  }

  py::sequence results = evaluator_(snapshots, need_value).cast<py::sequence>();
  if (static_cast<std::size_t>(py::len(results)) != states.size()) {
    throw std::runtime_error("C++ MCTS batched evaluator returned the wrong number of rows");
  }

  evals.reserve(states.size());
  for (std::size_t row = 0; row < states.size(); ++row) {
    py::tuple result = py::cast<py::tuple>(results[row]);
    if (result.size() != 3) {
      throw std::runtime_error("C++ MCTS batched evaluator rows must be (probs, mask, value)");
    }

    EvalResult eval;
    eval.probs = sequence_to_doubles(result[0]);
    eval.mask = sequence_to_bools(result[1]);
    eval.value = py::cast<double>(result[2]);
    if (static_cast<int>(eval.probs.size()) != states[row].n ||
        static_cast<int>(eval.mask.size()) != states[row].n) {
      throw std::runtime_error("C++ MCTS batched evaluator returned vectors with the wrong graph size");
    }
    evals.push_back(std::move(eval));
  }

  counters_.decode += static_cast<long long>(states.size());
  if (need_value) {
    counters_.value += static_cast<long long>(states.size());
  }
  counters_.batch_eval_calls += 1;
  counters_.batch_eval_rows += static_cast<long long>(states.size());
  return evals;
}

void Solver::fill_priors(Node& node, const EvalResult& eval) {
  std::fill(node.prior.begin(), node.prior.end(), 0.0);
  std::fill(node.has_prior.begin(), node.has_prior.end(), 0);

  std::vector<int> legal;
  std::vector<double> raw;
  legal.reserve(static_cast<std::size_t>(node.state.n));
  raw.reserve(static_cast<std::size_t>(node.state.n));

  for (int action = 0; action < node.state.n; ++action) {
    if (eval.mask[static_cast<std::size_t>(action)] == 0) {
      double p = eval.probs[static_cast<std::size_t>(action)];
      if (!std::isfinite(p) || p < 0.0) {
        p = 0.0;
      }
      legal.push_back(action);
      raw.push_back(p);
    }
  }

  if (legal.empty()) {
    throw std::runtime_error("C++ MCTS prior fill found no legal action at a non-terminal node");
  }

  const double total = std::accumulate(raw.begin(), raw.end(), 0.0);
  if (total > 0.0 && std::isfinite(total)) {
    for (std::size_t i = 0; i < legal.size(); ++i) {
      const int action = legal[i];
      node.prior[static_cast<std::size_t>(action)] = raw[i] / total;
      node.has_prior[static_cast<std::size_t>(action)] = 1;
    }
  } else {
    const double uniform = 1.0 / static_cast<double>(legal.size());
    for (int action : legal) {
      node.prior[static_cast<std::size_t>(action)] = uniform;
      node.has_prior[static_cast<std::size_t>(action)] = 1;
    }
  }
}

void Solver::populate_priors(Node& node, double bl_val) {
  if (node.is_terminal()) {
    throw std::runtime_error("populate_priors called on a terminal node");
  }
  const bool need_value = cfg_.leaf_eval == "value_head";
  EvalResult eval = evaluate(node.state, need_value);
  fill_priors(node, eval);

  if (cfg_.leaf_eval == "value_head") {
    node.v_estimate = eval.value;
  } else if (cfg_.leaf_eval == "rollout") {
    node.v_estimate = rollout_remaining_real(node.state) / bl_val;
  } else {
    throw std::runtime_error("unknown leaf_eval: " + cfg_.leaf_eval);
  }
}

double Solver::expand(Node& node, double bl_val) {
  if (node.is_terminal()) {
    throw std::runtime_error("expand called on a terminal node");
  }
  const bool need_value = cfg_.leaf_eval == "value_head";
  EvalResult eval = evaluate(node.state, need_value);
  fill_priors(node, eval);

  if (cfg_.leaf_eval == "value_head") {
    return eval.value;
  }
  if (cfg_.leaf_eval == "rollout") {
    return rollout_remaining_real(node.state) / bl_val;
  }
  throw std::runtime_error("unknown leaf_eval: " + cfg_.leaf_eval);
}

double Solver::rollout_remaining_real(const TspState& state) {
  if (!rollout_evaluator_.is_none()) {
    py::tuple result = rollout_evaluator_(state.to_python()).cast<py::tuple>();
    if (result.size() != 3) {
      throw std::runtime_error(
          "C++ MCTS rollout evaluator must return (remaining_cost, decode_steps, rollout_steps)");
    }
    counters_.decode += py::cast<long long>(result[1]);
    counters_.rollout += py::cast<long long>(result[2]);
    return py::cast<double>(result[0]);
  }

  const double start_length = state.length;
  TspState cur = state;
  while (!cur.all_finished()) {
    EvalResult eval = evaluate(cur, false);
    counters_.rollout += 1;

    int best_action = -1;
    double best_prob = -std::numeric_limits<double>::infinity();
    for (int action = 0; action < cur.n; ++action) {
      if (eval.mask[static_cast<std::size_t>(action)] != 0) {
        continue;
      }
      const double p = eval.probs[static_cast<std::size_t>(action)];
      if (p > best_prob) {
        best_prob = p;
        best_action = action;
      }
    }
    if (best_action < 0) {
      throw std::runtime_error("rollout found no legal action");
    }
    cur.update_in_place(best_action);
  }
  return cur.final_cost() - start_length;
}

std::vector<double> Solver::rollout_many_remaining_real(const std::vector<TspState>& states) {
  std::vector<double> remaining;
  if (states.empty()) {
    return remaining;
  }

  if (!rollout_evaluator_.is_none()) {
    py::list snapshots;
    for (const TspState& state : states) {
      snapshots.append(state.to_python());
    }

    py::tuple result = rollout_evaluator_(snapshots).cast<py::tuple>();
    if (result.size() != 3) {
      throw std::runtime_error(
          "C++ MCTS batched rollout evaluator must return (remaining_costs, decode_steps, rollout_steps)");
    }
    py::sequence costs = py::reinterpret_borrow<py::sequence>(result[0]);
    if (static_cast<std::size_t>(py::len(costs)) != states.size()) {
      throw std::runtime_error("C++ MCTS batched rollout evaluator returned the wrong number of costs");
    }

    remaining.reserve(states.size());
    for (py::handle item : costs) {
      remaining.push_back(py::cast<double>(item));
    }
    counters_.decode += py::cast<long long>(result[1]);
    counters_.rollout += py::cast<long long>(result[2]);
    return remaining;
  }

  remaining.reserve(states.size());
  for (const TspState& state : states) {
    remaining.push_back(rollout_remaining_real(state));
  }
  return remaining;
}

void Solver::simulate(Node& root, double bl_val) {
  std::vector<std::pair<Node*, int>> path;
  Node* node = &root;

  while (node->is_expanded() && !node->is_terminal()) {
    const double fpu = fpu_value_for(*node, bl_val);
    const int action = select_action(*node, fpu);
    path.emplace_back(node, action);

    std::unique_ptr<Node>& child = node->children[static_cast<std::size_t>(action)];
    if (!child) {
      child = std::make_unique<Node>(node->state.updated(action));
    }
    node = child.get();
  }

  double total_norm = 0.0;
  if (node->is_terminal()) {
    total_norm = node->state.final_cost() / bl_val;
    node->v_estimate = 0.0;
  } else {
    const double v_remaining_norm = expand(*node, bl_val);
    node->v_estimate = v_remaining_norm;
    total_norm = node->state.length / bl_val + v_remaining_norm;
  }

  const double value_for_backup = -total_norm;
  for (auto& entry : path) {
    Node* parent = entry.first;
    const int action = entry.second;
    const std::size_t idx = static_cast<std::size_t>(action);
    parent->n_visits[idx] += 1;
    parent->w_total[idx] += value_for_backup;
    parent->q_value[idx] = parent->w_total[idx] / static_cast<double>(parent->n_visits[idx]);
  }
}

void Solver::simulate_batched(Node& root, double bl_val) {
  int completed = 0;
  while (completed < cfg_.n_simulations) {
    const int target = std::min(cfg_.simulation_batch_size, cfg_.n_simulations - completed);
    std::vector<PendingSimulation> pending;
    pending.reserve(static_cast<std::size_t>(target));

    const int max_attempts = std::max(target * 4, target + root.state.n);
    for (int attempt = 0; attempt < max_attempts && static_cast<int>(pending.size()) < target; ++attempt) {
      PendingSimulation sim;
      counters_.pending_collection_attempts += 1;
      if (collect_pending(root, bl_val, sim)) {
        counters_.pending_collection_successes += 1;
        pending.push_back(std::move(sim));
      }
    }

    if (pending.empty()) {
      simulate(root, bl_val);
      completed += 1;
      continue;
    }

    counters_.pending_batch_calls += 1;
    counters_.pending_batch_rows += static_cast<long long>(pending.size());
    evaluate_pending(pending, bl_val);
    completed += static_cast<int>(pending.size());
    counters_.max_virtual_visits_remaining =
        std::max<long long>(counters_.max_virtual_visits_remaining, max_virtual_visits(root));
  }
}

bool Solver::collect_pending(Node& root, double bl_val, PendingSimulation& pending) {
  Node* node = &root;

  while (node->is_expanded() && !node->is_terminal()) {
    const double fpu = fpu_value_for(*node, bl_val);
    const int action = select_action(*node, fpu);
    const std::size_t idx = static_cast<std::size_t>(action);
    pending.path.push_back(PathEntry{node, action});
    node->virtual_n[idx] += 1;

    std::unique_ptr<Node>& child = node->children[idx];
    if (!child) {
      child = std::make_unique<Node>(node->state.updated(action));
    }
    node = child.get();
  }

  if (!node->is_terminal()) {
    if (node->pending_eval) {
      undo_virtual_visits(pending.path);
      pending.path.clear();
      counters_.virtual_collision_count += 1;
      return false;
    }
    node->pending_eval = true;
  }
  pending.leaf = node;
  return true;
}

void Solver::evaluate_pending(std::vector<PendingSimulation>& pending, double bl_val) {
  std::vector<std::size_t> eval_indices;
  std::vector<TspState> eval_states;
  eval_indices.reserve(pending.size());
  eval_states.reserve(pending.size());

  for (std::size_t i = 0; i < pending.size(); ++i) {
    PendingSimulation& sim = pending[i];
    Node& leaf = *sim.leaf;
    if (leaf.is_terminal()) {
      leaf.v_estimate = 0.0;
      sim.total_norm = leaf.state.final_cost() / bl_val;
    } else {
      eval_indices.push_back(i);
      eval_states.push_back(leaf.state);
    }
  }

  if (!eval_states.empty()) {
    const bool need_value = cfg_.leaf_eval == "value_head";
    std::vector<EvalResult> evals = evaluate_many(eval_states, need_value);
    std::vector<double> rollout_values;
    if (cfg_.leaf_eval == "rollout") {
      rollout_values = rollout_many_remaining_real(eval_states);
      if (rollout_values.size() != eval_states.size()) {
        throw std::runtime_error("batched rollout returned a mismatched number of values");
      }
    }

    for (std::size_t row = 0; row < eval_indices.size(); ++row) {
      PendingSimulation& sim = pending[eval_indices[row]];
      Node& leaf = *sim.leaf;
      fill_priors(leaf, evals[row]);

      double v_remaining_norm = 0.0;
      if (cfg_.leaf_eval == "value_head") {
        v_remaining_norm = evals[row].value;
      } else if (cfg_.leaf_eval == "rollout") {
        v_remaining_norm = rollout_values[row] / bl_val;
      } else {
        throw std::runtime_error("unknown leaf_eval: " + cfg_.leaf_eval);
      }
      leaf.v_estimate = v_remaining_norm;
      sim.total_norm = leaf.state.length / bl_val + v_remaining_norm;
    }
  }

  for (PendingSimulation& sim : pending) {
    if (sim.leaf != nullptr) {
      sim.leaf->pending_eval = false;
    }
    backup_pending(sim, -sim.total_norm);
  }
}

void Solver::backup_pending(PendingSimulation& pending, double value_for_backup) {
  undo_virtual_visits(pending.path);
  for (PathEntry& entry : pending.path) {
    Node* parent = entry.parent;
    const int action = entry.action;
    const std::size_t idx = static_cast<std::size_t>(action);
    parent->n_visits[idx] += 1;
    parent->w_total[idx] += value_for_backup;
    parent->q_value[idx] = parent->w_total[idx] / static_cast<double>(parent->n_visits[idx]);
  }
}

void Solver::undo_virtual_visits(const std::vector<PathEntry>& path) {
  for (const PathEntry& entry : path) {
    if (entry.parent == nullptr || entry.action < 0) {
      continue;
    }
    const std::size_t idx = static_cast<std::size_t>(entry.action);
    if (entry.parent->virtual_n[idx] <= 0) {
      throw std::runtime_error("attempted to remove a missing virtual visit");
    }
    entry.parent->virtual_n[idx] -= 1;
  }
}

int Solver::max_virtual_visits(const Node& node) const {
  int max_virtual = 0;
  for (int v : node.virtual_n) {
    max_virtual = std::max(max_virtual, v);
  }
  for (const std::unique_ptr<Node>& child : node.children) {
    if (child) {
      max_virtual = std::max(max_virtual, max_virtual_visits(*child));
    }
  }
  return max_virtual;
}

double Solver::total_effective_visits(const Node& node) const {
  const double pending_weight = cfg_.virtual_loss_weight > 0.0 ? cfg_.virtual_loss_weight : 1.0;
  double total = 0.0;
  for (std::size_t i = 0; i < node.n_visits.size(); ++i) {
    total += static_cast<double>(node.n_visits[i]) +
             static_cast<double>(node.virtual_n[i]) * pending_weight;
  }
  return total;
}

double Solver::fpu_value_for(const Node& node, double bl_val) const {
  if (cfg_.fpu_mode == "fallback") {
    return cfg_.fpu_fallback;
  }
  if (cfg_.fpu_mode == "running_q") {
    const int total_n = node.total_visits();
    if (total_n > 0) {
      return node.sum_w() / static_cast<double>(total_n);
    }
    return cfg_.fpu_fallback;
  }
  if (cfg_.fpu_mode == "node_value") {
    if (std::isfinite(node.v_estimate)) {
      return -(node.state.length / bl_val + node.v_estimate);
    }
    return cfg_.fpu_fallback;
  }
  throw std::runtime_error("unknown fpu_mode: " + cfg_.fpu_mode);
}

int Solver::select_action(const Node& node, double fpu_value) const {
  const double total_n = total_effective_visits(node);
  const double sqrt_total = std::sqrt(std::max(total_n, 1.0));
  const double pending_weight = cfg_.virtual_loss_weight > 0.0 ? cfg_.virtual_loss_weight : 1.0;

  int best_action = -1;
  double best_score = -std::numeric_limits<double>::infinity();
  for (int action = 0; action < node.state.n; ++action) {
    const std::size_t idx = static_cast<std::size_t>(action);
    if (node.has_prior[idx] == 0) {
      continue;
    }
    const int n_sa = node.n_visits[idx];
    const int pending = node.virtual_n[idx];
    const double effective_n_sa = static_cast<double>(n_sa) +
                                  static_cast<double>(pending) * pending_weight;
    double q_sa = n_sa > 0 ? node.q_value[idx] : fpu_value;
    if (pending > 0 && cfg_.virtual_loss_weight > 0.0 && cfg_.virtual_loss_margin > 0.0) {
      const double real_weight = std::max(0.25, static_cast<double>(n_sa));
      const double loss_weight = static_cast<double>(pending) * cfg_.virtual_loss_weight;
      const double loss_q = q_sa - cfg_.virtual_loss_margin;
      q_sa = (q_sa * real_weight + loss_q * loss_weight) / (real_weight + loss_weight);
    }
    const double u_sa =
        cfg_.c_puct * node.prior[idx] * sqrt_total / (1.0 + effective_n_sa);
    const double score = q_sa + u_sa;
    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  if (best_action < 0) {
    throw std::runtime_error("PUCT found no legal action");
  }
  return best_action;
}

int Solver::pick_root_action(const Node& root) {
  if (root.total_visits() == 0) {
    int best_action = -1;
    double best_prior = -std::numeric_limits<double>::infinity();
    for (int action = 0; action < root.state.n; ++action) {
      const std::size_t idx = static_cast<std::size_t>(action);
      if (root.has_prior[idx] == 0) {
        continue;
      }
      if (root.prior[idx] > best_prior) {
        best_prior = root.prior[idx];
        best_action = action;
      }
    }
    if (best_action < 0) {
      throw std::runtime_error("root has no prior legal action");
    }
    return best_action;
  }

  if (cfg_.root_select == "q") {
    int best_action = -1;
    double best_q = -std::numeric_limits<double>::infinity();
    for (int action = 0; action < root.state.n; ++action) {
      const std::size_t idx = static_cast<std::size_t>(action);
      if (root.n_visits[idx] <= 0) {
        continue;
      }
      if (root.q_value[idx] > best_q) {
        best_q = root.q_value[idx];
        best_action = action;
      }
    }
    if (best_action < 0) {
      throw std::runtime_error("root_select='q' found no visited action");
    }
    return best_action;
  }

  if (cfg_.root_select != "visits") {
    throw std::runtime_error("unknown root_select: " + cfg_.root_select);
  }

  std::vector<int> actions;
  std::vector<double> weights;
  int max_count = 0;
  for (int action = 0; action < root.state.n; ++action) {
    const int count = root.n_visits[static_cast<std::size_t>(action)];
    if (count <= 0) {
      continue;
    }
    actions.push_back(action);
    weights.push_back(static_cast<double>(count));
    max_count = std::max(max_count, count);
  }
  if (actions.empty()) {
    throw std::runtime_error("root_select='visits' found no visited action");
  }
  if (cfg_.temperature == 0.0 || max_count == 0) {
    return actions[static_cast<std::size_t>(
        std::distance(weights.begin(), std::max_element(weights.begin(), weights.end())))];
  }

  const double inv_temp = 1.0 / cfg_.temperature;
  for (double& weight : weights) {
    weight = std::pow(weight, inv_temp);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return actions[static_cast<std::size_t>(dist(rng_))];
}

void Solver::apply_dirichlet(Node& root) {
  std::vector<int> actions;
  for (int action = 0; action < root.state.n; ++action) {
    if (root.has_prior[static_cast<std::size_t>(action)] != 0) {
      actions.push_back(action);
    }
  }
  if (actions.empty()) {
    return;
  }

  std::gamma_distribution<double> gamma(cfg_.dirichlet_alpha, 1.0);
  std::vector<double> noise(actions.size(), 0.0);
  double noise_sum = 0.0;
  for (double& eta : noise) {
    eta = gamma(rng_);
    noise_sum += eta;
  }

  if (!(noise_sum > 0.0) || !std::isfinite(noise_sum)) {
    const double uniform = 1.0 / static_cast<double>(actions.size());
    for (int action : actions) {
      root.prior[static_cast<std::size_t>(action)] = uniform;
    }
    return;
  }

  double mixed_sum = 0.0;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const int action = actions[i];
    const std::size_t idx = static_cast<std::size_t>(action);
    const double eta = noise[i] / noise_sum;
    root.prior[idx] = (1.0 - cfg_.dirichlet_epsilon) * root.prior[idx] +
                      cfg_.dirichlet_epsilon * eta;
    mixed_sum += root.prior[idx];
  }

  if (mixed_sum > 0.0 && std::isfinite(mixed_sum)) {
    for (int action : actions) {
      root.prior[static_cast<std::size_t>(action)] /= mixed_sum;
    }
  } else {
    const double uniform = 1.0 / static_cast<double>(actions.size());
    for (int action : actions) {
      root.prior[static_cast<std::size_t>(action)] = uniform;
    }
  }
}

py::dict solve_instance(py::array coords,
                        py::function evaluator,
                        py::dict cfg,
                        double bl_val,
                        py::object rollout_evaluator) {
  py::array_t<double, py::array::c_style | py::array::forcecast> coords64(coords);
  py::buffer_info info = coords64.request();
  if (info.ndim != 2 || info.shape[1] != 2) {
    throw std::runtime_error("coords must have shape (N, 2)");
  }
  const int n = static_cast<int>(info.shape[0]);
  const double* ptr = static_cast<const double*>(info.ptr);
  auto storage = std::make_shared<const std::vector<double>>(
      ptr, ptr + static_cast<std::ptrdiff_t>(2 * n));

  Solver solver(Config::from_python(cfg), std::move(evaluator), std::move(rollout_evaluator));
  return solver.solve_instance(std::move(storage), n, bl_val);
}

}  // namespace am_mcts_cpp
