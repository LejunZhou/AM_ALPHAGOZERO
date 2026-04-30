#pragma once

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace am_mcts_cpp {

namespace py = pybind11;

struct Config {
  int n_simulations = 200;
  int simulation_batch_size = 1;
  double virtual_loss_weight = 3.0;
  double virtual_loss_margin = 0.5;
  double c_puct = 0.05;
  double temperature = 0.0;
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.0;
  std::string leaf_eval = "rollout";
  std::string value_norm = "bl";
  std::string fpu_mode = "running_q";
  double fpu_fallback = -1.0;
  std::string root_select = "visits";
  bool tree_reuse = true;
  // Stage 4 Phase A: when true, Solver::solve_instance / BatchSearch dump
  // per-tour-step root visit counts (raw root.n_visits) into the result dict.
  // Defaults false to preserve Stage 2/3 wire format.
  bool return_root_visits = false;
  std::uint64_t seed = 0;

  static Config from_python(py::dict cfg);
};

struct Counters {
  long long decode = 0;
  long long value = 0;
  long long rollout = 0;
  long long batch_eval_calls = 0;
  long long batch_eval_rows = 0;
  long long pending_batch_calls = 0;
  long long pending_batch_rows = 0;
  long long pending_collection_attempts = 0;
  long long pending_collection_successes = 0;
  long long virtual_collision_count = 0;
  long long max_virtual_visits_remaining = 0;
};

struct TspState {
  int n = 0;
  std::shared_ptr<const std::vector<double>> coords;
  std::vector<unsigned char> visited;
  std::vector<int> tour;
  int first = -1;
  int prev = 0;
  int step = 0;
  double length = 0.0;

  static TspState initial(std::shared_ptr<const std::vector<double>> coords, int n);

  bool all_finished() const;
  double distance(int a, int b) const;
  double final_cost() const;
  TspState updated(int action) const;
  void update_in_place(int action);
  py::dict to_python() const;
};

struct EvalResult {
  std::vector<double> probs;
  std::vector<unsigned char> mask;
  double value = 0.0;
};

struct Node {
  explicit Node(const TspState& state);

  bool is_terminal() const;
  bool is_expanded() const;
  int total_visits() const;
  int total_effective_visits() const;
  double sum_w() const;

  TspState state;
  std::vector<int> n_visits;
  std::vector<int> virtual_n;
  std::vector<double> w_total;
  std::vector<double> q_value;
  std::vector<double> prior;
  std::vector<unsigned char> has_prior;
  std::vector<std::unique_ptr<Node>> children;
  double v_estimate;
  bool pending_eval;
};

struct PathEntry {
  Node* parent = nullptr;
  int action = -1;
};

struct PendingSimulation {
  std::vector<PathEntry> path;
  Node* leaf = nullptr;
  double total_norm = 0.0;
};

class Solver {
 public:
  Solver(Config cfg, py::function evaluator, py::object rollout_evaluator);

  py::dict solve_instance(std::shared_ptr<const std::vector<double>> coords,
                          int n,
                          double bl_val);

 private:
  EvalResult evaluate(const TspState& state, bool need_value);
  std::vector<EvalResult> evaluate_many(const std::vector<TspState>& states, bool need_value);
  void fill_priors(Node& node, const EvalResult& eval);
  void populate_priors(Node& node, double bl_val);
  double expand(Node& node, double bl_val);
  double rollout_remaining_real(const TspState& state);
  std::vector<double> rollout_many_remaining_real(const std::vector<TspState>& states);
  void simulate(Node& root, double bl_val);
  void simulate_batched(Node& root, double bl_val);
  bool collect_pending(Node& root, double bl_val, PendingSimulation& pending);
  void evaluate_pending(std::vector<PendingSimulation>& pending, double bl_val);
  void backup_pending(PendingSimulation& pending, double value_for_backup);
  void undo_virtual_visits(const std::vector<PathEntry>& path);
  int max_virtual_visits(const Node& node) const;
  double total_effective_visits(const Node& node) const;
  double fpu_value_for(const Node& node, double bl_val) const;
  int select_action(const Node& node, double fpu_value) const;
  int pick_root_action(const Node& root);
  void apply_dirichlet(Node& root);

  Config cfg_;
  py::function evaluator_;
  py::object rollout_evaluator_;
  Counters counters_;
  std::mt19937_64 rng_;
};

py::dict solve_instance(py::array coords,
                        py::function evaluator,
                        py::dict cfg,
                        double bl_val,
                        py::object rollout_evaluator = py::none());

class BatchSearch {
 public:
  BatchSearch(py::array coords, py::dict cfg, py::object bl_vals);
  ~BatchSearch();

  py::list collect_requests();
  void apply_results(py::sequence results);
  bool is_done() const;
  py::dict results() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace am_mcts_cpp
