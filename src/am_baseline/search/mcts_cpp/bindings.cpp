#include "mcts.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_mcts_cpp, m) {
  m.doc() = "Single-threaded pybind11 MCTS backend for AM AlphaGoZero TSP search.";
  m.def(
      "solve_instance",
      &am_mcts_cpp::solve_instance,
      py::arg("coords"),
      py::arg("evaluator"),
      py::arg("cfg"),
      py::arg("bl_val"),
      py::arg("rollout_evaluator") = py::none(),
      "Solve one TSP instance with the C++ MCTS tree loop.");

  py::class_<am_mcts_cpp::BatchSearch>(m, "BatchSearch")
      .def(py::init<py::array, py::dict, py::object>(),
           py::arg("coords"),
           py::arg("cfg"),
           py::arg("bl_vals"))
      .def("collect_requests", &am_mcts_cpp::BatchSearch::collect_requests)
      .def("apply_results", &am_mcts_cpp::BatchSearch::apply_results)
      .def("is_done", &am_mcts_cpp::BatchSearch::is_done)
      .def("results", &am_mcts_cpp::BatchSearch::results);
}
