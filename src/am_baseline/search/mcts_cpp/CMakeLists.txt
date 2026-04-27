cmake_minimum_required(VERSION 3.18)
project(am_mcts_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_mcts_cpp bindings.cpp mcts.cpp)
target_compile_features(_mcts_cpp PRIVATE cxx_std_17)
