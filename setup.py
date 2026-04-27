from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import pybind11


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],
        "unix": ["-O3", "-std=c++17"],
    }

    def build_extensions(self):
        opts = self.c_opts.get(self.compiler.compiler_type, [])
        for ext in self.extensions:
            ext.extra_compile_args = list(opts)
        super().build_extensions()


ext_modules = [
    Extension(
        "am_baseline.search.mcts_cpp._mcts_cpp",
        [
            "src/am_baseline/search/mcts_cpp/bindings.cpp",
            "src/am_baseline/search/mcts_cpp/mcts.cpp",
        ],
        include_dirs=[pybind11.get_include()],
        language="c++",
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
