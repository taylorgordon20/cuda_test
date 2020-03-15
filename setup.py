from skbuild import setup

setup(
    name = "happy",
    version = "0.0.1",
    description = "A minimal C++/CUDA extension for CPython",
    author = "Taylor Gordon",
    license = "MIT",
)
"""
setup(
    name = "happy",
    version = "0.0.1",
    license = "MIT",
    ext_modules = [
        Extension(
            name = "happy",
            sources = [
                "src/happy.cu",
                "src/happy.cpp",
            ],
            include_dirs = [
                pybind11.get_include(),
            ],
            language = "c++",
        )
    ],
    install_requires = [
        "pybind11",
    ],
    setup_requires = [
        "pybind11",
    ],
)
"""