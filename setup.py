import glob
import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
  """Provides a CPython extension build via CMake."""
  def __init__(self, name, root_dir = ""):
    Extension.__init__(self, name, sources = [])
    self.root_dir = os.path.abspath(root_dir)


class CMakeBuild(build_ext):
  """Build command used to build CMake extensions."""
  def run(self):
    # Fail if the required cmake version is not available.
    subprocess.check_output(["cmake", "--version"])

    # Build each extension.
    for ext in self.extensions:
      self.build_extension(ext)

  def build_extension(self, ext):
    # Load default environment variables and set CXXFLAGS to match Python install.
    env = os.environ.copy()
    env["CXXFLAGS"] = (
        env.get("CXXFLAGS", "") +
        f" -DVERSION_INFO=\\\"{self.distribution.get_version()}\\\""
    )

    # Set the CMake output directory to the usual Python extension output directory.
    output_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath(ext.name))
    )

    # Set the build configuration type.
    config = "Debug" if self.debug else "Release"

    # Run CMake generation.
    print("Generating cmake build...")
    os.makedirs(self.build_temp, exist_ok = True)
    subprocess.check_call(
        args = [
            "cmake",
            ext.root_dir,
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output_dir}",
            "-Ax64",
        ],
        cwd = self.build_temp,
        env = env,
    )

    # Run CMake build.
    print("Executing cmake build...")
    subprocess.check_call(
        args = ["cmake", "--build", ".", "--config", config],
        cwd = self.build_temp,
    )


# List of all public header files.
HEADERS = glob.glob("include/**/*.hpp")

setup(
    name = "happy",
    version = "0.0.1",
    author = "Taylor Gordon",
    description = "A test library to build C++/Cuda Python extensions",
    long_description = "",
    headers = HEADERS,
    ext_modules = [CMakeExtension("happy")],
    cmdclass = {"build_ext": CMakeBuild},
    zip_safe = False
)