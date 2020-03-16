import os
import platform
import subprocess
import sys

import colorama
import termcolor

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
OUTPUT_DIR = os.path.join(BUILD_DIR, "out")


def log(msg, **kwargs):
  termcolor.cprint(msg, "green", **kwargs)


def init():
  # Enable color logging.
  colorama.init()

  # Create build directory.
  log("Creating build into: " + BUILD_DIR)
  os.makedirs(BUILD_DIR, exist_ok = True)


def generate():
  log("Generating build...")
  subprocess.run(
      args = [
          "cmake",
          "-DPYTHON_EXECUTABLE=" + sys.executable,
          "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + OUTPUT_DIR,
          "-Ax64" if platform.system() == "Windows" else "",
          "..",
      ],
      cwd = BUILD_DIR,
      check = True,
      stdout = sys.stdout,
      stderr = sys.stderr,
  )


def build(config):
  log(f"Executing {config} build...")
  subprocess.run(
      ["cmake", "--build", ".", "--config", config],
      cwd = BUILD_DIR,
      check = True,
      stdout = sys.stdout,
      stderr = sys.stderr,
  )


def test(config):
  log(f"Executing {config} tests...")
  subprocess.run(
      ["ctest", "-C", config],
      cwd = BUILD_DIR,
      check = True,
      stdout = sys.stdout,
      stderr = sys.stderr,
  )


if __name__ == "__main__":
  init()
  generate()
  build("Debug")
  build("Release")
  test("Debug")
  test("Release")