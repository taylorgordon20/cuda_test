import os
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


# Enable color logging.
colorama.init()

# Create build directory.
log("Creating build into: " + BUILD_DIR)
os.makedirs(BUILD_DIR, exist_ok = True)

# Run cmake from the build directory.
log("Generating build...")
subprocess.run(
    args = [
        "cmake",
        "-DPYTHON_EXECUTABLE=" + sys.executable,
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + OUTPUT_DIR,
        "-Ax64",
        "..",
    ],
    cwd = BUILD_DIR,
    shell = True,
    check = True,
    stdout = sys.stdout,
    stderr = sys.stderr,
)

# Run cmake from the build directory.
log("Executing build...")
subprocess.run(
    ["cmake", "--build", "."],
    cwd = BUILD_DIR,
    shell = True,
    check = True,
    stdout = sys.stdout,
    stderr = sys.stderr,
)
