import os
from setuptools import setup, find_packages
from pathlib import Path
import subprocess
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

# In case user doesn't use `--recurse-submodules` when cloning this repo
cutlass_dir = os.path.join("csrc", "cutlass") 
if os.path.exists(cutlass_dir) and not os.listdir(cutlass_dir): 
    subprocess.run(["rm", "-rf", cutlass_dir], check=True)
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"]) 

ext_modules = [
    CUDAExtension(
        name="flash_muon_cuda",
        sources=[
            "csrc/flash_api.cpp",
            "csrc/matmul_transpose.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"],
            "nvcc": [
                "-O3",
                "-arch=sm_80",
                "-std=c++17",
                "-DNDEBUG",
                "--expt-relaxed-constexpr",
                "-Wno-deprecated-declarations"
            ],
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
        ],
    ),
]

setup(
    name="flash_muon",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=1.10.0",
    ],
)