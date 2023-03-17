# CUDAsap: Statically-Determined Execution Statistics as Alternative to Execution-Based Profiling

[![DOI](https://zenodo.org/badge/597108344.svg)](https://zenodo.org/badge/latestdoi/597108344)

The goal of this project is to compute the execution frequencies of basic blocks of CUDA kernels, derived from LLVM IR before kernel runtime.
The code has been used to evaluate and compare the static method to the execution-based profiling tool [CUDA Flux](https://github.com/UniHD-CEG/cuda-flux).
More details on the underlying methods and results will be provided in corresponding publication **CUDAsap: Statically-Determined Execution Statistics as Alternative to Execution-Based Profiling** by Emonds, Braun and Fröning (to be published at CCGRID 2023).

## Prerequisites

* LLVM 10.0
* CUDA 11.0
* Python 3 with Sympy 1.4 and NumPy >= 1.13
* re2c lexer generator (http://re2c.org/)

## Compilation
The host and the device pass can be compiled into one shared library using the CMake file given in `passes/combinedHDPass`.
Note that it expects the directory `mekong-utils` in `passes/` and the build directory to be `passes/combinedHDPass/build`.
Combined CMake configuration and building of the pass took about 30 seconds on an [Intel Xeon CPU E5-2609](https://ark.intel.com/content/www/de/de/ark/products/64588/intel-xeon-processor-e52609-10m-cache-2-40-ghz-6-40-gts-intel-qpi.html) (running in parallel on four cores).  
The `conditionAnalyzer` consists of multiple Python scripts and does not require any prior set-up.

## Application
To apply the passes compile the CUDA code in consideration as follows:
```
$ clang++ -Xclang -load -Xclang /path/to/libCombinedHostDevicePass.so -mllvm -inline-threshold=1000000 -finline-functions --cuda-gpu-arch=<arch> -O3 -std=c++11 <code file> -o <exec> -lcudart -ldl -lrt -pthread
```
The first part loads the shared library and commands the compiler to inline functions.
This is necessary for the code analysis as the device code analyzer does not follow function calls in the kernel definition.
The part at the end of the command is needed for proper loading of libraries.

The resulting executable is instrumented to call the `conditionAnalyzer` scripts before the kernel launch.
There exist two switches for the scripts implemented as environment variables:
1) `SOLVE_NUMERIC`: switch to solve the system of linear equations numerically (value 1, default) or symbolically (value 0). The further is usually faster, whereas in the latter case, the results may be re-used for different parameters (not implemented yet).
2) `SEQUENTIAL`: if 1 (default), the conditions are evaluated sequentially. If 0, the evaluation might be parallelized based on the number of threads available.

## Outputs
The pass as well as the `conditionAnalyzer` produce several output files:
* For each CUDA kernel two CFG files are generated: one containing the plain CFG and the branch conditions, and a second one containing the CFG with dependent conditions. The latter is used for the computation of the basic block execution frequencies.
* One file that contains the kernel parameters extracted by the host runtime (`host_param.out`).
* One file that contains the names of the basic block (`block_id.out`).
* One file that contains the basic block execution frequencies (`block_freq.txt`).

### `host_param.out`
This file contains the necessary parameters to analyze the CFG and the branch conditions within it:
* kernel name
* grid dimension XY (combined as unsigned 64-bit integer)
* grid dimension Z
* CTA dimension XY (combined as unsigned 64-bit integer)
* CTA dimension Z
* parameters that appear in the branch conditions 

### `block_freq.txt`
This file contains the following information (in CSV format):
* kernel name
* analysis switch (see comment below)
* grid dimensions (six entries)
* kernel parameters with names (arbitrary number of entries)
* execution times (three entries): total, evaluation of conditions, solving the system of linear equations
* basic block execution frequencies (arbitrary entries)

The order of the basic block execution frequencies coincides with the definition in the LLVM IR code (and hence the `block_id.out` file).

The above mentioned analysis switch allows the user to compute the basic block execution frequencies either by solving the system of linear equations given by the adjacency matrix of the CFG (switch value 1, default), or by following all the possible paths through the CFG and sum up the number of involved threads.
The latter is not used in the paper and not actively supported at the moment.
Your mileage may vary. 

## Known Limitations
Naturally, CUDA kernels with dynamic control flow are not supported by CUDAsap.
By "dynamic control flow" it is meant that at least one of the branch conditions in the CFG depends on data passed to the kernel.
When detected by the device analysis pass, they are skipped and the next kernel is examined.

In addition, path-dependent branch conditions as well as very large grid configurations are currently not supported.

Due to the used version of the clang compiler texture memory is not supported either.
