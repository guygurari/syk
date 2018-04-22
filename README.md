# The SYK Spectrum

The Sachdev-Ye-Kitaev (SYK) model is a quantum-mechanical model of disordered fermions with applications in both condensed matter and high-energy physics. This code computes the spectrum of the SYK Hamiltonian with 4-fermion interactions.

Includes a GPU-based implementation of the Lanczos for computing the edge of the spectrum. This is highly optimized and allows going up to at least 46 Majorana fermions. Also includes CPU-based full diagonalization.

# Available Programs

These are the main programs (see below for build instructions):

* `syk-gpu-lanczos`: A GPU (CUDA) based implementation of Lanczos partial diagonalization for Majorana fermions.
    - Run `syk-gpu-lanczos --help` for a full list of options. 
    - Iteratively computes eigenvalues with error estimates starting with the largest ones in absolute value.
    - Supports multi-GPU (uses all available GPUs by default). This is useful when only a single Hamiltonian fits in host memory, but multiple GPUs are available.
* `kitaev`: A CPU-based full diagonalization of the SYK Hamiltonian. 
    - Run `kitaev --help` to see full list of options.
    - Supports both the Dirac fermion and Majorana fermion vesion.
    - Can compute the full eigenvalue spectrum
    - Can compute thermal fermion 2-point functions (using the eigenvectors)

Other programs and scripts are available for:

* Submitting SLURM jobs (see `scripts/submits-xxx`)
* Analyzing the data (e.g. computing partition functions)

# Building and Unit Testing

Once all dependencies are in installed, simply run `make`.

Dependencies:

* googletest
* CUDA
* Eigen
* Boost
* LAPACK and BLAS
* gfortran

The locations of these libraries will probably have to be adjusted in `Makefile`.

Unit tests, separated by required hardware:

* `test-kitaev`: Test the CPU-based full diagonalization
* `test-gpu`: Test the GPU-based Lanczos diagonalization
* `test-multi-gpu`: Test the multi-GPU version of Lanczos
