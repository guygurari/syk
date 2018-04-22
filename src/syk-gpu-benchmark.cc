////////////////////////////////////////////////////////////
//
// Benchmarks of GPU code.
// 
////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <boost/program_options.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <cuda_profiler_api.h>

#include "defs.h"
#include "BasisState.h"
#include "DisorderParameter.h"
#include "FactorizedHamiltonian.h"
#include "Timer.h"
#include "CudaUtils.h"
#include "CudaHamiltonian.h"
#include "CudaMultiGpuHamiltonian.h"
#include "CudaMultiGpuHamiltonianNaive.h"
#include "CudaState.h"
#include "CudaLanczos.h"

namespace po = boost::program_options;

typedef unsigned int uint;

typedef struct command_line_options {
    string run_name;
    string data_dir;
    int N;
    int N2;
    double J;
    double mu;
    double available_memory_gb;
    size_t available_memory;
    int steps;
    bool streams;

    int seed;
    bool mock_hamiltonian;
    bool debug;
    bool profile;
    boost::random::mt19937* gen;

    string to_s() {
        stringstream ss;
        ss << "run_name = " << run_name;
        ss << "\ndata_dir = " << data_dir;
        ss << "\nN = " << N;
        ss << "\nN2 = " << N;
        ss << "\nJ = " << J;
        ss << "\nmu = " << mu;
        ss << "\nsteps = " << steps;
        ss << "\navailable memory (GB) = " << available_memory_gb;
        ss << "\nseed = " << seed;

        ss << "\n";
        return ss.str();
    }
} command_line_options;

int parse_command_line_options(int argc, char** argv,
                               command_line_options& opts) {
    opts.N = 8;
    opts.N2 = 8;
    opts.J = 1.;
    opts.mu = 0.;
    opts.available_memory_gb = 5.;
    opts.steps = 20;
    opts.streams = false;

    opts.data_dir = "data/lanczos";
    opts.mock_hamiltonian = false;

    opts.profile = false;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("run-name", 
         po::value<string>(&opts.run_name),
         "set run's name (required for fixing a unique seed for each job)")
	    ("data-dir", 
	     po::value<string>(&opts.data_dir),
         "where to save data")
	    ("N", 
	     po::value<int>(&opts.N),
         "number of fermions")
	    ("N2", 
	     po::value<int>(&opts.N2),
         "go from N to N2 when benchmarking")
        ("J",
         po::value<double>(&opts.J),
         "coupling")
        ("mu",
         po::value<double>(&opts.mu),
         "Hamiltonian shift (default = 0)")
        ("steps",
         po::value<int>(&opts.steps),
         "Number of Lanczos steps")
        ("streams",
         "use multiple streams")
        ("available-memory",
         po::value<double>(&opts.available_memory_gb),
         "Available device memory, in GB")
        ("mock",
         "use mock hamiltonian")
        ("debug",
         "set debugging flag")
        ("profile",
         "profile act_even instead of benchmarking. for use with nvprof")
	    ("seed",
         po::value<int>(&opts.seed),
	     "random seed")
	    ("help", "produce help message");

        po::variables_map vm;        
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        // This should throw an exception if there are missing required
        // arguments
        po::notify(vm);

        if (!vm.count("seed")) {
            opts.seed = get_random_seed(opts.run_name);
        }

        if (opts.N % 2 != 0) {
            cerr << "N must be even (majorana)" << endl;
            exit(1);
        }

        if (vm.count("streams")) {
            opts.streams = true;
        }

        if (vm.count("mock")) {
            opts.mock_hamiltonian = true;
        }

        if (vm.count("debug")) {
            opts.debug = true;
        }

        if (vm.count("profile")) {
            opts.profile = true;
        }

        if (!vm.count("run-name")) {
            opts.run_name = "test";
        }

        if (!vm.count("N2")) {
            opts.N2 = opts.N;
        }

        opts.available_memory =
            (size_t) (opts.available_memory_gb * 1000000000.);
        opts.gen = new boost::random::mt19937(opts.seed);

        return 0;
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

void benchmark_single_lanczos(command_line_options& opts,
                              cublasHandle_t handle,
                              cusparseHandle_t handle_sp,
                              FactorizedSpace space) {
    // Compute the Hamiltonian
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., opts.gen);
    FactorizedParityState initial_state(space, EVEN_CHARGE, opts.gen);

    cout << "Creating Hamiltonian..." << endl;
    Timer timer;
    CudaHamiltonian H(space, Jtensor,
                                opts.available_memory,
                                opts.mock_hamiltonian,
                                opts.debug);
    double H_time_sec = timer.seconds();

    cout << "Running Lanczos with " << opts.steps << " steps" << endl;
    timer.reset();
    CudaLanczos lanczos(opts.steps, initial_state);
    lanczos.compute(handle, handle_sp, H, 1., opts.steps);
    checkCudaErrors(cudaDeviceSynchronize());
    double lanc_time = timer.msecs();

    RealVec alpha;
    RealVec beta;
    lanczos.read_coeffs(alpha, beta);

    cout << space.N
         << "\t" << space.Nd
         << "\t" << space.left.Nd
         << "\t" << round(H_time_sec)
         << "\t" << lanc_time/opts.steps
         << "\t\t" << H.total_h_alloc_size() / 1.e9
         << "\t" << H.total_d_alloc_size() / 1.e9
         << endl;
}

static string lanczos_benchmark_header = "N\tNd\tleft-Nd\tH (sec)\tlanczos (ms)\thost (GB)\tdevice (GB)";

void benchmark_lanczos_varying_left(
    command_line_options& opts,
    cublasHandle_t handle,
    cusparseHandle_t handle_sp,
    int Nd) {

    if (opts.mock_hamiltonian) {
        cout << "(using MOCK hamiltonian)" << endl << endl;
    }

    cout << lanczos_benchmark_header << endl;

    for (int Nd_left = Nd/2; Nd_left <= Nd/2 + 1; Nd_left++) {
        benchmark_single_lanczos(opts, handle, handle_sp,
                                 FactorizedSpace::from_dirac(Nd, Nd_left));
    }

    cout << endl;
}

void benchmark_lanczos(command_line_options& opts,
                       cublasHandle_t handle,
                       cusparseHandle_t handle_sp,
                       int min_N, int max_N) {

    if (opts.mock_hamiltonian) {
        cout << "(using MOCK hamiltonian)" << endl << endl;
    }

    cout << lanczos_benchmark_header << endl;

    for (int N = min_N; N <= max_N; N += 2) {
        benchmark_single_lanczos(opts, handle, handle_sp,
                                 FactorizedSpace::from_majorana(N));
    }
}

void benchmark_act(command_line_options& opts) {
    if (opts.mock_hamiltonian) {
        cout << "(using MOCK hamiltonian)" << endl << endl;
    }

    // Compute the Hamiltonian
    FactorizedSpace space = FactorizedSpace::from_majorana(opts.N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., opts.gen);
    CudaEvenState input(space);
    CudaEvenState output(space);

    cout << "Creating Hamiltonian..." << endl;
    Timer timer;
    CudaHamiltonian H(space, Jtensor,
                                opts.available_memory,
                                opts.mock_hamiltonian);
    // double H_time_sec = timer.seconds();
    timer.print();

    // Warmup
    H.act_even(output, input);

    cout << "Acting " << opts.steps << " times" << endl;
    timer.reset();

    for (int i = 0; i < opts.steps; i++) {
        H.act_even(output, input);
    }

    // timer.print_msec("After act_even: ");
    checkCudaErrors(cudaDeviceSynchronize());
    // timer.print_msec("After final devSync: ");
    double act_time = timer.msecs();

    cout << "N=" << space.N
         << "\ttime=" << act_time/opts.steps
         << "\t\tHsize=" << H.total_h_alloc_size() / 1.e9
         << "\tDsize=" << H.total_d_alloc_size() / 1.e9
         << endl;
}

void benchmark_act_multi_gpu(command_line_options& opts) {
    if (opts.mock_hamiltonian) {
        cout << "(using MOCK hamiltonian)" << endl << endl;
    }

    // Compute the Hamiltonian
    FactorizedSpace space = FactorizedSpace::from_majorana(opts.N);
    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., opts.gen);
    CudaEvenState input(space);
    CudaEvenState output(space);

    cout << "Creating Multi-GPU Hamiltonian..." << endl;
    Timer timer;
    CudaMultiGpuHamiltonian H(
            space, Jtensor,
            opts.available_memory,
            opts.mock_hamiltonian);
    // double H_time_sec = timer.seconds();
    timer.print();

    // Warmup
    H.act_even(output, input);

    cout << "Acting " << opts.steps << " times" << endl;
    timer.reset();

    for (int i = 0; i < opts.steps; i++) {
        H.act_even(output, input);
    }

    // timer.print_msec("After act_even: ");
    checkCudaErrors(cudaDeviceSynchronize());
    // timer.print_msec("After final devSync: ");
    double act_time = timer.msecs();

    cout << "Multi-GPU:\n";
    cout << "N=" << space.N
         << "\ttime=" << act_time/opts.steps
         // << "\t\tHsize=" << H.total_h_alloc_size() / 1.e9
         // << "\tDsize=" << H.total_d_alloc_size() / 1.e9
         << endl;
}

void benchmark_single_act(command_line_options& opts, int N) {
    // Compute the Hamiltonian
    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., opts.gen);

    CudaEvenState state(space, opts.gen);
    CudaEvenState out_state(space);

    // for debugging
    // MajoranaKitaevDisorderParameter Jtensor(space.N);
    // Jtensor.Jelems[0][1][2][4] = 1.;
    // Mat state = Mat::Zero(space.left.D, space.right.D);
    // state(0,0) = 1.;

    //cout << "Creating Hamiltonian..." << endl;
    Timer timer;
    CudaHamiltonian H(space, Jtensor, opts.available_memory);
    double H_time = timer.msecs();

    timer.reset();
    H.act_even(out_state, state);
    checkCudaErrors(cudaDeviceSynchronize());
    double act_time = timer.msecs();
    cout << N
         << "\t" << H_time
         << "\t" << act_time
         << endl;
}

void profile_act(command_line_options& opts) {
    int N = opts.N;
    FactorizedSpace space = FactorizedSpace::from_majorana(N);

    MajoranaKitaevDisorderParameter Jtensor(space.N, 1., opts.gen);

    CudaEvenState state(space, opts.gen);
    CudaEvenState out_state(space);

    cout << "Creating Hamiltonian...\n";
    Timer timer;
    CudaHamiltonian H(space, Jtensor,
                                opts.available_memory);
    timer.print_msec();

    // warm up
    H.act_even(out_state, state);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaProfilerStart();
    // for (int i = 0; i < 10; i++) {
    H.act_even(out_state, state);
    // }
    cudaProfilerStop();
}

void benchmark_hemm(command_line_options& opts,
                    cublasHandle_t handle) {
    int num_rows = 4096;
    int num_cols = 4096;
        
    int alloc_size = sizeof(cucpx) * num_rows * num_cols;
    cucpx* A = d_alloc(alloc_size);
    cucpx* B = d_alloc(alloc_size);
    cucpx* C = d_alloc(alloc_size);

    CudaScalar d_one(1., 0.);
    CudaScalar d_zero(0., 0.);

    cout << "hemm:" << endl;
    Timer timer;
    checkCublasErrors(cublasZhemm(
                          handle,
                          CUBLAS_SIDE_LEFT,
                          CUBLAS_FILL_MODE_LOWER,
                          num_rows,
                          num_cols,
                          d_one.ptr,
                          A,
                          num_rows,
                          B,
                          num_rows,
                          d_zero.ptr,
                          C,
                          num_rows));
    checkCudaErrors(cudaDeviceSynchronize());
    timer.print_msec();

    cout << "gemm:" << endl;
    timer.reset();
    checkCublasErrors(cublasZgemm(
                          handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          num_rows, num_rows, num_rows,
                          d_one.ptr,
                          A, num_rows,
                          B, num_rows,
                          d_zero.ptr,
                          C, num_rows));
    checkCudaErrors(cudaDeviceSynchronize());
    timer.print_msec();
                          
    d_free(A);
    d_free(B);
    d_free(C);
}    

void benchmark_memcpy(command_line_options& opts) {
    Timer timer;

    size_t alloc_size = 10000000000;

    cout << "Allocating device mem" << endl;
    timer.reset();
    cucpx* d_ptr = d_alloc(alloc_size);
    timer.print_msec();

    cout << "Allocating host pageable memory" << endl;
    timer.reset();
    cucpx* h_ptr = (cucpx*) malloc(alloc_size);
    timer.print_msec();

    cout << "Memcpy pageable memory" << endl;
    timer.reset();
    checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, alloc_size,
                               cudaMemcpyHostToDevice));
    timer.print_msec();

    free(h_ptr);

    cout << "Allocating host pinned memory" << endl;
    cucpx* h_pinned;
    timer.reset();
    checkCudaErrors(cudaMallocHost((void**) &h_pinned, alloc_size));
    timer.print_msec();

    cout << "Memcpy pinned memory" << endl;
    timer.reset();
    checkCudaErrors(cudaMemcpy(d_ptr, h_pinned, alloc_size,
                               cudaMemcpyHostToDevice));
    timer.print_msec();

    cout << "Freeing device mem" << endl;
    timer.reset();
    d_free(d_ptr);
    timer.print_msec();

    checkCudaErrors(cudaFreeHost(h_pinned));
}

void benchmark_gemm_and_memcpy(command_line_options& opts,
                               cublasHandle_t handle,
                               cusparseHandle_t handle_sp) {
    // big matrices
    int n = 4000; // 8000;
    cucpx* d_A = d_alloc(sizeof(cucpx) * n * n);
    cucpx* d_B = d_alloc(sizeof(cucpx) * n * n);
    cucpx* d_C = d_alloc(sizeof(cucpx) * n * n);

    CudaScalar d_one(1., 0.);

    // big vectors
    size_t alloc_size = 12 * sizeof(cucpx) * n * n;
    cpx* h_mem;
    checkCudaErrors(cudaMallocHost((void**) &h_mem, alloc_size));
    cucpx* d_mem = d_alloc(alloc_size);

    Timer timer;
    cudaProfilerStart();

    if (opts.streams) {
        cout << "using streams\n";
        cudaStream_t memcpy_stream = d_create_stream();
        cudaStream_t gemm_stream = d_create_stream();
    
        timer.reset();

        checkCudaErrors(cudaMemcpyAsync(d_mem, h_mem, alloc_size,
                                        cudaMemcpyHostToDevice, memcpy_stream));

        checkCublasErrors(cublasSetStream(handle, gemm_stream));
        checkCublasErrors(cublasZgemm(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      n, n, n,
                                      d_one.ptr, // alpha
                                      d_A, n,
                                      d_B, n,
                                      d_one.ptr, // beta
                                      d_C, n));

        checkCudaErrors(cudaDeviceSynchronize());
        timer.print_msec();

        d_destroy_stream(memcpy_stream);
        d_destroy_stream(gemm_stream);
    }
    else {
        cout << "not using streams\n";
        timer.reset();

        checkCudaErrors(cudaMemcpy(d_mem, h_mem, alloc_size,
                                   cudaMemcpyHostToDevice));

        checkCublasErrors(cublasZgemm(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      n, n, n,
                                      d_one.ptr, // alpha
                                      d_A, n,
                                      d_B, n,
                                      d_one.ptr, // beta
                                      d_C, n));

        checkCudaErrors(cudaDeviceSynchronize());
        timer.print_msec();
    }

    cudaProfilerStop();

    d_free(d_A);
    d_free(d_B);
    d_free(d_C);

    checkCudaErrors(cudaFreeHost(h_mem));
    d_free(d_mem);
}

void benchmark_multiply_right_dense(command_line_options& opts,
                                    cublasHandle_t handle,
                                    cusparseHandle_t handle_sp) {
    // The following is the left-2, right-2 case of
    // CudaHamiltonian::multiply_right_dense()
    FactorizedSpace space = FactorizedSpace::from_majorana(opts.N);
    // int num_blocks = binomial(space.N, 2);
    // int num_blocks = 1;
    int num_blocks = 60;

    int A_rows = space.left.D / 2;
    int A_cols = space.right.D / 2;
    int leading_dim_A = space.left_block_rows() * num_blocks;
    // This shorter dim doesn't seem to improve performance
    // int leading_dim_A = A_cols; 
    // stride just on the rows -- the A blocks are stores in a column)
    int A_stride = A_rows;

    int B_rows = A_cols;
    int B_cols = space.right.D / 2;
    int B_stride = B_rows * B_cols;

    int C_rows = A_rows;
    int C_cols = B_cols;
    int C_stride = C_rows * C_cols;

    cucpx* A0 = d_alloc(sizeof(cucpx) * A_rows * A_cols * num_blocks);
    cucpx* B0 = d_alloc(sizeof(cucpx) * B_rows * B_cols * num_blocks);
    cucpx* C0 = d_alloc(sizeof(cucpx) * C_rows * C_cols * num_blocks);

    // total number of instructions it's a multiply-add, so factor 2
    // complex gives factor of 4
    double expected_Tflop =
        4. * 2. * A_rows * A_cols * B_cols * num_blocks / 1e12;
    // cout << "expected_flop = " << (size_t) (expected_Tflop * 1e12) << endl;
    cout << "expected_Tflop = " << expected_Tflop << endl;

    vector<cudaStream_t> streams;
    
    while (streams.size() < num_blocks) {
        streams.push_back(d_create_stream());
    }

    CudaScalar d_zero(0., 0.);
    CudaScalar d_one(1., 0.);

    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;

    // Warmup
    checkCublasErrors(cublasZgemm( // D (double), Z (complex)
                          handle, opA, opB,
                          A_rows, B_cols, A_cols, 
                          /* (double*) */ d_one.ptr,
                          /* (double*) */ A0, leading_dim_A,
                          /* (double*) */ B0, B_rows, 
                          /* (double*) */ d_zero.ptr,
                          /* (double*) */ C0, C_rows));  

    // checkCublasErrors(cublasZgemm3m( // D (double), Z (complex)
    //                       handle, opA, opB,
    //                       A_rows, B_cols, A_cols, 
    //                       d_one.ptr,
    //                       A0, leading_dim_A,
    //                       B0, B_rows, 
    //                       d_zero.ptr,
    //                       C0, C_rows));  

    checkCudaErrors(cudaDeviceSynchronize());

    Timer timer;

    // Profile
    cudaProfilerStart();

    for (int i = 0; i < num_blocks; i++) {
        // Each block multiplication runs in its own stream
        checkCublasErrors(cublasSetStream(handle, streams.at(i)));

        cucpx* A = A0 + i * A_stride;
        cucpx* B = B0 + i * B_stride;
        cucpx* C = C0 + i * C_stride;

        checkCublasErrors(cublasZgemm( // D (double), Z (complex)
                              handle, opA, opB,
                              A_rows, B_cols, A_cols, 
                              /* (double*) */ d_one.ptr,
                              /* (double*) */ A, leading_dim_A,
                              /* (double*) */ B, B_rows, 
                              /* (double*) */ d_zero.ptr,
                              /* (double*) */ C, C_rows));  

        // Unsupported on K80
        // checkCublasErrors(cublasZgemm3m( // Z (complex)
        //                       handle, opA, opB,
        //                       A_rows, B_cols, A_cols, 
        //                       d_one.ptr,
        //                       A0, leading_dim_A,
        //                       B0, B_rows, 
        //                       d_zero.ptr,
        //                       C0, C_rows));  
    }

    // Wait for all streams to finish (don't do a full device sync),
    // because the memory-copy stream is running independently of this.
    for (int i = 0; i < num_blocks; i++) {
        checkCudaErrors(cudaStreamSynchronize(streams.at(i)));
    }

    double msecs = timer.msecs();
    double Tflops = expected_Tflop / timer.seconds();

    cudaProfilerStop();

    cout << "Time = " << msecs << " msecs" << endl;
    cout << "Teraflops = " << Tflops << endl;

    d_free(A0);
    d_free(B0);
    d_free(C0);
}

int main(int argc, char *argv[]) {
    command_line_options opts;

    if (parse_command_line_options(argc, argv, opts)) {
        return 1;
    }

    cuda_print_device_properties();
    cublasHandle_t handle = cublas_init(argc, argv);
    cusparseHandle_t handle_sp = cusparse_init();

    if (opts.profile) {
        // profile_act(opts);
        // benchmark_gemm_and_memcpy(opts, handle, handle_sp);
        // benchmark_multiply_right_dense(opts, handle, handle_sp);
    }
    else {
        // benchmark_multiply_right_dense(opts, handle, handle_sp);
        // benchmark_lanczos(opts, handle, handle_sp, opts.N, opts.N2);
        // benchmark_act(opts);
        benchmark_act_multi_gpu(opts);
        // benchmark_lanczos(opts, handle, handle_sp, 30, 40);
        // benchmark_lanczos_varying_left(opts, handle, handle_sp, 20);
        // benchmark_memcpy(opts);

        // cout << "--- benchmark_gemm_and_memcpy ---" << endl;
        // for (int i = 0; i < 2; i++) {
        //     benchmark_gemm_and_memcpy(opts, handle, handle_sp);
        // }
    }

    // benchmark_hemm(opts, handle);

    cublas_destroy(handle);
    cusparse_destroy(handle_sp);

    return 0;
}
