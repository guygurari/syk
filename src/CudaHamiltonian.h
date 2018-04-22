#ifndef __CUDA_HAMILTONIAN_H__
#define __CUDA_HAMILTONIAN_H__

#include <boost/shared_ptr.hpp>

#include "defs.h"
#include "DisorderParameter.h"
#include "FactorizedHamiltonian.h"
#include "CudaUtils.h"
#include "CudaState.h"

/* #define UNLIMITED_MEMORY 0 */

using boost::shared_ptr;

typedef enum {
    TOP_BLOCK, BOTTOM_BLOCK
} MatBlock;

typedef enum {
    LEFT, RIGHT
} Side;

class HamiltonianData;
struct SomeHamiltonianTermBlocks;
typedef boost::shared_ptr<HamiltonianData> HamiltonianData_ptr;

class CudaHamiltonianInterface {
public:
    CudaHamiltonianInterface();
    virtual ~CudaHamiltonianInterface();

    // Act with the Hamiltonian on a state with even charge,
    // writing the result to output. The result is written to device
    // memory, and is not copied to the host by default.
    virtual void act_even(CudaEvenState& output,
                          CudaEvenState& state) = 0;

    // Total memory allocated on host and device
    virtual size_t total_h_alloc_size() = 0;
    virtual size_t total_d_alloc_size() = 0;
    virtual size_t predicted_total_d_alloc_size() = 0;

    virtual void print_memory_allocation() = 0;
};

class ActionContext {
public:
    ActionContext(CudaResourceManager& manager);
    ActionContext(const ActionContext& other);
    ActionContext& operator=(const ActionContext& other);
    
    cudaStream_t memcpy_stream;
    cudaStream_t kernel_stream;
    int resident_chunk;
};

//
// Factorized Hamiltonian implementation for GPUs.
// 
class CudaHamiltonian
    : public HamiltonianTermProcessor,
      public CudaHamiltonianInterface {
public:
    // Initialize the Hamiltonian and the GPU data structures
    // available_device_memory is in bytes. 0 means unlimited.
    CudaHamiltonian(
        FactorizedSpace& _space,
        MajoranaKitaevDisorderParameter& Jtensor,
        size_t available_device_memory,
        bool mock_hamiltonian = false,
        bool debug = false);

    // Constructs the Hamiltonian using pre-prepared chunk allocation.
    // Following construction the chunks are not populated. This should
    // be done by calling process() for each operator. Once this is done,
    // call finalize_construction_after_processing().
    //
    // device_id is the CUDA device to use for allocating memory.
    CudaHamiltonian(
        FactorizedSpace& _space,
        vector<HamiltonianData_ptr> op_chunks,
        int device_id,
        bool debug);

    // Constructs the Hamiltonian without any chunk allocation. Operators
    // should first be allocated into chunks by calling record_op_pair().
    // The actual operators should then be added by calling process().
    // Finally, finalize_construction_after_processing() should be called.
    //
    // device_id is the CUDA device to use for allocating memory.
    CudaHamiltonian(
        FactorizedSpace& _space,
        int device_id,
        bool debug);

    void finalize_construction_after_processing(size_t available_memory);

    virtual ~CudaHamiltonian();

    // Free the GPU data etc.
    void destroy();

    virtual size_t total_h_alloc_size();
    virtual size_t total_d_alloc_size();
    virtual size_t predicted_total_d_alloc_size();
    virtual void print_memory_allocation();

    // Act with the Hamiltonian on a state with even charge,
    // writing the result to output. The result is written to device
    // memory, and is not copied to the host by default.
    virtual void act_even(CudaEvenState& output,
                          CudaEvenState& state);

    ////// Use the following methods for async action //////
    ActionContext prepare_to_act(CudaEvenState& output);

    // Starts an async chunk of act_even. Returns true if processing a chunk, and
    // false if done. If true is returned, should call sync_act_even() next to wait
    // for the chunk to complete. A new context should be created for each new run
    // through.
    bool act_even_chunk_async(CudaEvenState& output,
                              CudaEvenState& state,
                              ActionContext& context);

    // Synchronize the actions launched by act_even_chunk_async()
    void sync_act_even(ActionContext& context);

    // Async version of act_even. Launches the calculation, have the event
    // record when calculation is complete, and returns.
    void act_even_single_chunk_async(
        CudaEvenState& output,
        CudaEvenState& state,
        cudaEvent_t completion);

    FactorizedSpace space;

    void record_op_pair(int left_idx, size_t memory_per_chunk);

    // Process Hamiltonian terms on the host during construction
    virtual void process(int left_idx, FactorizedOperatorPair& ops_pair);

    // Whether this Hamiltonian still has some operators to process
    bool done_processing_ops();

    // True if everything is ready for calling act_even()
    bool ready_to_act();

    void print_chunks();

    // Given some amount of available memory, estimate the net available amount
    // when accounting for things like memory alignment.
    static size_t net_available_memory(size_t available_memory);

    // Holds the operator data in host and device memory. Each element holds
    // part of the data that fits in device memory.
    vector<HamiltonianData_ptr> op_chunks;

private:

    ////////// Used in process() callback ///////////////////

    // How many operators were seen so far per left_idx, regardless of chunk
    map<int, int> seen_ops;

    // Accumulates the number of operators inserted into previous chunks
    map<int, int> process_prev_num_ops;

    // Tally operators that have been processed so far
    map<int, int> num_processed_ops;

    // The chunk we're currently processing
    int process_chunk_idx;

    ///////////////////////////////////////////////////////

    bool host_mem_allocated;

    int device_id;
    CudaScalar* d_zero;
    CudaScalar* d_one;
    cucpx* d_ones_vector;

    CudaAllocator* allocators[2];
    int current_allocator;

    CudaResourceManager cuda_manager;

    void basic_init(FactorizedSpace& _space, int _device_id, bool _debug);

    void set_device();
    void prepare_op_chunks(size_t available_memory);
    void combine_two_chunks();
    
    void alloc_host_memory();
    void alloc_device_constants();

    // If completion is 0, blocks until the calculation is done.
    // Otherwise, runs asyncly and records the event when calculation is done.
    // Many streams are used but when final_stream is done, the calculation is
    // done.
    void launch_chunk_kernels(
        CudaEvenState& output,
        CudaEvenState& state,
        HamiltonianData& chunk,
        cudaStream_t final_stream,
        cudaEvent_t* completion = 0);

    cudaStream_t act_on_blocks(
        CudaHandles& handles,
        Side sparse_side,
        SomeHamiltonianTermBlocks& left_blocks,
        cucpx* d_state_block,
        SomeHamiltonianTermBlocks& right_blocks);

    // Sparse matrix description
    cusparseMatDescr_t sp_description;

    void setup_cuda_structures();

    void set_sum_to_zero(SomeHamiltonianTermBlocks& blocks,
                         cudaStream_t stream);

    // multiply left-sparse * state * right-dense, then sum over blocks
    void mult_l_sparse_r_dense_and_sum(
        cublasHandle_t handle, cusparseHandle_t handle_sp,
        cudaStream_t stream,
        SomeHamiltonianTermBlocks& left_blocks,
        cucpx* d_state_block,
        SomeHamiltonianTermBlocks& right_blocks);

    // multiply left-dense * state * right-sparse, then sum over blocks
    void mult_r_sparse_l_dense_and_sum(
        cublasHandle_t handle, cusparseHandle_t handle_sp,
        cudaStream_t stream,
        SomeHamiltonianTermBlocks& left_blocks,
        cucpx* d_state_block,
        SomeHamiltonianTermBlocks& right_blocks);

    // Compute (O_L * state) where O_L is sparse
    void multiply_left_sparse(
        cublasHandle_t handle, cusparseHandle_t handle_sp,
        SomeHamiltonianTermBlocks& blocks,
        cucpx* d_state_block);

    // Compute ((O_L * state) * O_R) where O_R is dense
    void multiply_right_dense(
        cublasHandle_t handle,
        cudaStream_t parent_stream,
        SomeHamiltonianTermBlocks& left_blocks,
        SomeHamiltonianTermBlocks& right_blocks);
    
    // Another implementation, slower at large N
    void multiply_right_dense_gemmBatched(
        cublasHandle_t handle,
        SomeHamiltonianTermBlocks& left_blocks,
        SomeHamiltonianTermBlocks& right_blocks);

    // Yet another implementation, slower at large N
    void multiply_right_dense_gemmStridedBatched(
        cublasHandle_t handle,
        SomeHamiltonianTermBlocks& left_blocks,
        SomeHamiltonianTermBlocks& right_blocks);

    // Compute (O_R^T * state^T) where O_R is sparse
    void multiply_right_sparse(
        cublasHandle_t handle, cusparseHandle_t handle_sp,
        SomeHamiltonianTermBlocks& blocks,
        cucpx* d_state_block);

    // Compute (O_L * (O_R^T * state^T)^T where O_L is dense
    void multiply_left_dense(
        cublasHandle_t handle,
        cudaStream_t parent_stream,
        SomeHamiltonianTermBlocks& left_blocks,
        SomeHamiltonianTermBlocks& right_blocks);

    void multiply_left_dense_gemmStridedBatched(
        cublasHandle_t handle,
        SomeHamiltonianTermBlocks& left_blocks,
        SomeHamiltonianTermBlocks& right_blocks);

    // Sum all the term contributions from a (O_L * state * O_R)
    // product, for a given block
    void sum_blocks(cublasHandle_t handle,
                    SomeHamiltonianTermBlocks& blocks);

    // dst += src, where size is the number of elements
    void d_add_vector(cublasHandle_t handle,
                      cucpx* dst,
                      cucpx* src,
                      int size);

    bool destroyed;
    bool construction_complete;
    bool debug;
};

// A row-major sparse matrix, stored in pinned memory
class PinnedRowMajorSpMat {
public:
    PinnedRowMajorSpMat(const RowMajorSpMat& sp);
    PinnedRowMajorSpMat(int rows, int cols, vector<CpxTriplet>& triplets);
    ~PinnedRowMajorSpMat();

    int rows; // number of rows
    int cols; // number of columns
    int nz_elems; // number of non-zero elements

    // non-zero values, of length nz_elems
    cucpx* h_values; 

    // indices into d_values, giving initial the first non-zero
    // element in each row, of length rows+1. the last element
    // is equal to nz_elems + d_row_ptr[0].
    int* h_row_ptr; 

    // column indices of the non-zero elements, of length nz_elems
    int* h_col_ind;

    size_t values_alloc_size;
    size_t row_ptr_alloc_size;
    size_t col_ind_alloc_size;

private:
    void init(const RowMajorSpMat& sp);
};

// Manages memory for a pinned memory matrix
class PinnedMat {
public:
    PinnedMat(int rows, int cols);
    ~PinnedMat();

    Map<Mat>* mat;
};
                    
//
// Holds a chunk of Hamiltonian operator data, both on the host and on the
// device. Each chunk can fit in device memory. When the whole Hamiltonian
// cannot fit in device memory, multiple chunks of data get shuffled around
//
class HamiltonianData {
public:
    HamiltonianData(FactorizedSpace& _space);
    HamiltonianData(const HamiltonianData& other);
    ~HamiltonianData();
    HamiltonianData& operator=(const HamiltonianData& other);

    void free_device_objects();

    // How many operators are in this chunk (sum of num_operators[i])
    int total_num_operators();

    // Allocate host memory based on num_operators
    void alloc_host_memory();

    // Free all the host memory, keeping device memory intact
    void free_host_memory();

    // Add an operator pair (must call alloc_host_memory() first)
    void add_op_pair(int left_idx, int idx,
                     FactorizedOperatorPair& ops_pair);

    // After all operator pairs were added, call this to prepare the
    // sparse matrices and copy the dense matrices to pinned memory
    void finalize_host_memory();

    // The required device memory for the stored number of operators.
    // Does not include the sparse memory, which is harder to predict
    // but is negligible at larger N values (at N=34 it's less than 1%
    // error).
    size_t predicted_d_alloc_size();

    // Allocate and copy the host data to the device
    void copy_to_device(
        CudaAllocator& allocator,
        cudaStream_t stream = 0);

    // How many operators are in this chunk, indexed by number
    // of left indices
    map< int, int > num_operators;
    
    //// Host memory, for easy copying to device ////

    // (Using pinned memory as the matrix stroage)

    // A shared pointer to a sparse matrix in pinned memory
    typedef boost::shared_ptr<PinnedRowMajorSpMat> PinnedSp_ptr;
    typedef boost::shared_ptr<PinnedMat> PinnedMat_ptr;

    typedef map< int, map<MatBlock, PinnedSp_ptr> > h_sparse_ops_t;
    typedef map< int, map< MatBlock, PinnedMat_ptr > > h_dense_ops_t;

    // left_idx -> TOP/BOTTOM -> non-zero triplets
    typedef map< int,
                 map< MatBlock, vector<CpxTriplet> > > h_sparse_ops_work_t;

    // Left operators are indexed by number of left indices.
    // Right operators are indexed by number of right indices.
    // Sparse blocks as stored as one big sparse matrix.
    // Dense blocks are stored as a row of dense blocks in memory.
    h_sparse_ops_t h_left_sparse_ops;
    h_sparse_ops_t h_right_sparse_ops;

    // Dense matrices, sitting in pinned memory
    h_dense_ops_t h_left_dense_ops;
    h_dense_ops_t h_right_dense_ops;

    // Working memory for creating the sparse matrices: we store
    // them as dense first, then make them sparse.
    h_sparse_ops_work_t h_left_sparse_ops_work;
    h_sparse_ops_work_t h_right_sparse_ops_work;

    //// Device memory ////
    typedef map< int, map< MatBlock, SomeHamiltonianTermBlocks > > terms_t;

    terms_t terms;

    // Total allocated size on the host and device
    size_t h_alloc_size;
    size_t d_alloc_size;

    FactorizedSpace space;

private:
    // Return the device allocated size
    size_t copy_to_device(
        SomeHamiltonianTermBlocks& d_blocks,
        PinnedSp_ptr h_sparse_ops,
        PinnedMat_ptr dense_ops,
        int num_blocks,
        CudaAllocator& allocator,
        cudaStream_t stream);

    // Allocate pinned memory for a matrix
    /* PinnedMat* alloc_pinned_matrix(int rows, int cols); */

    /* void free_pinned_mem_map(h_dense_ops_t& dense_ops); */

    int state_block_size();
};

//
// Hamiltonian data and working device memory for acting on a state.
// Operators on one side are sparse, on the other side dense.
// This struct contains one set of blocks, either top or bottom.
// 
struct SomeHamiltonianTermBlocks {
    SomeHamiltonianTermBlocks();
    ~SomeHamiltonianTermBlocks();

    // Hamiltonian data: left and right operators
    /* CudaSparseMatrix* d_sparse_ops; */
    int d_sparse_rows;
    int d_sparse_cols;
    int d_sparse_nz_elems;

    cucpx* d_sparse_values;
    int* d_sparse_row_ptr;
    int* d_sparse_col_ind;

    cucpx* d_dense_ops;

    // result of sparse * state
    cucpx* d_sparse_prod;

    // result of sparse * state * dense
    cucpx* d_final_prod;

    // a vector of 1's
    /* cucpx* d_ones; */

    // result of summing the different products
    cucpx* d_sum;

    // how many blocks are stored (= number of terms)
    int num_blocks;
};

#endif
