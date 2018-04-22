#include <iostream>
#include <sstream>
#include <unistd.h>
#include <cuda_profiler_api.h>
#include "CudaHamiltonian.h"
#include "Timer.h"

#define min(a,b) ((a) < (b) ? (a) : (b))

using namespace std;

CudaHamiltonianInterface::CudaHamiltonianInterface() {}
CudaHamiltonianInterface::~CudaHamiltonianInterface() {}

CudaHamiltonian::CudaHamiltonian(
    FactorizedSpace& _space,
    MajoranaKitaevDisorderParameter& Jtensor,
    size_t available_memory,
    bool mock_hamiltonian,
    bool _debug) {

    basic_init(_space, -1, _debug);
    assert(available_memory > 0);

    // Split up operators into chunks. Allow for some wiggle room,
    // because the predictions for memory usage are not exact due to
    // memory alignment issues.
    prepare_op_chunks(net_available_memory(available_memory));

    // cout << "Found " << op_chunks.size() << " chunks" << endl;

    // Compute Hamiltonian in host memory
    generate_factorized_hamiltonian_terms(
        space, Jtensor, mock_hamiltonian, *this);

    finalize_construction_after_processing(available_memory);
}

CudaHamiltonian::CudaHamiltonian(
    FactorizedSpace& _space,
    vector<HamiltonianData_ptr> _op_chunks,
    int _device_id,
    bool _debug) {

    basic_init(_space, _device_id, _debug);
    op_chunks = _op_chunks;
}

CudaHamiltonian::CudaHamiltonian(
    FactorizedSpace& _space,
    int _device_id,
    bool _debug) {

    basic_init(_space, _device_id, _debug);
}

void CudaHamiltonian::basic_init(FactorizedSpace& _space,
                                 int _device_id, bool _debug) {
    host_mem_allocated = false;
    destroyed = false;
    construction_complete = false;
    device_id = _device_id;
    debug = _debug;
    d_ones_vector = 0;

    allocators[0] = 0;
    allocators[1] = 0;

    space = _space;

    // Initialize operator processing
    process_chunk_idx = 0;
    process_prev_num_ops.clear();

    for (int i = 0; i <= q; i++) {
        process_prev_num_ops[i] = 0;
        num_processed_ops[i] = 0;
    }
}

void CudaHamiltonian::finalize_construction_after_processing(
    size_t available_memory) {

    assert(done_processing_ops());
    set_device();

    alloc_device_constants();
    
    for (int i = 0; i < op_chunks.size(); i++) {
        op_chunks.at(i)->finalize_host_memory();
    }

    // Allocate device memory
    if (op_chunks.size() == 1) {
        allocators[0] = new CudaAllocator(available_memory);
        allocators[1] = 0;
    }
    else {
        allocators[0] = new CudaAllocator(available_memory / 2);
        allocators[1] = new CudaAllocator(available_memory / 2);
    }
    
    // Copy the first chunk to the device
    current_allocator = 0;
    op_chunks.at(0)->copy_to_device(*allocators[current_allocator]);

    // If only one chunk is used then we can release the host memory
    if (op_chunks.size() == 1) {
        op_chunks.at(0)->free_host_memory();
    }

    setup_cuda_structures();
    construction_complete = true;
}

// Record a pair of operators in a chunk for later processing. Create a new chunk
// if the current one is full.
void CudaHamiltonian::record_op_pair(int left_idx, size_t memory_per_chunk) {
    if (op_chunks.size() == 0) {
        op_chunks.push_back(HamiltonianData_ptr(new HamiltonianData(space)));
    }

    int chunk_idx = op_chunks.size() - 1;
    HamiltonianData& chunk = *op_chunks.at(chunk_idx);
    chunk.num_operators.at(left_idx)++;

    if (chunk.predicted_d_alloc_size() > memory_per_chunk) {
        // This chunk is out of memory, create a new one.
        // Once we push a new chunk, the 'chunk' reference is
        // invalidated.
        chunk.num_operators.at(left_idx)--;

        op_chunks.push_back(HamiltonianData_ptr(new HamiltonianData(space)));
        HamiltonianData& new_chunk = *op_chunks.at(op_chunks.size() - 1);
        new_chunk.num_operators.at(left_idx)++;

        if (new_chunk.predicted_d_alloc_size() > memory_per_chunk) {
            cerr << "Cannot fit even one operator in new chunk: "
                 << " predicted-size="
                 << new_chunk.predicted_d_alloc_size() 
                 << " memory-per-chunk=" << memory_per_chunk
                 << endl;
            exit(1);
        }
    }
}

void CudaHamiltonian::set_device() {
    if (device_id >= 0) {
        // cout << "Setting device to " << device_id << endl;
        checkCudaErrors(cudaSetDevice(device_id));
    }
}

void CudaHamiltonian::prepare_op_chunks(size_t available_memory) {
    size_t memory_per_chunk = available_memory / 2;

    for (int left_idx = 0; left_idx <= q; left_idx++) {
        int num_ops = num_factorized_hamiltonian_operator_pairs(
            space, left_idx);
        int num_ops_accounted_for = 0;

        while (num_ops_accounted_for < num_ops) {
            record_op_pair(left_idx, memory_per_chunk);
            num_ops_accounted_for++;
        }
    }

    // If it's only 2 chunks, it should have fit into one big chunk
    combine_two_chunks();
    assert(op_chunks.size() == 1 || op_chunks.size() > 2);

    // print_chunks();
}

void CudaHamiltonian::print_chunks() {
    // cout << "\nGlobal num operators: ";
    // for (int j = 0; j <= q; j++) {
    //     cout << "\t" << j << ":"
    //          << num_factorized_hamiltonian_operator_pairs(space, j);
    // }
    // cout << endl;

    cout << "\nPrepared chunks:\n";
    for (int i = 0; i < op_chunks.size(); i++) {
        HamiltonianData& chunk = *op_chunks.at(i);
        cout << "chunk=" << i << ":";

        for (int j = 0; j <= q; j++) {
            cout << "\t" << j << ":" << chunk.num_operators.at(j);
        }

        cout << "\t(total=" << chunk.total_num_operators() << ", "
             << "predicted " << chunk.predicted_d_alloc_size()/1e9 << " bytes)";
        cout << "\n";
    }
    cout << endl;
}

void CudaHamiltonian::alloc_host_memory() {
    set_device();
    
    for (int i = 0; i < op_chunks.size(); i++) {
        op_chunks.at(i)->alloc_host_memory();
    }

    host_mem_allocated = true;
}

void CudaHamiltonian::alloc_device_constants() {
    assert(op_chunks.size() > 0);
    set_device();
    
    d_zero = new CudaScalar(0., 0.);
    d_one = new CudaScalar(1., 0.);
        
    int max_num_operators = 0;

    // Allocate a ones vector of length equal to the maximal number
    // of blocks. We sum blocks by multiplying against the ones vector.
    // So shorter vectors of blocks can use the same long vector of
    // ones.
    for (int i = 0; i < op_chunks.size(); i++) {
        HamiltonianData& chunk = *op_chunks.at(i);

        for (int j = 0; j <= q; j++) {
            int n = chunk.num_operators.at(j);

            if (n > max_num_operators) {
                max_num_operators = n;
            }
        }
    }

    d_ones_vector = d_alloc_ones_vector(max_num_operators);
}

size_t CudaHamiltonian::total_h_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < op_chunks.size(); i++) {
        size += op_chunks.at(i)->h_alloc_size;
    }

    return size;
}

size_t CudaHamiltonian::total_d_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < op_chunks.size(); i++) {
        // Otherwise d_alloc_size not initialized yet (it is only
        // initialized when calling copy_to_device).
        assert(op_chunks.at(i)->d_alloc_size > 0);
        size += op_chunks.at(i)->d_alloc_size;
    }

    return size;
}

size_t CudaHamiltonian::predicted_total_d_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < op_chunks.size(); i++) {
        size += op_chunks.at(i)->predicted_d_alloc_size();
    }

    return size;
}

void CudaHamiltonian::print_memory_allocation() {
    cout << "Hamiltonian memory allocated:\n"
         << total_h_alloc_size() / 1e9 << " GB host memory\n"
         << predicted_total_d_alloc_size() / 1e9
         << " GB device memory (estimated)\n"
         << predicted_total_d_alloc_size()
        / 1e9 / (size_t) op_chunks.size()
         << " GB per chunk (estimated)\n\n";
}

void CudaHamiltonian::setup_cuda_structures() {
    // Sparse matrix description
    // (note: if we want to use hermitian matrices we need to set the
    // matrix type *and* the fill mode CUSPARSE_FILL_MODE_UPPER)
    checkCusparseErrors(cusparseCreateMatDescr(&sp_description));

    checkCusparseErrors(
        cusparseSetMatType(
            sp_description, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCusparseErrors(
        cusparseSetMatIndexBase(
            sp_description, CUSPARSE_INDEX_BASE_ZERO));
}

// When only two chunks are present we combine them into one big chunk.
void CudaHamiltonian::combine_two_chunks() {
    if (op_chunks.size() != 2) {
        return;
    }

    HamiltonianData& chunk1 = *op_chunks.at(0);
    HamiltonianData& chunk2 = *op_chunks.at(1);

    for (int i = 0; i <= q; i++) {
        chunk1.num_operators.at(i) += chunk2.num_operators.at(i);
    }

    // Remove the second element
    op_chunks.erase(op_chunks.begin() + 1, op_chunks.end());
    assert(op_chunks.size() == 1);
}

// Process an operator pair.
// 
// For the sparse operators:
// Layout a set of sparse operators as sparse matrices in host memory. The
// top blocks are stored as a column of blocks in a sparse row-major
// matrix. The bottom blocks as stored as another sparse matrix.
// This allows for easy copying to the device.
//
// If the operator is on the right side, it is also transposed, because
// CUDA only supposed sparse-dense multiplication (not dense-sparse).
void CudaHamiltonian::process(int left_idx, FactorizedOperatorPair& ops_pair) {
    if (op_chunks.size() == 2) {
        combine_two_chunks();
    }

    if (!host_mem_allocated) {
        alloc_host_memory();
    }

    if (seen_ops.count(left_idx) == 0) {
        seen_ops[left_idx] = 0;
    }

    int idx = seen_ops.at(left_idx);
    seen_ops.at(left_idx)++;

    HamiltonianData& trial_chunk = *op_chunks.at(process_chunk_idx);

    // idx is global index of the operator for given left_idx.
    // idx_in_chunk is the operator index within the current chunk.
    int idx_in_chunk = idx - process_prev_num_ops.at(left_idx);

    assert(idx_in_chunk >= 0);

    // Be defensive: use >=, not ==, then check equality
    if (idx_in_chunk >= trial_chunk.num_operators.at(left_idx)) {
        // trial_chunk is full, go to the next
        assert(idx_in_chunk == trial_chunk.num_operators.at(left_idx));
        assert(process_chunk_idx + 1 < op_chunks.size());

        process_prev_num_ops.at(left_idx) += trial_chunk.num_operators.at(left_idx);
        idx_in_chunk = idx - process_prev_num_ops.at(left_idx);
        assert(idx_in_chunk == 0);

        process_chunk_idx++;
    }

    HamiltonianData& chunk = *op_chunks.at(process_chunk_idx);
    assert(idx_in_chunk < chunk.num_operators.at(left_idx));

    chunk.add_op_pair(left_idx, idx_in_chunk, ops_pair);
    num_processed_ops.at(left_idx)++;
}

bool CudaHamiltonian::done_processing_ops() {
    assert(num_processed_ops.size() == q+1);
    map<int, int> expected_num_ops;

    for (int left_idx = 0; left_idx <= q; left_idx++) {
        expected_num_ops[left_idx] = 0;
    }

    for (int i = 0; i < op_chunks.size(); i++) {
        HamiltonianData& chunk = *op_chunks.at(i);

        for (int left_idx = 0; left_idx <= q; left_idx++) {
            expected_num_ops.at(left_idx) += chunk.num_operators.at(left_idx);
        }
    }

    for (int left_idx = 0; left_idx <= q; left_idx++) {
        assert(num_processed_ops.at(left_idx) <= expected_num_ops.at(left_idx));
        
        if (num_processed_ops.at(left_idx) < expected_num_ops.at(left_idx)) {
            return false;
        }
    }

    return true;
}

bool CudaHamiltonian::ready_to_act() {
    return construction_complete;
}

size_t CudaHamiltonian::net_available_memory(size_t available_memory) {
    // Allow some overhead for byte alignment
    size_t overhead = min(1000000, (size_t) (available_memory * 0.01));
    return available_memory - overhead;
}

CudaHamiltonian::~CudaHamiltonian() {
    destroy();
}

// Free the GPU data etc.
void CudaHamiltonian::destroy() {
    if (destroyed) {
        return;
    }

    set_device();
    
    // Free the device memory
    checkCusparseErrors(cusparseDestroyMatDescr(sp_description));

    for (int i = 0; i < 2; i++) {
        if (allocators[i] != 0) {
            delete allocators[i];
            allocators[i] = 0;
        }
    }

    delete d_zero;
    delete d_one;
    d_free(d_ones_vector);

    destroyed = true;
}

SomeHamiltonianTermBlocks::SomeHamiltonianTermBlocks() {
    num_blocks = 0;
    // d_sparse_ops = 0;
    d_sparse_rows = 0;
    d_sparse_cols = 0;
    d_sparse_nz_elems = 0;
    d_sparse_values = 0;
    d_sparse_row_ptr = 0;
    d_sparse_col_ind = 0;
    d_dense_ops = 0;
    d_sparse_prod = 0;
    d_final_prod = 0;
    d_sum = 0;
}

SomeHamiltonianTermBlocks::~SomeHamiltonianTermBlocks() {}

ActionContext CudaHamiltonian::prepare_to_act(CudaEvenState& output) {
    set_device();
    cuda_manager.release_all();

    assert(construction_complete);
    assert(op_chunks.size() > 0);
    assert(done_processing_ops());

    output.set_to_zero();

    ActionContext context(cuda_manager);
    return context;
}

void CudaHamiltonian::act_even_single_chunk_async(
    CudaEvenState& output,
    CudaEvenState& state,
    cudaEvent_t completion) {

    set_device();
    assert(ready_to_act());
    assert(op_chunks.size() == 1);
    ActionContext context = prepare_to_act(output);

    HamiltonianData& chunk = *op_chunks.at(0);
    launch_chunk_kernels(output, state, chunk,
                            context.kernel_stream, &completion);
    cuda_manager.release_all();
}

bool CudaHamiltonian::act_even_chunk_async(CudaEvenState& output,
                                           CudaEvenState& state,
                                           ActionContext& context) {
    set_device();
    assert(ready_to_act());
    
    if (context.resident_chunk == op_chunks.size()) {
        return false;
    }

    if (op_chunks.size() == 1) {
        HamiltonianData& chunk = *op_chunks.at(0);
        launch_chunk_kernels(output, state, chunk, context.kernel_stream);
    }
    else {
        assert(op_chunks.size() > 2);

        // Copy the next chunk into memory (if we're at the last chunk,
        // the first chunk is copied for next time.)
        int next_chunk = (context.resident_chunk + 1) % op_chunks.size();
        int next_allocator = (current_allocator + 1) % 2;
        allocators[next_allocator]->free_all();

        // copy_to_device is async
        op_chunks.at(next_chunk)->copy_to_device(
            *allocators[next_allocator], context.memcpy_stream);

        Timer timer;

        // Let the resident chunk work
        HamiltonianData& current_chunk = *op_chunks.at(context.resident_chunk);
        launch_chunk_kernels(output, state, current_chunk, context.kernel_stream);

        current_allocator = next_allocator;
    }

    context.resident_chunk++;
    return true;
}

void CudaHamiltonian::sync_act_even(ActionContext& context) {
    set_device();
    checkCudaErrors(cudaStreamSynchronize(context.memcpy_stream));
    checkCudaErrors(cudaStreamSynchronize(context.kernel_stream));
}

void CudaHamiltonian::act_even(CudaEvenState& output, CudaEvenState& state) {
    assert(ready_to_act());
    ActionContext context = prepare_to_act(output);

    while (act_even_chunk_async(output, state, context)) {
        sync_act_even(context);
    }
}

cudaStream_t CudaHamiltonian::act_on_blocks(
    CudaHandles& handles,
    Side sparse_side,
    SomeHamiltonianTermBlocks& left_blocks,
    cucpx* d_state_block,
    SomeHamiltonianTermBlocks& right_blocks) {

    cudaStream_t stream = cuda_manager.get_stream();
    handles.set_stream(stream);

    if (sparse_side == LEFT) {
        mult_l_sparse_r_dense_and_sum(handles.cublas_handle,
                                      handles.cusparse_handle,
                                      stream,
                                      left_blocks, d_state_block,
                                      right_blocks);
    }
    else {
        mult_r_sparse_l_dense_and_sum(handles.cublas_handle,
                                      handles.cusparse_handle,
                                      stream,
                                      left_blocks, d_state_block,
                                      right_blocks);
    }

    return stream;
}

void CudaHamiltonian::launch_chunk_kernels(
    CudaEvenState& output,
    CudaEvenState& state,
    HamiltonianData& ops,
    cudaStream_t final_stream,
    cudaEvent_t* completion) {

    // The stream should not be the default stream, because operations
    // in default stream only happen after *all* previous operations
    // are done.
    // checkCublasErrors(cublasSetStream(handle, act_stream));
    // checkCusparseErrors(cusparseSetStream(handle_sp, act_stream));

    // Multiply (O_L * state) * O_R

    // Even operators
    //
    // ( Ltop 0    )   ( tl  0  )   ( Rtop 0    )    
    // (    0 Lbot ) . (  0  br ) . (    0 Rbot ) = 
    // 
    //    ( Ltop.tl.Rtop  0            )
    //    (            0  Lbot.br.Rbot )
    // 

    CudaHandles handles = cuda_manager.get_handles();
    vector<cudaStream_t> block_streams;

    if (debug) cout << " In act_even_on_chunk" << endl;
    Timer timer;

    // left 0, right 4
    block_streams.push_back(act_on_blocks(
        handles, LEFT,
        ops.terms.at(0).at(TOP_BLOCK),
        state.d_top_left_block,
        ops.terms.at(0).at(TOP_BLOCK)));

    block_streams.push_back(act_on_blocks(
        handles, LEFT,
        ops.terms.at(0).at(BOTTOM_BLOCK),
        state.d_bottom_right_block,
        ops.terms.at(0).at(BOTTOM_BLOCK)));

    // left 2, right 2
    block_streams.push_back(act_on_blocks(
        handles, LEFT,
        ops.terms.at(2).at(TOP_BLOCK),
        state.d_top_left_block,
        ops.terms.at(2).at(TOP_BLOCK)));

    block_streams.push_back(act_on_blocks(
        handles, LEFT,
        ops.terms.at(2).at(BOTTOM_BLOCK),
        state.d_bottom_right_block,
        ops.terms.at(2).at(BOTTOM_BLOCK)));

    // left 4, right 0
    block_streams.push_back(act_on_blocks(
        handles, RIGHT,
        ops.terms.at(4).at(TOP_BLOCK),
        state.d_top_left_block,
        ops.terms.at(4).at(TOP_BLOCK)));

    block_streams.push_back(act_on_blocks(
        handles, RIGHT,
        ops.terms.at(4).at(BOTTOM_BLOCK),
        state.d_bottom_right_block,
        ops.terms.at(4).at(BOTTOM_BLOCK)));

    // Odd operators
    //
    // ( 0     Ltop )   ( tl  0  )   ( 0    Rtop )   
    // ( Lbot  0    ) . (  0  br ) . ( Rbot 0    ) = 
    // 
    //    ( Ltop.br.Rbot  0            )
    //    (            0  Lbot.tl.Rtop )
    // 

    // left 1, right 3
    block_streams.push_back(act_on_blocks(
        handles, LEFT,
        ops.terms.at(1).at(TOP_BLOCK),
        state.d_bottom_right_block,
        ops.terms.at(1).at(BOTTOM_BLOCK)));

    block_streams.push_back(act_on_blocks(
        handles, LEFT,
        ops.terms.at(1).at(BOTTOM_BLOCK),
        state.d_top_left_block,
        ops.terms.at(1).at(TOP_BLOCK)));

    // left 3, right 1

    // computes: (O_R^top)^T * state^T
    block_streams.push_back(act_on_blocks(
        handles, RIGHT,
        ops.terms.at(3).at(TOP_BLOCK),
        state.d_bottom_right_block,
        ops.terms.at(3).at(BOTTOM_BLOCK)));

    block_streams.push_back(act_on_blocks(
        handles, RIGHT,
        ops.terms.at(3).at(BOTTOM_BLOCK),
        state.d_top_left_block,
        ops.terms.at(3).at(TOP_BLOCK)));

    // if (debug) timer.print_msec(" done: ");

    //// sum multiplication results ////

    // CudaHandles handles = cuda_manager.get_handles();
    handles.set_stream(final_stream);

    // Wait for the blocks to finish computing products
    CudaEvents block_events(block_streams.size());

    for (int i = 0; i < block_streams.size(); i++) {
        block_events.record(i, block_streams.at(i));
    }

    block_events.wait_all(final_stream);

    // Copy to the output state
    for (int left_idx = 0; left_idx <= q; left_idx++) {
        d_add_vector(handles.cublas_handle,
                     output.d_top_left_block,
                     ops.terms.at(left_idx).at(TOP_BLOCK).d_sum,
                     space.state_block_size());

        d_add_vector(handles.cublas_handle,
                     output.d_bottom_right_block,
                     ops.terms.at(left_idx).at(BOTTOM_BLOCK).d_sum,
                     space.state_block_size());
    }

    if (debug) timer.print_msec(" summed results: ");

    if (completion != 0) {
        checkCudaErrors(cudaEventRecord(*completion, final_stream));
    }
}

void CudaHamiltonian::d_add_vector(cublasHandle_t handle,
                                   cucpx* dst, cucpx* src, int size) {
    // y = \alpha * x + y
    checkCublasErrors(cublasZaxpy(
        handle,
        size,
        d_one->ptr, // alpha
        src, 1,    // x (stride=1)
        dst, 1));  // y (stride=1)
}

void CudaHamiltonian::set_sum_to_zero(
    SomeHamiltonianTermBlocks& blocks,
    cudaStream_t stream) {

    // Important to do this in the action stream, not the
    // default stream because the default stream will block
    // waiting for the memcpys.

    checkCudaErrors(cudaMemsetAsync(
                        blocks.d_sum, 0, 
                        sizeof(cucpx) * space.state_block_size(),
                        stream));
}

void CudaHamiltonian::mult_l_sparse_r_dense_and_sum(
    cublasHandle_t handle,
    cusparseHandle_t handle_sp,
    cudaStream_t stream,
    SomeHamiltonianTermBlocks& left_blocks,
    cucpx* d_state_block,
    SomeHamiltonianTermBlocks& right_blocks) {

    if (left_blocks.num_blocks > 0) {
        multiply_left_sparse(handle, handle_sp,
                             left_blocks, d_state_block);
        multiply_right_dense(handle, stream, left_blocks, right_blocks);
        sum_blocks(handle, left_blocks);
    }
    else {
        set_sum_to_zero(left_blocks, get_cublas_stream(handle));
    }
}

void CudaHamiltonian::mult_r_sparse_l_dense_and_sum(
    cublasHandle_t handle,
    cusparseHandle_t handle_sp,
    cudaStream_t stream,
    SomeHamiltonianTermBlocks& left_blocks,
    cucpx* d_state_block,
    SomeHamiltonianTermBlocks& right_blocks) {

    if (left_blocks.num_blocks > 0) {
        multiply_right_sparse(handle, handle_sp,
                              right_blocks, d_state_block);
        multiply_left_dense(handle, stream, left_blocks, right_blocks);
        sum_blocks(handle, left_blocks);
    }
    else {
        set_sum_to_zero(left_blocks, get_cublas_stream(handle));
    }
}

//
// Compute a (sparse-blocks * state-block) product.
// sparse-blocks are stored a sparse matrix that contains a column of
// blocks, where each block is the block (top/bottom) of a left operator.
// The product is again a column of blocks, with each block equal to
// (top/bottom part of) an (O_L * state) product. This is written to
// left_blocks.d_sparse_prod.
// 
void CudaHamiltonian::multiply_left_sparse(
    cublasHandle_t handle,
    cusparseHandle_t handle_sp,
    SomeHamiltonianTermBlocks& blocks,
    cucpx* d_state_block) {

    assert(blocks.num_blocks > 0);

    int A_rows = blocks.d_sparse_rows;
    int A_cols = blocks.d_sparse_cols;

    int B_rows = A_cols;
    int B_cols = space.right.D / 2;

    int C_rows = A_rows;
    // int C_cols = B_cols;

    checkCusparseErrors(
        cusparseZcsrmm(handle_sp,
                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                       A_rows,
                       B_cols,
                       A_cols,
                       blocks.d_sparse_nz_elems, // nnz
                       d_one->ptr, // alpha
                       sp_description,
                       blocks.d_sparse_values,  // A values
                       blocks.d_sparse_row_ptr, // A row ptr
                       blocks.d_sparse_col_ind, // A col ind
                       d_state_block, // B
                       B_rows,
                       d_zero->ptr, // beta
                       blocks.d_sparse_prod, // C
                       C_rows));
}

//
// This function takes over 85% of the time for N >= 36, because it is
// reponsiblse for doing dense-dense multiplication on operators with 2 left
// and 2 right indices, so their number is O(N^2). This is the largest set
// of operators to mulfiplis O(N^2). This is the largest set
// of operators to multiply.
//
// The result of a (sparse-blocks * state-block) is a column of
// blocks, of (top/bottom part of) a (sp_left_op * d_state_block)
// product.  We want to multiply each such block with a corresponding
// block in right_blocks.d_dense_ops, which is a row of ops.  We do this
// using a strided-batched matrix-matrix multiplication. We need to be
// careful with the left matrix because the layout is column-major, so
// the blocks are not stored contiguously in memory. In practice this
// means we need to set the lda and stride of this matrix
// correctly. (Basically, each left block is a submatrix of the left
// matrix with a non-trivial layout.)
//
// left_blocks contains the (sparse-blocks * state-block) result.
// right_blocks contains the dense right operators to multiply by.
// The result is written to left_blocks.d_final_prod.
//
// This implementation starts a stream for each block, running a gemm.
// This is the fastest implementation I found for N > 34. The streams stay
// synchronized by using CUDA events.
// 
void CudaHamiltonian::multiply_right_dense(
    cublasHandle_t handle,
    cudaStream_t parent_stream,
    SomeHamiltonianTermBlocks& left_blocks,
    SomeHamiltonianTermBlocks& right_blocks) {

    assert(left_blocks.num_blocks == right_blocks.num_blocks);
    int num_blocks = left_blocks.num_blocks;

    int A_rows = space.left.D / 2;
    int A_cols = space.right.D / 2;
    int leading_dim_A = space.left_block_rows() * num_blocks;
    // stride just on the rows -- the A blocks are stores in a column)
    int A_stride = A_rows;

    int B_rows = A_cols;
    int B_cols = space.right.D / 2;
    int B_stride = B_rows * B_cols;

    int C_rows = A_rows;
    int C_cols = B_cols;
    int C_stride = C_rows * C_cols;

    vector<cudaStream_t> child_streams =
        cuda_manager.get_streams(num_blocks);

    CudaEvents child_events(num_blocks);
    CudaEvent parent_event;
    parent_event.record(parent_stream);

    // if (debug) cout << "  in multiply_right_dense, calling gemm"
    //                 << " num_blocks=" << num_blocks
    //                 << endl;
    // if (debug) cudaProfilerStart();
    Timer timer;

    for (int i = 0; i < num_blocks; i++) {
        // Each block multiplication runs in its own stream
        checkCublasErrors(cublasSetStream(handle, child_streams.at(i)));

        // This stream should ...
        
        cucpx* A = left_blocks.d_sparse_prod + i * A_stride;
        cucpx* B = right_blocks.d_dense_ops + i * B_stride;
        cucpx* C = left_blocks.d_final_prod + i * C_stride;

        // 1. Wait for the parent
        parent_event.wait(child_streams.at(i));

        // 2. Compute
        checkCublasErrors(cublasZgemm(
                              handle,
                              CUBLAS_OP_N,   // don't change on A
                              CUBLAS_OP_N,   // don't change on B
                              A_rows,        // num A rows
                              B_cols,        // num B columns
                              A_cols,        // num A columns
                              d_one->ptr,     // alpha
                              A,             // A = O_L * state_block
                              leading_dim_A, // lda = sparse matrix rows
                              B,             // B = O_R
                              B_rows,        // ldb = B rows
                              d_zero->ptr,    // beta
                              C,             // C = output, same shape as state
                              C_rows));      // ldc = C rows

        // 3. Signal that it's done
        child_events.record(i, child_streams.at(i));
    }

    // Set back to the parent stream
    checkCublasErrors(cublasSetStream(handle, parent_stream));

    // Parent should wait for children to finish
    for (int i = 0; i < num_blocks; i++) {
        child_events.wait_all(parent_stream);
    }

    // Events will be destroyed once we return, even though the kernels are still
    // running. That's okay -- the events will be released once they are complete.
}

//////////////////////////////////////////////////////////////////////
// In this implementation we use batched gemm. For N < 34 it is
// faster, but for N > 34 using gemm in multiple streams it is slower.
//////////////////////////////////////////////////////////////////////
void CudaHamiltonian::multiply_right_dense_gemmBatched(
    cublasHandle_t handle,
    SomeHamiltonianTermBlocks& left_blocks,
    SomeHamiltonianTermBlocks& right_blocks) {

    assert(left_blocks.num_blocks == right_blocks.num_blocks);
    int num_blocks = left_blocks.num_blocks;

    int A_rows = space.left.D / 2;
    int A_cols = space.right.D / 2;
    int leading_dim_A = space.left_block_rows() * num_blocks;
    // stride just on the rows -- the A blocks are stores in a column)
    int A_stride = A_rows;

    int B_rows = A_cols;
    int B_cols = space.right.D / 2;
    int B_stride = B_rows * B_cols;

    int C_rows = A_rows;
    int C_cols = B_cols;
    int C_stride = C_rows * C_cols;

    const cucpx** h_A = new const cucpx*[num_blocks];
    const cucpx** h_B = new const cucpx*[num_blocks];
    cucpx** h_C = new cucpx*[num_blocks];

    int alloc_size = sizeof(cucpx*) * num_blocks;
    const cucpx** d_A = (const cucpx**) d_alloc(alloc_size);
    const cucpx** d_B = (const cucpx**) d_alloc(alloc_size);
    cucpx** d_C = (cucpx**) d_alloc(alloc_size);

    for (int i = 0; i < num_blocks; i++) {
        h_A[i] = left_blocks.d_sparse_prod + i * A_stride;
        h_B[i] = right_blocks.d_dense_ops + i * B_stride;
        h_C[i] = left_blocks.d_final_prod + i * C_stride;
    }

    checkCudaErrors(cudaMemcpy(d_A, h_A, alloc_size,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, alloc_size,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, alloc_size,
                               cudaMemcpyHostToDevice));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    checkCublasErrors(cublasZgemmBatched(
                          handle,
                          CUBLAS_OP_N,   // don't change on A
                          CUBLAS_OP_N,   // don't change on B
                          A_rows,        // num A rows
                          B_cols,        // num B columns
                          A_cols,        // num A columns
                          d_one->ptr,     // alpha
                          d_A,             // A = O_L * state_block
                          leading_dim_A, // lda = sparse matrix rows
                          d_B,             // B = O_R
                          B_rows,        // ldb = B rows
                          d_zero->ptr,    // beta
                          d_C,             // C = output, same shape as state
                          C_rows,        // ldc = C rows
                          num_blocks));      

    d_free(d_A);
    d_free(d_B);
    d_free(d_C);
}

//////////////////////////////////////////////////////////////////////
// This is another implementation that uses strided batch. For
// N < 34 it is faster, but for N > 34 using gemm in multiple streams
// it is slower.
//////////////////////////////////////////////////////////////////////
void CudaHamiltonian::multiply_right_dense_gemmStridedBatched(
    cublasHandle_t handle,
    SomeHamiltonianTermBlocks& left_blocks,
    SomeHamiltonianTermBlocks& right_blocks) {

    assert(left_blocks.num_blocks == right_blocks.num_blocks);
    int num_blocks = left_blocks.num_blocks;

    int A_rows = space.left.D / 2;
    int A_cols = space.right.D / 2;
    int leading_dim_A = space.left_block_rows() * num_blocks;
    // stride just on the rows -- the A blocks are stores in a column)
    int A_stride = A_rows;

    int B_rows = A_cols;
    int B_cols = space.right.D / 2;
    int B_stride = B_rows * B_cols;

    int C_rows = A_rows;
    int C_cols = B_cols;
    int C_stride = C_rows * C_cols;

    checkCublasErrors(cublasZgemmStridedBatched(
        handle,
        CUBLAS_OP_N,     // don't change on A
        CUBLAS_OP_N,     // don't change on B
        A_rows,          // num A  rows
        B_cols,          // num B columns
        A_cols,          // num A columns
        d_one->ptr,       // alpha
        left_blocks.d_sparse_prod,  // A = O_L * state_block
        leading_dim_A,   // lda = sparse matrix rows
        A_stride,
        right_blocks.d_dense_ops,    // B = O_R
        B_rows,          // ldb = B rows
        B_stride,
        d_zero->ptr,      // beta
        left_blocks.d_final_prod, // C = output, same shape as state
        C_rows,          // ldc = C rows
        C_stride,
        // batch count: how many mat-mat mults to run
        num_blocks)); 
}

// Multiply (state-block * sparse-O_R-block). cuSPARSE doesn't have
// a right-multiply operation so we need to transpose. The right
// sparse matrix is already transposed. Here we will transpose the state
// (B) as part of the multiplication. The result will be the transpose of
// what we want, and we will take this into account when doing the
// subsequent left multiplication.
//
// 'blocks' holds a column of transposed blocks. We multiply this with
// the transposed state.
void CudaHamiltonian::multiply_right_sparse(
    cublasHandle_t handle,
    cusparseHandle_t handle_sp,
    SomeHamiltonianTermBlocks& blocks,
    cucpx* d_state_block) {

    assert(blocks.num_blocks > 0);

    int A_rows = blocks.d_sparse_rows;
    int A_cols = blocks.d_sparse_cols;

    cucpx* B = d_state_block;
    int B_rows = space.state_block_rows();
    int B_cols = space.state_block_cols();

    // Because B is transposed below
    assert(A_cols == B_cols);

    cucpx* C = blocks.d_sparse_prod;
    int C_rows = A_rows;

    checkCusparseErrors(
        cusparseZcsrmm2(handle_sp,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_TRANSPOSE,
                        A_rows,
                        B_rows, // num cols of B^T
                        A_cols,
                        blocks.d_sparse_nz_elems, // nnz
                        d_one->ptr, // alpha
                        sp_description,
                        blocks.d_sparse_values,  // A values
                        blocks.d_sparse_row_ptr, // A row ptr
                        blocks.d_sparse_col_ind, // A col ind
                        B,
                        B_rows,
                        d_zero->ptr, // beta
                        C,
                        C_rows));
}

// Takes the result of (O_R^T * state^T) and computes:
// O_L * (O_R^T * state^T)^T = O_L * state * O_R.
// O_L is dense. Writes the result to left_blocks.d_final_prod.
//
// See multiply_right_dense() for details of memory layout.
void CudaHamiltonian::multiply_left_dense(
    cublasHandle_t handle,
    cudaStream_t parent_stream,
    SomeHamiltonianTermBlocks& left_blocks,
    SomeHamiltonianTermBlocks& right_blocks) {

    assert(left_blocks.num_blocks == right_blocks.num_blocks);
    int num_blocks = left_blocks.num_blocks;

    int A_rows = space.left_block_rows();
    int A_cols = space.left_block_cols();
    int leading_dim_A = A_cols;
    int A_stride = A_rows * A_cols;

    // B has shape of a transposed state
    int B_rows = space.state_block_cols();
    int B_cols = space.state_block_rows();
    int leading_dim_B = B_rows * right_blocks.num_blocks;

    // B stride: just the rows, stored in column of blocks
    int B_stride = B_rows;

    // We are going to transpose B, so this is the correct condition
    assert(A_cols == B_cols); 
    assert(left_blocks.num_blocks == right_blocks.num_blocks);

    int C_rows = A_rows;
    int C_cols = B_rows; // We are going to transpose B
    int C_stride = C_rows * C_cols;

    vector<cudaStream_t> child_streams =
        cuda_manager.get_streams(num_blocks);

    CudaEvents child_events(num_blocks);
    CudaEvent parent_event;
    parent_event.record(parent_stream);

    for (int i = 0; i < num_blocks; i++) {
        // Each block multiplication runs in its own stream
        checkCublasErrors(cublasSetStream(handle, child_streams.at(i)));

        cucpx* A = left_blocks.d_dense_ops + i * A_stride;
        cucpx* B = right_blocks.d_sparse_prod + i * B_stride;
        cucpx* C = left_blocks.d_final_prod + i * C_stride;

        parent_event.wait(child_streams.at(i));

        checkCublasErrors(cublasZgemm(
                              handle,
                              CUBLAS_OP_N,   // don't change on A
                              CUBLAS_OP_T,   // transpose B
                              A_rows,        
                              B_rows,        // num B^T columns
                              A_cols,        
                              d_one->ptr,     // alpha
                              A,             
                              leading_dim_A, // lda
                              B,             
                              leading_dim_B, // ldb
                              d_zero->ptr,    // beta
                              C,             
                              C_rows));      // ldc

        child_events.record(i, child_streams.at(i));
    }

    // Set back to the parent stream
    checkCublasErrors(cublasSetStream(handle, parent_stream));

    // Parent should wait for children to finish
    child_events.wait_all(parent_stream);
}


//////////////////////////////////////////////////////////////////////
// In this implementation we use batched gemm. For N < 34 it is
// faster, but for N > 34 using gemm in multiple streams it is slower.
//////////////////////////////////////////////////////////////////////
void CudaHamiltonian::multiply_left_dense_gemmStridedBatched(
    cublasHandle_t handle,
    SomeHamiltonianTermBlocks& left_blocks,
    SomeHamiltonianTermBlocks& right_blocks) {

    assert(left_blocks.num_blocks == right_blocks.num_blocks);

    int A_rows = space.left_block_rows();
    int A_cols = space.left_block_cols();
    int leading_dim_A = A_cols;

    // B has shape of a transposed state
    int B_rows = space.state_block_cols();
    int B_cols = space.state_block_rows();
    int leading_dim_B = B_rows * right_blocks.num_blocks;

    // We are going to transpose B, so this is the correct condition
    assert(A_cols == B_cols); 
    assert(left_blocks.num_blocks == right_blocks.num_blocks);

    int C_rows = A_rows;
    int C_cols = B_rows; // We are going to transpose B

    cublasStatus_t status = cublasZgemmStridedBatched(
        handle,
        CUBLAS_OP_N,     // don't transpose A
        CUBLAS_OP_T,     // transpose B
        A_rows,          // num A rows
        B_rows,          // num B^T columns
        A_cols,          // num A columns
        d_one->ptr,       // alpha
        left_blocks.d_dense_ops,  // A = O_L
        leading_dim_A,   // lda = sparse matrix rows
        A_rows * A_cols, // A stride
        right_blocks.d_sparse_prod,    // B = O_R^T * state^T
        leading_dim_B,   // ldb is big because B is column of blocks
        B_rows,          // B stride: just the rows, stored in column of
                         // blocks
        d_zero->ptr,      // beta
        left_blocks.d_final_prod, // C = output, same shape as state
        C_rows,          // ldc = C rows
        C_rows * C_cols, // C stride
        // batch count: how many mat-mat mults to run
        left_blocks.num_blocks); 

    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "Error in right multiply: " << status << endl;
        exit(1);
    }
}

// Compute sum of top/bottom blocks. This can be phrased by treating
// the blocks together as a matrix (each column is a single block,
// the columns are the different blocks coming from different
// operators, and note this is column-major). The sum is then
// obtained by multiplying the 'block matrix' on the right by a
// column vector of ones.
//
// The blocks all have the shape of a state block.
void CudaHamiltonian::sum_blocks(
    cublasHandle_t handle, SomeHamiltonianTermBlocks& blocks) {
    
    int A_rows = space.state_block_size();
    int A_cols = blocks.num_blocks;

    // y = alpha * A * x + beta * y
    cublasStatus_t status = cublasZgemv(
        handle,
        CUBLAS_OP_N,
        A_rows,
        A_cols,
        d_one->ptr, // alpha
        blocks.d_final_prod, // A
        A_rows,
        d_ones_vector, // x
        1, // incx
        d_zero->ptr, // beta
        blocks.d_sum, // y
        1); // incy

    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "Error in summing blocks: " << status << endl;
        exit(1);
    }
}

HamiltonianData::HamiltonianData(FactorizedSpace& _space) {
    space = _space;
    h_alloc_size = 0;
    d_alloc_size = 0;

    for (int left_idx = 0; left_idx <= q; left_idx++) {
        num_operators[left_idx] = 0;
    }
}

// Only copies bare essentials placing new objects in STL vector
HamiltonianData::HamiltonianData(const HamiltonianData& other) {
    space = other.space;
    num_operators = other.num_operators;
    h_alloc_size = other.h_alloc_size;
    d_alloc_size = other.d_alloc_size;
}

// Only copies bare essentials placing new objects in STL vector
HamiltonianData& HamiltonianData::operator=(const HamiltonianData& other) {
    space = other.space;
    num_operators = other.num_operators;
    h_alloc_size = other.h_alloc_size;
    d_alloc_size = other.d_alloc_size;
    return *this;
}

HamiltonianData::~HamiltonianData() {
    free_device_objects();
}

void HamiltonianData::free_device_objects() {
    terms.clear();
}

int HamiltonianData::total_num_operators() {
    int n = 0;
    
    for (int left_idx = 0; left_idx <= q; left_idx++) {
        n += num_operators.at(left_idx);
    }

    return n;
}

size_t HamiltonianData::predicted_d_alloc_size() {
    size_t total_alloc_size = 0;
    
    for (int left_idx = 0; left_idx <= q; left_idx++) {
        int num_ops = num_operators.at(left_idx);

        // Which side is dense?
        size_t dense_op_size =
            left_idx <= q/2 ?
            space.right_block_size() :
            space.left_block_size();

        size_t sparse_op_rows =
            left_idx <= q/2 ?
            space.left_block_rows() :
            space.right_block_rows();

        size_t work_alloc_size =
            sizeof(cucpx) * space.state_block_size() * num_ops;

        size_t working_mem = 2 * work_alloc_size; // two multiplications

        size_t dense_blocks_mem = sizeof(cucpx) * num_ops * dense_op_size;

        // In our gamma matrix elements, there is one non-zero element
        // per row because each product of gamma matrices flips some
        // fixed number of qubits in the state
        size_t sparse_nonzero_elems = sparse_op_rows;

        size_t sparse_blocks_mem =
            num_ops * 
            CudaSparseMatrix::get_alloc_size(sparse_op_rows,
                                             sparse_nonzero_elems);

        size_t sum_result_mem = sizeof(cucpx) * space.state_block_size();

        total_alloc_size += 2 * ( // top/bottom
            + dense_blocks_mem
            + sparse_blocks_mem
            + working_mem
            + sum_result_mem );
    }

    return total_alloc_size;
}

void HamiltonianData::free_host_memory() {
    h_left_sparse_ops.clear();
    h_right_sparse_ops.clear();

    h_left_dense_ops.clear();
    h_right_dense_ops.clear();

    h_left_sparse_ops_work.clear();
    h_right_sparse_ops_work.clear();
}

void HamiltonianData::alloc_host_memory() {
    h_alloc_size = 0;
    
    for (int i = 0; i <= q/2; i++) {
        // Right dense (ignore the sparse)
        h_alloc_size +=
            sizeof(cucpx)
            * 2 * num_operators.at(i)
            * space.right_block_size();
    }

    for (int i = q/2 + 1; i <= q; i++) {
        // Left dense (ignore the sparse)
        h_alloc_size +=
            sizeof(cucpx)
            * 2 * num_operators.at(i)
            * space.left_block_size();
    }
    
    // Left sparse, right dense
    for (int i = 0; i <= q/2; i++) {
        // Dense memory
        h_right_dense_ops[q-i] = map<MatBlock, PinnedMat_ptr>();

        h_right_dense_ops[q-i][TOP_BLOCK] = PinnedMat_ptr(
            new PinnedMat(
                space.right_block_rows(),
                space.right_block_cols() * num_operators.at(i)));

        h_right_dense_ops[q-i][BOTTOM_BLOCK] = PinnedMat_ptr(
            new PinnedMat(
                space.right_block_rows(),
                space.right_block_cols() * num_operators.at(i)));

        // Sparse working memory
        h_left_sparse_ops_work[i] = map<MatBlock, vector<CpxTriplet> >();

        h_left_sparse_ops_work[i][TOP_BLOCK] = vector<CpxTriplet>();
        h_left_sparse_ops_work[i][BOTTOM_BLOCK] = vector<CpxTriplet>();
    }

    // Left dense, right sparse
    for (int i = q/2 + 1; i <= q; i++) {
        // Dense memory
        h_left_dense_ops[i] = map<MatBlock, PinnedMat_ptr>();

        h_left_dense_ops[i][TOP_BLOCK] = PinnedMat_ptr(
            new PinnedMat(
                space.left_block_rows(),
                space.left_block_cols() * num_operators.at(i)));

        h_left_dense_ops[i][BOTTOM_BLOCK] = PinnedMat_ptr(
            new PinnedMat(
                space.left_block_rows(),
                space.left_block_cols() * num_operators.at(i)));

        // Sparse working memory
        h_right_sparse_ops_work[q-i] = map<MatBlock, vector<CpxTriplet> >();

        h_right_sparse_ops_work[q-i][TOP_BLOCK] = vector<CpxTriplet>();
        h_right_sparse_ops_work[q-i][BOTTOM_BLOCK] = vector<CpxTriplet>();
    }
}

void HamiltonianData::finalize_host_memory() {
    // Left sparse, right dense
    for (int i = 0; i <= q/2; i++) {
        h_left_sparse_ops[i] = map<MatBlock, PinnedSp_ptr>();

        h_left_sparse_ops[i][TOP_BLOCK] = PinnedSp_ptr(
            new PinnedRowMajorSpMat(
                space.left_block_rows() * num_operators.at(i),
                space.left_block_cols(),
                h_left_sparse_ops_work.at(i).at(TOP_BLOCK)));

        h_left_sparse_ops[i][BOTTOM_BLOCK] = PinnedSp_ptr(
            new PinnedRowMajorSpMat(
                space.left_block_rows() * num_operators.at(i),
                space.left_block_cols(),
                h_left_sparse_ops_work.at(i).at(BOTTOM_BLOCK)));
    }

    // Left dense, right sparse
    for (int i = q/2 + 1; i <= q; i++) {
        h_right_sparse_ops[q-i] = map<MatBlock, PinnedSp_ptr>();

        h_right_sparse_ops[q-i][TOP_BLOCK] = PinnedSp_ptr(
            new PinnedRowMajorSpMat(
                space.right_block_rows() * num_operators.at(i),
                space.right_block_cols(),
                h_right_sparse_ops_work.at(q-i).at(TOP_BLOCK)));

        h_right_sparse_ops[q-i][BOTTOM_BLOCK] = PinnedSp_ptr(
            new PinnedRowMajorSpMat(
                space.right_block_rows() * num_operators.at(i),
                space.right_block_cols(),
                h_right_sparse_ops_work.at(q-i).at(BOTTOM_BLOCK)));
    }
    
    // Free the work memory
    h_left_sparse_ops_work.clear();
    h_right_sparse_ops_work.clear();
}

void HamiltonianData::add_op_pair(
    int left_idx, int idx,
    FactorizedOperatorPair& ops_pair) {

    assert(idx >= 0);

    // We have 4 blocks to copy: left/right x top/bottom
    int right_idx = q - left_idx;
    BlockShape shape = get_block_shape(left_idx);
    BlockOperator2x2 left(ops_pair.O_left, shape);
    BlockOperator2x2 right(ops_pair.O_right, shape);

    if (left_idx <= q/2) {
        // Left sparse
        add_nonzeros_to_triplets(
            h_left_sparse_ops_work.at(left_idx).at(TOP_BLOCK),
            left.top_block,
            idx * space.left_block_rows(), 0);

        add_nonzeros_to_triplets(
            h_left_sparse_ops_work.at(left_idx).at(BOTTOM_BLOCK),
            left.bottom_block,
            idx * space.left_block_rows(), 0);
            
        // Right dense
        h_right_dense_ops.at(right_idx).at(TOP_BLOCK)->mat->block(
            0, idx * space.right_block_cols(),
            space.right_block_rows(), space.right_block_cols()) =
            right.top_block;

        h_right_dense_ops.at(right_idx).at(BOTTOM_BLOCK)->mat->block(
            0, idx * space.right_block_cols(),
            space.right_block_rows(), space.right_block_cols()) =
            right.bottom_block;
    }
    else {
        // Left dense
        h_left_dense_ops.at(left_idx).at(TOP_BLOCK)->mat->block(
            0, idx * space.left_block_cols(),
            space.left_block_rows(), space.left_block_cols()) =
            left.top_block;

        h_left_dense_ops.at(left_idx).at(BOTTOM_BLOCK)->mat->block(
            0, idx * space.left_block_cols(),
            space.left_block_rows(), space.left_block_cols()) =
            left.bottom_block;

        // Right sparse -- need to transpose, because sparse multiplication
        // is only supported on the left
        add_nonzeros_to_triplets(
            h_right_sparse_ops_work.at(right_idx).at(TOP_BLOCK),
            right.top_block.transpose(),
            idx * space.right_block_rows(), 0);

        add_nonzeros_to_triplets(
            h_right_sparse_ops_work.at(right_idx).at(BOTTOM_BLOCK),
            right.bottom_block.transpose(),
            idx * space.right_block_rows(), 0);
    }
}

void HamiltonianData::copy_to_device(CudaAllocator& allocator,
                                     cudaStream_t stream) {
    d_alloc_size = 0;

    // Remove any leftovers from last time
    free_device_objects();

    // Left sparse
    for (h_sparse_ops_t::iterator iter = h_left_sparse_ops.begin();
         iter != h_left_sparse_ops.end();
         ++iter) {

        int left_idx = iter->first;
        int right_idx = q - left_idx;

        terms[left_idx] = map<MatBlock, SomeHamiltonianTermBlocks>();
        terms[left_idx][TOP_BLOCK] = SomeHamiltonianTermBlocks();
        terms[left_idx][BOTTOM_BLOCK] = SomeHamiltonianTermBlocks();

        d_alloc_size += copy_to_device(
            terms.at(left_idx).at(TOP_BLOCK),
            h_left_sparse_ops.at(left_idx).at(TOP_BLOCK),
            h_right_dense_ops.at(right_idx).at(TOP_BLOCK),
            num_operators.at(left_idx),
            allocator,
            stream);

        d_alloc_size += copy_to_device(
            terms.at(left_idx).at(BOTTOM_BLOCK),
            h_left_sparse_ops.at(left_idx).at(BOTTOM_BLOCK),
            h_right_dense_ops.at(right_idx).at(BOTTOM_BLOCK),
            num_operators.at(left_idx),
            allocator,
            stream);
    }

    // Right sparse
    for (h_sparse_ops_t::iterator iter = h_right_sparse_ops.begin();
         iter != h_right_sparse_ops.end();
         ++iter) {

        int right_idx = iter->first;
        int left_idx = q - right_idx;

        assert(terms.count(left_idx) == 0);

        terms[left_idx] = map<MatBlock, SomeHamiltonianTermBlocks>();
        terms[left_idx][TOP_BLOCK] = SomeHamiltonianTermBlocks();
        terms[left_idx][BOTTOM_BLOCK] = SomeHamiltonianTermBlocks();

        d_alloc_size += copy_to_device(
            terms.at(left_idx).at(TOP_BLOCK),
            h_right_sparse_ops.at(right_idx).at(TOP_BLOCK),
            h_left_dense_ops.at(left_idx).at(TOP_BLOCK),
            num_operators.at(left_idx),
            allocator,
            stream);

        d_alloc_size += copy_to_device(
            terms.at(left_idx).at(BOTTOM_BLOCK),
            h_right_sparse_ops.at(right_idx).at(BOTTOM_BLOCK),
            h_left_dense_ops.at(left_idx).at(BOTTOM_BLOCK),
            num_operators.at(left_idx),
            allocator,
            stream);
    }
}

size_t HamiltonianData::copy_to_device(
    SomeHamiltonianTermBlocks& d_blocks,
    PinnedSp_ptr h_sparse_ops,
    PinnedMat_ptr dense_ops,
    int num_blocks,
    CudaAllocator& allocator,
    cudaStream_t stream) {

    size_t d_sum_alloc_size = sizeof(cucpx) * space.state_block_size();
    d_blocks.d_sum = (cucpx*) allocator.alloc(d_sum_alloc_size);

    if (num_blocks == 0) {
        return d_sum_alloc_size;
    }

    // assert(d_blocks.d_sparse_ops == 0);

    // d_blocks.d_sparse_ops = new CudaSparseMatrix(h_sparse_ops,
    //                                              &allocator,
    //                                              stream);

    // d_blocks.d_sparse_ops = new CudaSparseMatrix(h_sparse_ops,
    //                                              &allocator);

    d_blocks.d_sparse_rows = h_sparse_ops->rows;
    d_blocks.d_sparse_cols = h_sparse_ops->cols;
    d_blocks.d_sparse_nz_elems = h_sparse_ops->nz_elems;
    d_blocks.d_sparse_values =
        (cucpx*) allocator.alloc(h_sparse_ops->values_alloc_size);
    d_blocks.d_sparse_row_ptr =
        (int*) allocator.alloc(h_sparse_ops->row_ptr_alloc_size);
    d_blocks.d_sparse_col_ind =
        (int*) allocator.alloc(h_sparse_ops->col_ind_alloc_size);

    size_t dense_ops_alloc_size = 
        sizeof(cucpx) * dense_ops->mat->rows() * dense_ops->mat->cols();
    d_blocks.d_dense_ops =
        (cucpx*) allocator.alloc(dense_ops_alloc_size);

    checkCudaErrors(cudaMemcpyAsync(
                        d_blocks.d_sparse_values,
                        h_sparse_ops->h_values,
                        h_sparse_ops->values_alloc_size,
                        cudaMemcpyHostToDevice,
                        stream));

    checkCudaErrors(cudaMemcpyAsync(
                        d_blocks.d_sparse_row_ptr,
                        h_sparse_ops->h_row_ptr,
                        h_sparse_ops->row_ptr_alloc_size,
                        cudaMemcpyHostToDevice,
                        stream));

    checkCudaErrors(cudaMemcpyAsync(
                        d_blocks.d_sparse_col_ind,
                        h_sparse_ops->h_col_ind,
                        h_sparse_ops->col_ind_alloc_size,
                        cudaMemcpyHostToDevice,
                        stream));

    checkCudaErrors(cudaMemcpyAsync(
                        d_blocks.d_dense_ops,
                        dense_ops->mat->data(),
                        dense_ops_alloc_size,
                        cudaMemcpyHostToDevice,
                        stream));

    size_t work_alloc_size =
        sizeof(cucpx) * space.state_block_size() * num_blocks;

    d_blocks.d_sparse_prod = (cucpx*) allocator.alloc(work_alloc_size);
    d_blocks.d_final_prod = (cucpx*) allocator.alloc(work_alloc_size);

    d_blocks.num_blocks = num_blocks;

    size_t total_alloc_size =
        // d_blocks.d_sparse_ops->d_alloc_size
        h_sparse_ops->values_alloc_size
        + h_sparse_ops->row_ptr_alloc_size
        + h_sparse_ops->col_ind_alloc_size
        + dense_ops_alloc_size
        + 2 * work_alloc_size
        + sizeof(cucpx) * num_blocks
        + d_sum_alloc_size;

    return total_alloc_size;
}

PinnedRowMajorSpMat::PinnedRowMajorSpMat(const RowMajorSpMat& sp) {
    init(sp);
}

PinnedRowMajorSpMat::PinnedRowMajorSpMat(int rows, int cols,
                                         vector<CpxTriplet>& triplets) {
    RowMajorSpMat sp(rows, cols);
    sp.setFromTriplets(triplets.begin(), triplets.end());
    init(sp);
}

void PinnedRowMajorSpMat::init(const RowMajorSpMat& sp) {
    rows = sp.rows();
    cols = sp.cols();
    nz_elems = sp.nonZeros();

    values_alloc_size = sp_values_alloc_size(sp);
    row_ptr_alloc_size = sp_row_ptr_alloc_size(sp);
    col_ind_alloc_size = sp_col_ind_alloc_size(sp);

    if (values_alloc_size == 0) {
        h_values = 0;
        h_row_ptr = 0;
        h_col_ind = 0;
        return;
    }

    checkCudaErrors(cudaMallocHost((void**) &h_values,
                                   values_alloc_size));
    checkCudaErrors(cudaMallocHost((void**) &h_row_ptr,
                                   row_ptr_alloc_size));
    checkCudaErrors(cudaMallocHost((void**) &h_col_ind,
                                   col_ind_alloc_size));

    memcpy(h_values, sp.valuePtr(), values_alloc_size);
    memcpy(h_row_ptr, sp.outerIndexPtr(), row_ptr_alloc_size);
    memcpy(h_col_ind, sp.innerIndexPtr(), col_ind_alloc_size);
}

PinnedRowMajorSpMat::~PinnedRowMajorSpMat() {
    checkCudaErrors(cudaFreeHost(h_values));
    checkCudaErrors(cudaFreeHost(h_row_ptr));
    checkCudaErrors(cudaFreeHost(h_col_ind));
}

PinnedMat::PinnedMat(int rows, int cols) {
    mat = 0;
    cpx* h_pinned_mem;
    size_t alloc_size = sizeof(cpx) * rows * cols;

    if (alloc_size == 0) {
        return;
    }

    checkCudaErrors(cudaMallocHost((void**) &h_pinned_mem, alloc_size));
    mat = new Map<Mat>(h_pinned_mem, rows, cols);
}

PinnedMat::~PinnedMat() {
    if (mat != 0) {
        checkCudaErrors(cudaFreeHost(mat->data()));
        delete mat;
    }
}

ActionContext::ActionContext(CudaResourceManager& manager) {
    memcpy_stream = manager.get_stream();
    kernel_stream = manager.get_stream();
    resident_chunk = 0;
}

ActionContext::ActionContext(const ActionContext& other) {
    memcpy_stream = other.memcpy_stream;
    kernel_stream = other.kernel_stream;
    resident_chunk = other.resident_chunk;
}

ActionContext& ActionContext::operator=(const ActionContext& other) {
    memcpy_stream = other.memcpy_stream;
    kernel_stream = other.kernel_stream;
    resident_chunk = other.resident_chunk;
    return *this;
}
