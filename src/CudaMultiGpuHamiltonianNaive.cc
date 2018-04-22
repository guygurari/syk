#include "Timer.h"
#include "CudaMultiGpuHamiltonianNaive.h"

CudaMultiGpuHamiltonianNaive::CudaMultiGpuHamiltonianNaive(
    FactorizedSpace& _space,
    MajoranaKitaevDisorderParameter& Jtensor,
    size_t _available_memory_per_device,
    bool mock_hamiltonian,
    bool _debug) {

    vector<size_t> mem_per_device;

    for (int i = 0; i < cuda_get_num_devices(); i++) {
        mem_per_device.push_back(_available_memory_per_device);
    }

    init(_space, Jtensor, mem_per_device, mock_hamiltonian, _debug);
}

CudaMultiGpuHamiltonianNaive::CudaMultiGpuHamiltonianNaive(
    FactorizedSpace& _space,
    MajoranaKitaevDisorderParameter& Jtensor,
    vector<size_t>& _available_memory_per_device,
    bool mock_hamiltonian,
    bool _debug) {

    init(_space, Jtensor, _available_memory_per_device, mock_hamiltonian, _debug);
}

void CudaMultiGpuHamiltonianNaive::init(
    FactorizedSpace& _space,
    MajoranaKitaevDisorderParameter& Jtensor,
    vector<size_t>& _available_memory_per_device,
    bool mock_hamiltonian,
    bool _debug) {

    assert(_available_memory_per_device.size() <= cuda_get_num_devices());

    available_memory_per_device = _available_memory_per_device;
    space = _space;
    debug = _debug;

    cuda_set_device(0);
    d_one = boost::shared_ptr<CudaScalar>(new CudaScalar(1.0, 0.0));

    init_op_processing();
    prepare_op_chunks();
    assert(hamiltonians.size() == num_devices());

    generate_factorized_hamiltonian_terms(
        space, Jtensor, mock_hamiltonian, *this);

    for (int i = 1; i < hamiltonians.size(); i++) {
        // Allocate input/output states on all devices except the first
        cuda_set_device(i);
        input_states[i] = CudaEvenState_ptr(new CudaEvenState(space));
        output_states[i] = CudaEvenState_ptr(new CudaEvenState(space));
    }

    // Allocate memory on device 0 to copy the outputs of devices 1,...,n.
    cuda_set_device(0);
    output_copy = CudaEvenState_ptr(new CudaEvenState(space));
}

int CudaMultiGpuHamiltonianNaive::num_devices() {
    return available_memory_per_device.size();
}

void CudaMultiGpuHamiltonianNaive::init_op_processing() {
    process_hamiltonian_idx = 0;
}

static void print_chunks(CudaHamiltonian& H, int device_id) {
    cout << "\nPrepared chunks device_id=" << device_id << ":\n";
    for (int i = 0; i < H.op_chunks.size(); i++) {
        HamiltonianData& chunk = *H.op_chunks.at(i);
        cout << "chunk=" << i << ":";

        for (int j = 0; j <= q; j++) {
            cout << "\t" << j << ":" << chunk.num_operators.at(j);
        }

        cout << "\t(total=" << chunk.total_num_operators() << ")";
        cout << "\n";
    }
    cout << endl;
}

void CudaMultiGpuHamiltonianNaive::prepare_op_chunks() {
    // Each device has an associated. CudaHamiltonian.  We construct the chunks
    // for the next Hamiltonian, then construct that Hamiltonian. All devices
    // except the last are going to have a single chunk.
    vector<HamiltonianData_ptr> op_chunks;

    op_chunks.push_back(HamiltonianData_ptr(new HamiltonianData(space)));

    for (int left_idx = 0; left_idx <= q; left_idx++) {
        int num_ops = num_factorized_hamiltonian_operator_pairs(
            space, left_idx);
        int num_ops_accounted_for = 0;

        while (num_ops_accounted_for < num_ops) {
            int chunk_idx = op_chunks.size() - 1;
            HamiltonianData& chunk = *op_chunks.at(chunk_idx);

            num_ops_accounted_for++;
            chunk.num_operators.at(left_idx)++;

            int device_id = hamiltonians.size();
            size_t memory_per_chunk = get_memory_per_chunk(device_id);

            if (chunk.predicted_d_alloc_size() > memory_per_chunk) {
                // This chunk is out of memory.
                chunk.num_operators.at(left_idx)--;

                if (device_id < num_devices() - 1) {
                    // The device is full. Create its Hamiltonian and go to
                    // the next one.
                    assert(op_chunks.size() == 1 || op_chunks.size() > 2);
                    CudaHamiltonian_ptr H(
                        new CudaHamiltonian(space, op_chunks, device_id, debug));
                    hamiltonians.push_back(H);
                    op_chunks.clear();
                    // print_chunks(*H, device_id);
                }

                // Create a new chunk and add the current operator to i. Once
                // we push a new chunk, the 'chunk' reference is invalidated.
                op_chunks.push_back(
                    HamiltonianData_ptr(new HamiltonianData(space)));
                HamiltonianData& new_chunk =
                    *op_chunks.at(op_chunks.size() - 1);

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
    }

    // Remaining chunks are pushed into the last Hamiltonian.

    // Edge case: If we have exactly two chunks, they can be combined into one chunk
    if (op_chunks.size() == 2) {
        HamiltonianData& chunk1 = *op_chunks.at(0);
        HamiltonianData& chunk2 = *op_chunks.at(1);

        for (int i = 0; i <= q; i++) {
            chunk1.num_operators.at(i) += chunk2.num_operators.at(i);
        }

        // Remove the second element
        op_chunks.erase(op_chunks.begin() + 1, op_chunks.end());
        assert(op_chunks.size() == 1);
    }

    if (op_chunks.size() > 0) {
        assert(op_chunks.size() == 1 || op_chunks.size() > 2);
        int device_id = hamiltonians.size();
        CudaHamiltonian_ptr H(
            new CudaHamiltonian(space, op_chunks, device_id, debug));
        hamiltonians.push_back(H);
        // print_chunks(*H, device_id);
    }

    // cout << "\nGlobal num operators: ";
    // int total_num_ops = 0;
    // for (int j = 0; j <= q; j++) {
    //     cout << "\t" << j << ":"
    //          << num_factorized_hamiltonian_operator_pairs(space, j);
    //     total_num_ops += num_factorized_hamiltonian_operator_pairs(space, j);
    // }
    // cout << "\nTotal ops: " << total_num_ops << endl;
}

size_t CudaMultiGpuHamiltonianNaive::get_memory_per_chunk(int device_id) {
    size_t net_device_avail_memory =
        CudaHamiltonian::net_available_memory(
            available_memory_per_device.at(device_id));

    size_t work_alloc_size;
    
    if (device_id == 0) {
        // The first device allocates a copy of the output states
        work_alloc_size = space.state_alloc_size();
    }
    else {
        // These devices allocate input/output states
        work_alloc_size = 2 * space.state_alloc_size();
    }

    assert(net_device_avail_memory > work_alloc_size);
    net_device_avail_memory -= work_alloc_size;
    
    // All devices except the last get to use the whole device
    if (device_id < available_memory_per_device.size() - 1) {
        return net_device_avail_memory;
    }
    else {
        return net_device_avail_memory / 2;
    }
}

size_t CudaMultiGpuHamiltonianNaive::total_h_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < hamiltonians.size(); i++) {
        size += hamiltonians.at(i)->total_h_alloc_size();
    }

    return size;
}

size_t CudaMultiGpuHamiltonianNaive::total_d_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < hamiltonians.size(); i++) {
        size += hamiltonians.at(i)->total_d_alloc_size();
    }

    return size;
}


size_t CudaMultiGpuHamiltonianNaive::predicted_total_d_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < hamiltonians.size(); i++) {
        size += hamiltonians.at(i)->predicted_total_d_alloc_size();
    }

    return size;
}

void CudaMultiGpuHamiltonianNaive::print_memory_allocation() {
    cout << "Hamiltonian memory allocated:\n"
         << total_h_alloc_size() / 1e9 << " GB host memory\n"
         << predicted_total_d_alloc_size() / 1e9
         << " GB device memory (estimated)\n"
         << predicted_total_d_alloc_size()
        / 1e9 / (size_t) hamiltonians.size()
         << " GB per device (estimated)\n\n";
}

CudaMultiGpuHamiltonianNaive::~CudaMultiGpuHamiltonianNaive() {
    hamiltonians.clear();
}

void CudaMultiGpuHamiltonianNaive::process(int left_idx, 
                                           FactorizedOperatorPair& ops_pair) {
    assert(process_hamiltonian_idx < hamiltonians.size());
    CudaHamiltonian& H = *hamiltonians.at(process_hamiltonian_idx);

    H.process(left_idx, ops_pair);

    if (H.done_processing_ops()) {
        cout << "Hamiltonian " << process_hamiltonian_idx << " done processing"
             << endl;
        // Finalize this Hamiltonian (this allows it to free its host memory)
        H.finalize_construction_after_processing(
            available_memory_per_device.at(process_hamiltonian_idx));

        process_hamiltonian_idx++;
    }
}

void CudaMultiGpuHamiltonianNaive::act_even(CudaEvenState& output,
                                            CudaEvenState& state) {
    cuda_manager.release_all();

    Timer timer;

    // Copy the input state to all other devices
    for (int i = 1; i < num_devices(); i++) {
        input_states.at(i)->set(state);
        // input_states.at(i).set(state, stream);
    }

    if (debug) timer.print_msec("Device memcpy: ");

    // checkCudaErrors(cudaStreamSynchronize(stream));

    timer.reset();

    // Have all Hamiltonians act
    int n = hamiltonians.size();
    CudaEvents act_events(n - 1);

    // Hamiltonian 0 is on device 0, so it acts directly on the given state.
    // It contains a single chunk so it can act asynchronously.
    hamiltonians.at(0)->act_even_single_chunk_async(output,
                                                    state,
                                                    act_events.events.at(0));

    // Hamiltonians 1,...,n-1 act on the copied states on devices 1,...,n-1.
    // Each of these contains a single chunk, so it acts asynchronously.
    for (int i = 1; i < n - 1; i++) {
        hamiltonians.at(i)->act_even_single_chunk_async(*output_states.at(i),
                                                        *input_states.at(i),
                                                        act_events.events.at(i));
    }

    // Act with the last Hamiltonian on device n. This blocks because this
    // Hamiltonian will generally have more than one chunk.
    hamiltonians.at(n-1)->act_even(*output_states.at(n-1),
                                   *input_states.at(n-1));

    // Wait for all the previous Hamiltonians
    act_events.synchronize_all();

    if (debug) timer.print_msec("All Hamiltonians acted: ");

    ///////////// Collect the outputs /////////////
    timer.reset();

    cuda_set_device(0);
    cudaStream_t collection_stream = cuda_manager.get_stream();
    CudaHandles handles = cuda_manager.get_handles();
    handles.set_stream(collection_stream);
 
    // Copy the outputs to device 0 and add them
    for (int i = 1; i < n; i++) {
        output_copy->set_async(*output_states.at(i), collection_stream);
        output.add(handles.cublas_handle, d_one->ptr, *output_copy);
    }

    checkCudaErrors(cudaStreamSynchronize(collection_stream));
    if (debug) timer.print_msec("Outputs collected: ");
}
