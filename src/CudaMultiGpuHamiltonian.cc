#include "Timer.h"
#include "CudaMultiGpuHamiltonian.h"

CudaMultiGpuHamiltonian::CudaMultiGpuHamiltonian(
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

CudaMultiGpuHamiltonian::CudaMultiGpuHamiltonian(
    FactorizedSpace& _space,
    MajoranaKitaevDisorderParameter& Jtensor,
    vector<size_t>& _available_memory_per_device,
    bool mock_hamiltonian,
    bool _debug) {

    init(_space, Jtensor, _available_memory_per_device, mock_hamiltonian, _debug);
}

void CudaMultiGpuHamiltonian::init(
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

    // print_all_chunks();

    // for (int i = 0; i < hamiltonians.size(); i++) {
    //     cout << "Device " << i << ": predicted "
    //          << hamiltonians.at(i)->predicted_total_d_alloc_size() << " bytes in " 
    //          << hamiltonians.at(i)->op_chunks.size() << " chunks" << endl;
    // }
    // cout << endl;

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

int CudaMultiGpuHamiltonian::num_devices() {
    return available_memory_per_device.size();
}

void CudaMultiGpuHamiltonian::init_op_processing() {
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

int CudaMultiGpuHamiltonian::get_next_device_id(int device_id) {
    return (device_id + 1) % num_devices();
}

void CudaMultiGpuHamiltonian::print_all_chunks() {
    for (int i = 0; i < num_devices(); i++) {
        cout << "\nDevice " << i << " (avail-mem = "
             << available_memory_per_device.at(i) / 1e9
             << " GB, net-avail-mem = "
             << CudaHamiltonian::net_available_memory(
                 available_memory_per_device.at(i)) / 1e9
             << " GB)" << endl;

        hamiltonians.at(i)->print_chunks();
    }
}

void CudaMultiGpuHamiltonian::prepare_op_chunks() {
    // Each device has an associated CudaHamiltonian. We construct the bare
    // Hamiltonians, then add operators to them using round-robin.
    // This way the operators are roughly evenly distributed across the devices.

    for (int i = 0; i < num_devices(); i++) {
        hamiltonians.push_back(
            CudaHamiltonian_ptr(new CudaHamiltonian(space, i, debug)));
    }

    int device_id = 0;

    for (int left_idx = 0; left_idx <= q; left_idx++) {
        int num_ops = num_factorized_hamiltonian_operator_pairs(space, left_idx);
        int num_ops_accounted_for = 0;

        while (num_ops_accounted_for < num_ops) {
            size_t mem_per_chunk = get_memory_per_chunk(device_id);
            hamiltonians.at(device_id)->record_op_pair(left_idx, mem_per_chunk);
            device_id = get_next_device_id(device_id);
            num_ops_accounted_for++;
        }
    }

    // print_all_chunks();
}

size_t CudaMultiGpuHamiltonian::get_memory_per_chunk(int device_id) {
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
    
    return net_device_avail_memory / 2;
}

size_t CudaMultiGpuHamiltonian::total_h_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < hamiltonians.size(); i++) {
        size += hamiltonians.at(i)->total_h_alloc_size();
    }

    return size;
}

size_t CudaMultiGpuHamiltonian::total_d_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < hamiltonians.size(); i++) {
        size += hamiltonians.at(i)->total_d_alloc_size();
    }

    return size;
}


size_t CudaMultiGpuHamiltonian::predicted_total_d_alloc_size() {
    size_t size = 0;

    for (int i = 0; i < hamiltonians.size(); i++) {
        size += hamiltonians.at(i)->predicted_total_d_alloc_size();
    }

    return size;
}

void CudaMultiGpuHamiltonian::print_memory_allocation() {
    cout << "Hamiltonian memory allocated:\n"
         << total_h_alloc_size() / 1e9 << " GB host memory\n"
         << predicted_total_d_alloc_size() / 1e9
         << " GB device memory (estimated)\n"
         << predicted_total_d_alloc_size()
        / 1e9 / (size_t) hamiltonians.size()
         << " GB per device (estimated)\n\n";
}

CudaMultiGpuHamiltonian::~CudaMultiGpuHamiltonian() {
    hamiltonians.clear();
}

size_t CudaMultiGpuHamiltonian::net_available_memory(int device_id) {
    size_t mem = available_memory_per_device.at(device_id);

    // Subtract memory used for storing state input/output across devices
    if (device_id == 0) {
        mem -= space.state_alloc_size();
    }
    else {
        mem -= 2*space.state_alloc_size();
    }

    return mem;
}

void CudaMultiGpuHamiltonian::process(int left_idx, 
                                      FactorizedOperatorPair& ops_pair) {
    assert(process_hamiltonian_idx < hamiltonians.size());
    CudaHamiltonian& H = *hamiltonians.at(process_hamiltonian_idx);

    H.process(left_idx, ops_pair);

    if (H.done_processing_ops()) {
        if (debug) cout << "Done processing Hamiltonian "
                        << process_hamiltonian_idx
                        << " of " << hamiltonians.size() << endl;

        // Finalize this Hamiltonian (this allows it to free its host memory)
        H.finalize_construction_after_processing(
            net_available_memory(process_hamiltonian_idx));
    }

    process_hamiltonian_idx = get_next_device_id(process_hamiltonian_idx);
}

typedef boost::shared_ptr<ActionContext> ActionContext_ptr;

void CudaMultiGpuHamiltonian::act_even(CudaEvenState& output,
                                       CudaEvenState& state) {
    cuda_manager.release_all();
    Timer timer;

    // Copy the input state to all other devices
    for (int i = 1; i < num_devices(); i++) {
        input_states.at(i)->set(state);
    }

    if (debug) timer.print_msec("Device memcpy: ");
    timer.reset();

    // Have all Hamiltonians act
    int n = hamiltonians.size();

    map<int, CudaEvenState*> outputs;
    map<int, CudaEvenState*> inputs;

    outputs[0] = &output;
    inputs[0] = &state;

    for (int i = 1; i < n; i++) {
        outputs[i] = output_states.at(i).get();
        inputs[i] = input_states.at(i).get();
    }

    map<int, ActionContext_ptr> live_contexts;
    map<int, ActionContext_ptr> done_contexts;

    CudaResourceManager test_manager;

    for (int i = 0; i < hamiltonians.size(); i++) {
        live_contexts[i] = ActionContext_ptr(
            new ActionContext(hamiltonians.at(i)->prepare_to_act(*outputs.at(i))));
    }

    while (live_contexts.size() > 0) {
        map<int, ActionContext_ptr> live_contexts_copy(live_contexts);

        // Launch the kernels
        for (map<int, ActionContext_ptr>::iterator iter = live_contexts_copy.begin();
             iter != live_contexts_copy.end(); ++iter) {

            int i = iter->first;
            ActionContext_ptr context = iter->second;

            bool alive = hamiltonians.at(i)->act_even_chunk_async(
                *outputs.at(i), *inputs.at(i), *context);

            if (!alive) {
                live_contexts.erase(i);
                done_contexts[i] = context;
            }
        }

        // Sync the live hamiltonians
        for (map<int, ActionContext_ptr>::iterator iter = live_contexts.begin();
             iter != live_contexts.end(); ++iter) {

            int i = iter->first;
            ActionContext_ptr context = iter->second;
            hamiltonians.at(i)->sync_act_even(*context);
        }

    }

    assert(live_contexts.size() == 0);
    assert(done_contexts.size() == hamiltonians.size());

    if (debug) timer.print_msec("All Hamiltonians acted: ");
    collect_device_outputs(output);
}

void CudaMultiGpuHamiltonian::collect_device_outputs(CudaEvenState& output) {
    Timer timer;
    cuda_set_device(0);
    cudaStream_t collection_stream = cuda_manager.get_stream();
    CudaHandles handles = cuda_manager.get_handles();
    handles.set_stream(collection_stream);
 
    // Copy the outputs to device 0 and add them
    for (int i = 1; i < hamiltonians.size(); i++) {
        output_copy->set_async(*output_states.at(i), collection_stream);
        output.add(handles.cublas_handle, d_one->ptr, *output_copy);
    }

    checkCudaErrors(cudaStreamSynchronize(collection_stream));
    if (debug) timer.print_msec("Outputs collected: ");
}
