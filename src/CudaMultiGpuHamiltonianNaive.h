#ifndef __CUDA_MULTI_GPU_HAMILTONIAN_NAIVE_H__
#define __CUDA_MULTI_GPU_HAMILTONIAN_NAIVE_H__

#include "defs.h"
#include "DisorderParameter.h"
#include "FactorizedHamiltonian.h"
#include "CudaUtils.h"
#include "CudaState.h"
#include "CudaHamiltonian.h"

//
// This implementation fills up n-1 GPUs, and uses chunks to handle the
// n'th GPU. This is host memory efficient because the host can free the memory
// of the first n-1 GPUs. But typically the last GPU has more work to do so the
// first n-1 GPUs have idle time. For N=44 with 2 GPUs this makes times longer by
// about 50%.
//
class CudaMultiGpuHamiltonianNaive
    : public HamiltonianTermProcessor,
      public CudaHamiltonianInterface {
public:
    CudaMultiGpuHamiltonianNaive(
        FactorizedSpace& _space,
        MajoranaKitaevDisorderParameter& Jtensor,
        vector<size_t>& available_memory_per_device,
        bool mock_hamiltonian = false,
        bool _debug = false);

    // Gives each CUDA device the same amount of memory
    CudaMultiGpuHamiltonianNaive(
        FactorizedSpace& _space,
        MajoranaKitaevDisorderParameter& Jtensor,
        size_t _available_memory_per_device,
        bool mock_hamiltonian = false,
        bool _debug = false);

    virtual ~CudaMultiGpuHamiltonianNaive();

    virtual size_t total_h_alloc_size();
    virtual size_t total_d_alloc_size();
    virtual size_t predicted_total_d_alloc_size();
    virtual void print_memory_allocation();

    virtual void process(int left_idx, FactorizedOperatorPair& ops_pair);

    // Act with the Hamiltonian on a state with even charge,
    // writing the result to output. The result is written to device
    // memory, and is not copied to the host by default.
    // Both output and state should be allocated on device id 0.
    virtual void act_even(CudaEvenState& output, CudaEvenState& state);

private:
    typedef boost::shared_ptr<CudaEvenState> CudaEvenState_ptr;
    typedef boost::shared_ptr<CudaHamiltonian> CudaHamiltonian_ptr;

    FactorizedSpace space;
    vector<size_t> available_memory_per_device;
    CudaResourceManager cuda_manager;
    boost::shared_ptr<CudaScalar> d_one;
    bool debug;

    // Used by process()
    int process_hamiltonian_idx;

    // Used to send the input state to all the devices, and to collect their
    // outputs
    map<int, CudaEvenState_ptr> input_states;
    map<int, CudaEvenState_ptr> output_states;
    CudaEvenState_ptr output_copy;

    vector<CudaHamiltonian_ptr> hamiltonians;

    void init(
        FactorizedSpace& _space,
        MajoranaKitaevDisorderParameter& Jtensor,
        vector<size_t>& available_memory_per_device,
        bool mock_hamiltonian,
        bool _debug);

    int num_devices();
    void init_op_processing();
    void prepare_op_chunks();
    size_t get_memory_per_chunk(int device_id);
};

#endif
