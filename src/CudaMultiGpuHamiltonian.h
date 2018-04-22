#ifndef __CUDA_MULTI_GPU_HAMILTONIAN_H__
#define __CUDA_MULTI_GPU_HAMILTONIAN_H__

#include "defs.h"
#include "DisorderParameter.h"
#include "FactorizedHamiltonian.h"
#include "CudaUtils.h"
#include "CudaState.h"
#include "CudaHamiltonian.h"

//
// This implementation splits the operators more or less evenly among the GPUs,
// and runs one chunk on each GPU in parallel. This way almost no cycles are wasted.
// The tradeoff is that the whole Hamiltonian has to be stored in host memory
// so the chunks can be rotated on the devices.
// 
class CudaMultiGpuHamiltonian
    : public HamiltonianTermProcessor,
      public CudaHamiltonianInterface {
public:
    CudaMultiGpuHamiltonian(
        FactorizedSpace& _space,
        MajoranaKitaevDisorderParameter& Jtensor,
        vector<size_t>& available_memory_per_device,
        bool mock_hamiltonian = false,
        bool _debug = false);

    // Gives each CUDA device the same amount of memory
    CudaMultiGpuHamiltonian(
        FactorizedSpace& _space,
        MajoranaKitaevDisorderParameter& Jtensor,
        size_t _available_memory_per_device,
        bool mock_hamiltonian = false,
        bool _debug = false);

    virtual ~CudaMultiGpuHamiltonian();

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

    void print_all_chunks();

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

    int get_next_device_id(int device_id);
    int num_devices();
    void init_op_processing();
    void prepare_op_chunks();
    size_t net_available_memory(int device_id);
    size_t get_memory_per_chunk(int device_id);
    void collect_device_outputs(CudaEvenState& output);
};

#endif // __CUDA_MULTI_GPU_HAMILTONIAN_H__
