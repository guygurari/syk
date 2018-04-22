#include <iostream>
#include "CudaUtils.h"

using namespace std;

int cuda_get_num_devices() {
    int num_devices;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));
    return num_devices;
}

size_t cuda_get_device_memory(int device_id) {
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
    return prop.totalGlobalMem;
}

void cuda_set_device(int device_id) {
    checkCudaErrors(cudaSetDevice(device_id));
}

int cuda_get_device() {
    int dev_id;
    checkCudaErrors(cudaGetDevice(&dev_id));
    return dev_id;
}

void cuda_print_device_properties() {
    cout << "Found " << cuda_get_num_devices() << " CUDA devices.\n";
    
    for (int dev_id = 0; dev_id < cuda_get_num_devices(); dev_id++) {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, dev_id));

        cout << "\nCUDA Device Number: " << dev_id << endl;
        cout << "  Device name: " << prop.name << endl;
        cout << "  Compute compatibility: " << prop.major
             << "." << prop.minor << endl;
        cout << "  Global Memory (GB): "
             << prop.totalGlobalMem / 1e9 << endl;
        cout << "  Memory Clock Rate (KHz): "
             << prop.memoryClockRate << endl;
        cout << "  Memory Bus Width (bits): "
             << prop.memoryBusWidth << endl;
        cout << "  Peak Memory Bandwidth (GB/s): "
             << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6
             << endl;
        cout << "  Async engine count: "
             << prop.asyncEngineCount << endl;
    }

    cout << endl;
}

cublasHandle_t cublas_init(int argc, char** argv) {
    cublasHandle_t handle;

    cudaDeviceProp device_prop;
    int dev_id = cuda_get_device();
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

    if ((device_prop.major << 4) + device_prop.minor < 0x35) {
        cerr << "Requires Compute Capability of SM 3.5 or higher" << endl;
        exit(1);
    }

    checkCublasErrors(cublasCreate(&handle));

    // Scalars should point to device memory
    checkCublasErrors(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    return handle;
}

void cublas_destroy(cublasHandle_t handle) {
    checkCublasErrors(cublasDestroy(handle));
}

cudaStream_t get_cublas_stream(cublasHandle_t handle) {
    cudaStream_t stream;
    checkCublasErrors(cublasGetStream(handle, &stream));
    return stream;
}

cusparseHandle_t cusparse_init() {
    cusparseHandle_t handle;
    checkCusparseErrors(cusparseCreate(&handle));

    // Scalars (alpha, beta in mm, etc.) should point to the device
    checkCusparseErrors(
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));

    return handle;
}

void cusparse_destroy(cusparseHandle_t handle) {
    checkCusparseErrors(cusparseDestroy(handle));
}

void* d_alloc_copy(const void* h_mem, size_t byte_size) {
    void* d_mem;
    checkCudaErrors(cudaMalloc(&d_mem, byte_size));
    checkCudaErrors(cudaMemcpy(d_mem,
                               h_mem,
                               byte_size,
                               cudaMemcpyHostToDevice));
    return d_mem;
}

void copy_matrix_host_to_host(cucpx* h_mat, const Mat& mat) {
    // column-major order
    memcpy(h_mat, mat.data(), sizeof(cucpx) * mat.rows() * mat.cols());
}

Mat copy_matrix_device_to_host(cucpx* d_mat, int rows, int cols) {
    Mat mat(rows, cols);

    // Both cuBLAS and eigen use column-major ordering.
    // The complex number layout is also the same (real, imag),
    // guaranteed by the C++ standard.
    // So we can just copy the data.
    checkCudaErrors(cudaMemcpy(mat.data(),
                               d_mat,
                               sizeof(cucpx) * rows * cols,
                               cudaMemcpyDeviceToHost));

    return mat;
}

void copy_matrix_host_to_device(cucpx* d_mat, const Mat& h_mat) {
    // Both use column-major ordering, so we can just copy
    checkCudaErrors(cudaMemcpy(d_mat,
                               h_mat.data(),
                               sizeof(cucpx) * h_mat.rows() * h_mat.cols(),
                               cudaMemcpyHostToDevice));
}

cpx d_read_scalar(cucpx* d_scalar) {
    cucpx x;
    checkCudaErrors(cudaMemcpy(&x, d_scalar, sizeof(cucpx),
                               cudaMemcpyDeviceToHost));
    return cpx(x.x, x.y);
}

cucpx* d_alloc_copy_matrix(const Mat& mat, size_t* size_output) {
    int alloc_size = sizeof(cucpx) * mat.rows() * mat.cols();

    if (size_output != 0) {
        *size_output = alloc_size;
    }
    
    cucpx* h_mat = (cucpx*) malloc(alloc_size);

    if (h_mat == 0) {
        cerr << "Error allocating " << alloc_size << " bytes" << endl;
        exit(1);
    }

    copy_matrix_host_to_host(h_mat, mat);
    
    cucpx* d_mat = (cucpx*) d_alloc_copy(h_mat, alloc_size);
    free(h_mat);
    return d_mat;
}

cucpx* d_alloc(size_t alloc_size) {
    cucpx* d_mem;
    checkCudaErrors(cudaMalloc((void**) &d_mem, alloc_size));
    return d_mem;
}

cucpx* d_alloc_ones_vector(size_t size) {
    int alloc_size = sizeof(cucpx) * size;
    cucpx* d_ones = d_alloc(alloc_size);
    d_set_ones_vector(d_ones, size);
    return d_ones;
}

void d_set_ones_vector(cucpx* d_ones, size_t size) {
    int alloc_size = sizeof(cucpx) * size;
    cucpx* h_ones = (cucpx*) malloc(alloc_size);

    // Can't use cublasSetVector easily, because can't set incx=0.
    cucpx one = make_cuDoubleComplex(1., 0.);

    for (int i = 0; i < size; i++) {
        h_ones[i] = one;
    }

    checkCudaErrors(cudaMemcpy(d_ones, h_ones,
                               alloc_size, cudaMemcpyHostToDevice));
    free(h_ones);
}

cucpx* d_alloc_scalar(cucpx x) {
    cucpx* d_x = d_alloc(sizeof(cucpx));
    checkCudaErrors(cudaMemcpy(d_x,
                               &x,
                               sizeof(cucpx),
                               cudaMemcpyHostToDevice));
    return d_x;
}

cucpx* d_alloc_scalar(double real, double imag) {
    cucpx x = make_cuDoubleComplex(real, imag);
    return d_alloc_scalar(x);
}

void d_free(void* d_ptr) {
    checkCudaErrors(cudaFree(d_ptr));
}

void d_print_vector(cucpx* d_vec, size_t size) {
    Vec h_vec(size);
    checkCudaErrors(cudaMemcpy(h_vec.data(), d_vec,
                               sizeof(cucpx) * size,
                               cudaMemcpyDeviceToHost));
    cout << h_vec << "\n";
}

void d_print_matrix(cucpx* d_mat, int rows, int cols) {
    Mat h_mat(rows, cols);
    checkCudaErrors(cudaMemcpy(h_mat.data(), d_mat,
                               sizeof(cucpx) * rows * cols,
                               cudaMemcpyDeviceToHost));
    cout << h_mat << "\n";
}

cudaStream_t d_create_stream() {
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return stream;
}

void d_destroy_stream(cudaStream_t stream) {
    checkCudaErrors(cudaStreamDestroy(stream));
}

void add_vector_times_scalar(
    cublasHandle_t handle,
    cucpx* d_y,
    cucpx* alpha,
    cucpx* d_x,
    int n) {

    checkCublasErrors(cublasZaxpy(handle,
                                  n,
                                  alpha,
                                  d_x, 1,
                                  d_y, 1));
}

CudaScalar::CudaScalar(cucpx x) {
    ptr = d_alloc_scalar(x);
}

CudaScalar::CudaScalar(cpx x) {
    ptr = d_alloc_scalar(x.real(), x.imag());
}

CudaScalar::CudaScalar(double real, double imag) {
    ptr = d_alloc_scalar(real, imag);
}

CudaScalar::~CudaScalar() {
    d_free(ptr);
}

CudaAllocator::CudaAllocator(size_t _capacity) : alignment(16) {
    device_id = cuda_get_device();

    allocated = 0;
    capacity = _capacity;
    cout << "CudaAllocator: Allocating " << capacity << " bytes on device " << device_id << endl;
    checkCudaErrors(cudaMalloc(&d_ptr, capacity));
    checkCudaErrors(cudaMemset(d_ptr, 0, capacity));
}

CudaAllocator::~CudaAllocator() {
    cuda_set_device(device_id);
    checkCudaErrors(cudaFree(d_ptr));
}

size_t CudaAllocator::available() {
    return capacity - allocated;
}

void* CudaAllocator::alloc(size_t size) {
    void* d_new = (void*) ((char*) d_ptr + allocated);
    size_t new_allocated = allocated + size;

    if (new_allocated % alignment != 0) {
        new_allocated -= (new_allocated % alignment);
        new_allocated += alignment;
    }

    if (new_allocated > capacity) {
        cerr << "CudaAllocator " << this << ": Cannot allocate " << size
             << " bytes (allocated=" << allocated
             << ", capacity=" << capacity << ")" << endl;
        exit(1);
    }

    allocated = new_allocated;
    assert(allocated % alignment == 0);
    // cout << "CudaAllocator " << this << ": allocated=" << allocated
    //      << "  capacity=" << capacity << endl;

    return d_new;
}

void CudaAllocator::free_all() {
    allocated = 0;
}

size_t sp_values_alloc_size(int nonzero_elems) {
    return sizeof(cucpx) * nonzero_elems;
}

size_t sp_row_ptr_alloc_size(int rows) {
    return sizeof(int) * (rows + 1);
}

size_t sp_col_ind_alloc_size(int nonzero_elems) {
    return sizeof(int) * nonzero_elems;
}

size_t sp_values_alloc_size(const RowMajorSpMat& sp) {
    return sp_values_alloc_size(sp.nonZeros());
}
    
size_t sp_row_ptr_alloc_size(const RowMajorSpMat& sp) {
    return sp_row_ptr_alloc_size(sp.rows());
}
    
size_t sp_col_ind_alloc_size(const RowMajorSpMat& sp) {
    return sp_col_ind_alloc_size(sp.nonZeros());
}
    

CudaSparseMatrix::CudaSparseMatrix(const RowMajorSpMat& sp,
                                   CudaAllocator* allocator) {
    init(sp, allocator);
}

CudaSparseMatrix::CudaSparseMatrix(const RowMajorSpMat& sp,
                                   CudaAllocator* allocator,
                                   cudaStream_t stream) {
    init_async(sp, allocator, stream);
}

// Our sparse matrices have exact zeros, so we don't care about
// the tolerance of identifying zeros here.
CudaSparseMatrix::CudaSparseMatrix(const Mat& mat) {
    init(RowMajorSpMat(mat.sparseView()), 0);
}

size_t CudaSparseMatrix::get_alloc_size(
    int rows, int nonzero_elems) {

    size_t values_alloc_size = sp_values_alloc_size(nonzero_elems);
    size_t row_ptr_alloc_size = sp_row_ptr_alloc_size(rows);
    size_t col_ind_alloc_size = sp_col_ind_alloc_size(nonzero_elems);

    return values_alloc_size
        + row_ptr_alloc_size
        + col_ind_alloc_size;
}

void CudaSparseMatrix::alloc(const RowMajorSpMat& sp,
                             CudaAllocator* allocator) {
    assert(sp.isCompressed());

    rows = sp.rows();
    cols = sp.cols();
    nz_elems = sp.nonZeros();

    values_alloc_size = sp_values_alloc_size(sp);
    row_ptr_alloc_size = sp_row_ptr_alloc_size(sp);
    col_ind_alloc_size = sp_col_ind_alloc_size(sp);

    d_alloc_size =
        values_alloc_size
        + row_ptr_alloc_size
        + col_ind_alloc_size; 

    assert(d_alloc_size == get_alloc_size(rows, nz_elems));

    if (allocator == 0) {
        d_values = d_alloc(values_alloc_size);
        d_row_ptr = (int*) d_alloc(row_ptr_alloc_size);
        d_col_ind = (int*) d_alloc(col_ind_alloc_size);
        custom_allocator = false;
    }
    else {
        d_values = (cucpx*) allocator->alloc(values_alloc_size);
        d_row_ptr = (int*) allocator->alloc(row_ptr_alloc_size);
        d_col_ind = (int*) allocator->alloc(col_ind_alloc_size);
        custom_allocator = true;
    }
}

void CudaSparseMatrix::init(const RowMajorSpMat& sp,
                            CudaAllocator* allocator) {
    alloc(sp, allocator);

    checkCudaErrors(cudaMemcpy(d_values, sp.valuePtr(),
                               values_alloc_size,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_row_ptr, sp.outerIndexPtr(),
                               row_ptr_alloc_size,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_col_ind, sp.innerIndexPtr(),
                               col_ind_alloc_size,
                               cudaMemcpyHostToDevice));
}

void CudaSparseMatrix::init_async(const RowMajorSpMat& sp,
                                  CudaAllocator* allocator,
                                  cudaStream_t stream) {
    alloc(sp, allocator);

    checkCudaErrors(cudaMemcpyAsync(d_values, sp.valuePtr(),
                                    values_alloc_size,
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMemcpyAsync(d_row_ptr, sp.outerIndexPtr(),
                                    row_ptr_alloc_size,
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMemcpyAsync(d_col_ind, sp.innerIndexPtr(),
                                    col_ind_alloc_size,
                                    cudaMemcpyHostToDevice, stream));
}

CudaSparseMatrix::~CudaSparseMatrix() {
    if (!custom_allocator) {
        d_free(d_values);
        d_free(d_row_ptr);
        d_free(d_col_ind);
    }
}

CudaHandles::CudaHandles() {
    cublas_handle = cublas_init();
    cusparse_handle = cusparse_init();
}

void CudaHandles::set_stream(cudaStream_t stream) {
    checkCublasErrors(cublasSetStream(cublas_handle, stream));
    checkCusparseErrors(cusparseSetStream(cusparse_handle, stream));
}

void CudaHandles::destroy() {
    cublas_destroy(cublas_handle);
    cusparse_destroy(cusparse_handle);
}

CudaResourceManager::CudaResourceManager() {
    boost::lock_guard<boost::recursive_mutex> lock(mutex);
    next_stream_idx = 0;
    next_handle_idx = 0;
}

CudaResourceManager::~CudaResourceManager() {
    boost::lock_guard<boost::recursive_mutex> lock(mutex);

    for (int i = 0; i < streams.size(); i++) {
        d_destroy_stream(streams[i]);
    }

    for (int i = 0; i < handles.size(); i++) {
        handles.at(i).destroy();
    }
}

cudaStream_t CudaResourceManager::get_stream() {
    boost::lock_guard<boost::recursive_mutex> lock(mutex);

    if (next_stream_idx >= streams.size()) {
        streams.push_back(d_create_stream());
    }

    return streams.at(next_stream_idx++);
}

vector<cudaStream_t> CudaResourceManager::get_streams(int n) {
    boost::lock_guard<boost::recursive_mutex> lock(mutex);
    vector<cudaStream_t> result;

    while (result.size() < n) {
        result.push_back(get_stream());
    }

    return result;
}

CudaHandles CudaResourceManager::get_handles() {
    boost::lock_guard<boost::recursive_mutex> lock(mutex);

    if (next_handle_idx >= handles.size()) {
        handles.push_back(CudaHandles());
    }

    return handles.at(next_handle_idx++);
}

void CudaResourceManager::release_all() {
    boost::lock_guard<boost::recursive_mutex> lock(mutex);
    next_stream_idx = 0;
    next_handle_idx = 0;
}

CudaEvent::CudaEvent() {
    checkCudaErrors(cudaEventCreate(&event));
}

CudaEvent::~CudaEvent() {
    checkCudaErrors(cudaEventDestroy(event));
}

void CudaEvent::record(cudaStream_t stream) {
    checkCudaErrors(cudaEventRecord(event, stream));
}

void CudaEvent::wait(cudaStream_t stream, int flags) {
    checkCudaErrors(cudaStreamWaitEvent(stream, event, flags));
}

CudaEvents::CudaEvents(int n) {
    for (int i = 0; i < n; i++) {
        cudaEvent_t event;
        checkCudaErrors(cudaEventCreate(&event));
        events.push_back(event);
    }
}

CudaEvents::~CudaEvents() {
    for (int i = 0; i < events.size(); i++) {
        checkCudaErrors(cudaEventDestroy(events.at(i)));
    }
}

void CudaEvents::record(int i, cudaStream_t stream) {
    checkCudaErrors(cudaEventRecord(events.at(i), stream));
}

void CudaEvents::wait_all(cudaStream_t stream, int flags) {
    for (int i = 0; i < events.size(); i++) {
        checkCudaErrors(cudaStreamWaitEvent(stream, events.at(i), flags));
    }
}

void CudaEvents::synchronize_all() {
    for (int i = 0; i < events.size(); i++) {
        checkCudaErrors(cudaEventSynchronize(events.at(i)));
    }
}
