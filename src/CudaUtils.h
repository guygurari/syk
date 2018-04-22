#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <boost/thread.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "helper_cuda.h"

#include "defs.h"
#include "eigen_utils.h"

typedef cuDoubleComplex cucpx;

// Device management helpers
int cuda_get_num_devices();
size_t cuda_get_device_memory(int idx = 0);
void cuda_set_device(int device_id);
int cuda_get_device();
void cuda_print_device_properties();

cublasHandle_t cublas_init(int argc = 0, char** argv = 0);
void cublas_destroy(cublasHandle_t handle);
cudaStream_t get_cublas_stream(cublasHandle_t handle);

cusparseHandle_t cusparse_init();
void cusparse_destroy(cusparseHandle_t handle);

void* d_alloc_copy(const void* h_mem, size_t byte_size);
void copy_matrix_host_to_host(cucpx* h_mat, const Mat& mat);
Mat copy_matrix_device_to_host(cucpx* d_mat, int rows, int cols);
void copy_matrix_host_to_device(cucpx* d_mat, const Mat& h_mat);
cpx d_read_scalar(cucpx* d_scalar);
cucpx* d_alloc_copy_matrix(const Mat& mat, size_t* size_output=0);
cucpx* d_alloc(size_t alloc_size);
cucpx* d_alloc_ones_vector(size_t size);
void d_set_ones_vector(cucpx* d_ones, size_t size);
void d_free(void* d_ptr);

void d_print_vector(cucpx* d_vec, size_t size);
void d_print_matrix(cucpx* d_mat, int rows, int cols);

cudaStream_t d_create_stream();
void d_destroy_stream(cudaStream_t stream);

// sets y to 1/x
void d_inverse(double* d_y, double* d_x);

// y = y + alpha * x
//
// d_x, d_y are vectors of length n.
// alpha points to a scalar (either on host or on device).
// 
void add_vector_times_scalar(
    cublasHandle_t handle,
    cucpx* d_y,
    cucpx* alpha,
    cucpx* d_x,
    int n);

//
// A complex scalar on the device
//
class CudaScalar {
public:
    CudaScalar(cucpx x);
    CudaScalar(cpx x);
    CudaScalar(double real, double imag);
    ~CudaScalar();

    cucpx* ptr;
};

//
// A simple device memory allocator. Has a big chunk of memory,
// allocates it sequentially, and only supports freeing all the memory
// at once.
// 
class CudaAllocator {
public:
    CudaAllocator(size_t capacity);
    ~CudaAllocator();

    void* alloc(size_t size);

    // Free all allocated memory
    void free_all();

    // How much memory is still available
    size_t available();

    // Allocated addresses must be aligned in memory
    const size_t alignment;

private:
    int device_id;
    void* d_ptr;
    size_t capacity;
    size_t allocated;
};

// Storage sizes of vectors used to store row-major sparse matrices.
size_t sp_values_alloc_size(int nonzero_elems);
size_t sp_row_ptr_alloc_size(int rows);
size_t sp_col_ind_alloc_size(int nonzero_elems);

size_t sp_values_alloc_size(const RowMajorSpMat& sp);
size_t sp_row_ptr_alloc_size(const RowMajorSpMat& sp);
size_t sp_col_ind_alloc_size(const RowMajorSpMat& sp);

//
// A sparse matrix stored on the device in CSR format.
//
class CudaSparseMatrix {
public:
    CudaSparseMatrix(const RowMajorSpMat& sp, CudaAllocator* allocator=0);

    // Copies the matrix to the device asynchronously in stream
    CudaSparseMatrix(const RowMajorSpMat& sp,
                     CudaAllocator* allocator,
                     cudaStream_t stream);

    CudaSparseMatrix(const Mat& mat);
    ~CudaSparseMatrix();

    int rows; // number of rows
    int cols; // number of columns
    int nz_elems; // number of non-zero elements

    // non-zero values, of length nz_elems
    cucpx* d_values; 

    // indices into d_values, giving initial the first non-zero
    // element in each row, of length rows+1. the last element
    // is equal to nz_elems + d_row_ptr[0].
    int* d_row_ptr; 

    // column indices of the non-zero elements, of length nz_elems
    int* d_col_ind;

    // Total allocated size on the device
    int d_alloc_size;

    static size_t get_alloc_size(int rows, int nonzero_elems);

private:
    size_t values_alloc_size;
    size_t row_ptr_alloc_size;
    size_t col_ind_alloc_size;

    bool custom_allocator;

    void alloc(const RowMajorSpMat& sp, CudaAllocator* allocator);
    void init(const RowMajorSpMat& sp, CudaAllocator* allocator);
    void init_async(const RowMajorSpMat& sp,
                    CudaAllocator* allocator,
                    cudaStream_t stream);
};

struct CudaHandles {
    CudaHandles();
    void destroy();
    void set_stream(cudaStream_t stream);
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
};

//
// Manage CUDA handles and streams in a thread-safe way.
// Reuses resources to avoid CUDA memory leaks.
// 
class CudaResourceManager {
public:
    CudaResourceManager();
    ~CudaResourceManager();

    // Returns handles that have not been used since the last
    // call to release_all(). Each thread should use
    // different handles.
    CudaHandles get_handles();

    // Returns a new stream that has not been used since the
    // last call to release_all().
    cudaStream_t get_stream();

    vector<cudaStream_t> get_streams(int n);

    // Release all handles and streams, but keep them around
    // for reuse
    void release_all();

private:
    int next_stream_idx;
    vector<cudaStream_t> streams;

    int next_handle_idx;
    vector<CudaHandles> handles;

    boost::recursive_mutex mutex;
};

//
// A single event.
// 
class CudaEvent {
public:
    CudaEvent();
    ~CudaEvent();

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream, int flags=0);

    cudaEvent_t event;
};

//
// A set of events.
// 
class CudaEvents {
public:
    CudaEvents(int n);
    ~CudaEvents();
    
    void record(int event, cudaStream_t stream);
    void wait_all(cudaStream_t stream, int flags=0);
    void synchronize_all();

    vector<cudaEvent_t> events;
};

#endif // __CUDA_UTILS_H__
