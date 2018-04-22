
__global__ void inverse_kernel(double* d_y, double* d_x) {
    double x = *d_x;
    *d_y = 1. / x;
}

void d_inverse(double* d_y, double* d_x) {
    inverse_kernel<<<1,1>>>(d_y, d_x);
}
