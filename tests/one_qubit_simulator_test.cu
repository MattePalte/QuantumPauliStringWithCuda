#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../src/one_qubit_simulator.h" // Include the header file

// Test function to check the initialization of the qubit
void testInitializeQubit() {
    Complex* d_state;
    cudaMalloc((void**)&d_state, 2 * sizeof(Complex));

    initializeQubit<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    Complex h_state[2];
    cudaMemcpy(h_state, d_state, 2 * sizeof(Complex), cudaMemcpyDeviceToHost);

    assert(cuCreal(h_state[0]) == 1.0 && cuCimag(h_state[0]) == 0.0);
    assert(cuCreal(h_state[1]) == 0.0 && cuCimag(h_state[1]) == 0.0);

    cudaFree(d_state);
    std::cout << "testInitializeQubit passed!" << std::endl;
}

// Test function to check the application of the Hadamard gate
void testApplyHadamard() {
    Complex* d_state;
    cudaMalloc((void**)&d_state, 2 * sizeof(Complex));

    initializeQubit<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    applyHadamard<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    Complex h_state[2];
    cudaMemcpy(h_state, d_state, 2 * sizeof(Complex), cudaMemcpyDeviceToHost);

    double sqrt2_inv = 1.0 / sqrt(2.0);
    assert(fabs(cuCreal(h_state[0]) - sqrt2_inv) < 1e-6);
    assert(fabs(cuCimag(h_state[0]) - 0.0) < 1e-6);
    assert(fabs(cuCreal(h_state[1]) - sqrt2_inv) < 1e-6);
    assert(fabs(cuCimag(h_state[1]) - 0.0) < 1e-6);

    cudaFree(d_state);
    std::cout << "testApplyHadamard passed!" << std::endl;
}

// Test that double application of X gate is identity
void testApplyX() {
    Complex* d_state;
    cudaMalloc((void**)&d_state, 2 * sizeof(Complex));

    initializeQubit<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    applyX<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    applyX<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    Complex h_state[2];
    cudaMemcpy(h_state, d_state, 2 * sizeof(Complex), cudaMemcpyDeviceToHost);

    assert(cuCreal(h_state[0]) == 1.0 && cuCimag(h_state[0]) == 0.0);
    assert(cuCreal(h_state[1]) == 0.0 && cuCimag(h_state[1]) == 0.0);

    cudaFree(d_state);
    std::cout << "testApplyX passed!" << std::endl;
}


int main() {
    testInitializeQubit();
    testApplyHadamard();
    testApplyX();
    return 0;
}