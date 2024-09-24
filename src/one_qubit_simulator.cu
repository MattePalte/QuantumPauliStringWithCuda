/**
 * @file one_qubit_simulator.cu
 * @brief This file contains the implementation of a one-qubit quantum simulator.
 *
 * Detailed description:
 * This CUDA program simulates the behavior of a single qubit. It includes functions
 * for initializing the qubit state, applying quantum gates (such as the Hadamard gate),
 * and measuring the qubit state. The simulation leverages the parallel processing
 * capabilities of CUDA to efficiently perform quantum state transformations.
 *
 * Usage:
 * - Compile the file using nvcc:
 *   nvcc -o one_qubit_simulator one_qubit_simulator.cu
 * - Run the compiled executable:
 *   ./one_qubit_simulator
 *
 * Implementation details:
 * - the statevector is of size 2 and it has to be normalized, it has complex values
 * - the Hadamard gate is applied to the qubit state
 * - the program prints the final state of the qubit after the Hadamard gate is applied
 * - use cuBLAS to implement the gate operations
 * - support five gates: X, Y, Z, H, and I
 *
 * Author: Matteo Paltenghi
 * Date: 2024
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>

// Define complex number type for statevector
typedef cuDoubleComplex Complex;

// CUDA Kernel to initialize the qubit state
__global__ void initializeQubit(Complex* state) {
    // |0> state is represented as (1, 0) and |1> as (0, 0)
    state[0] = make_cuDoubleComplex(1.0, 0.0);  // |0> state
    state[1] = make_cuDoubleComplex(0.0, 0.0);  // |1> state
}

// CUDA Kernel to apply Hadamard gate
__global__ void applyHadamard(Complex* state) {
    // Copy initial state
    Complex state0 = state[0];
    Complex state1 = state[1];

    // Apply Hadamard transformation
    // H = 1/sqrt(2) * [1 1; 1 -1]
    double norm = 1.0 / sqrt(2.0);

    state[0] = cuCadd(cuCmul(make_cuDoubleComplex(norm, 0), state0),
                      cuCmul(make_cuDoubleComplex(norm, 0), state1));

    state[1] = cuCadd(cuCmul(make_cuDoubleComplex(norm, 0), state0),
                      cuCmul(make_cuDoubleComplex(-norm, 0), state1));
}

// CUDA Kernel to apply X gate
__global__ void applyX(Complex* state) {
    // Copy initial state
    Complex state0 = state[0];
    Complex state1 = state[1];

    // Apply X transformation
    // X = [0 1; 1 0]
    state[0] = state1;
    state[1] = state0;
}


// Host function to print the statevector
void printState(Complex* state) {
    Complex h_state[2];
    cudaMemcpy(h_state, state, 2 * sizeof(Complex), cudaMemcpyDeviceToHost);

    std::cout << "Qubit state after Hadamard gate:" << std::endl;
    std::cout << "State |0>: " << cuCreal(h_state[0]) << " + " << cuCimag(h_state[0]) << "i" << std::endl;
    std::cout << "State |1>: " << cuCreal(h_state[1]) << " + " << cuCimag(h_state[1]) << "i" << std::endl;
}

// int main() {
//     // Allocate memory for the qubit state on the GPU
//     Complex* d_state;
//     cudaMalloc((void**)&d_state, 2 * sizeof(Complex));

//     // Initialize qubit to |0> state
//     initializeQubit<<<1, 1>>>(d_state);
//     cudaDeviceSynchronize();

//     // Apply Hadamard gate to the qubit
//     applyHadamard<<<1, 1>>>(d_state);
//     applyHadamard<<<1, 1>>>(d_state);
//     cudaDeviceSynchronize();

//     // Print the final qubit state after applying the Hadamard gate
//     printState(d_state);

//     // Free the GPU memory
//     cudaFree(d_state);

//     return 0;
// }
