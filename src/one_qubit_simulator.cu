/**
 * @file one_qubit_simulator.cu
 * @brief Quantum simulator kernels and host API.
 */

#include "one_qubit_simulator.h"

#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>

static constexpr double PROBABILITY_EPSILON = 1e-15;

__host__ __device__ static inline int stateVectorSize(int num_qubits) {
    return 1 << num_qubits;
}

__global__ void initializeQubit(Complex* state) {
    state[0] = make_cuDoubleComplex(1.0, 0.0);
    state[1] = make_cuDoubleComplex(0.0, 0.0);
}

__global__ void applyHadamard(Complex* state) {
    Complex state0 = state[0];
    Complex state1 = state[1];
    double norm = 1.0 / sqrt(2.0);

    state[0] = cuCadd(cuCmul(make_cuDoubleComplex(norm, 0.0), state0),
                      cuCmul(make_cuDoubleComplex(norm, 0.0), state1));
    state[1] = cuCadd(cuCmul(make_cuDoubleComplex(norm, 0.0), state0),
                      cuCmul(make_cuDoubleComplex(-norm, 0.0), state1));
}

__global__ void applyX(Complex* state) {
    Complex state0 = state[0];
    Complex state1 = state[1];
    state[0] = state1;
    state[1] = state0;
}

__global__ void singleQubitGateKernel(Complex* state,
                                      int num_qubits,
                                      int target,
                                      Complex g00,
                                      Complex g01,
                                      Complex g10,
                                      Complex g11) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = stateVectorSize(num_qubits) >> 1;
    if (idx >= pairs) {
        return;
    }

    int lower_mask = (1 << target) - 1;
    int low_bits = idx & lower_mask;
    int high_bits = idx >> target;

    int i0 = (high_bits << (target + 1)) | low_bits;
    int i1 = i0 | (1 << target);

    Complex v0 = state[i0];
    Complex v1 = state[i1];

    state[i0] = cuCadd(cuCmul(g00, v0), cuCmul(g01, v1));
    state[i1] = cuCadd(cuCmul(g10, v0), cuCmul(g11, v1));
}

__global__ void cnotKernel(Complex* state, int num_qubits, int control, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = stateVectorSize(num_qubits);
    if (idx >= size) {
        return;
    }

    int control_bit = (idx >> control) & 1;
    int target_bit = (idx >> target) & 1;
    if (control_bit == 1 && target_bit == 0) {
        int partner = idx | (1 << target);
        if (idx < partner) {
            Complex tmp = state[idx];
            state[idx] = state[partner];
            state[partner] = tmp;
        }
    }
}

void allocateQubits(QubitRegister* reg, int num_qubits) {
    reg->num_qubits = num_qubits;
    int size = stateVectorSize(num_qubits);
    cudaMalloc((void**)&reg->d_state, size * sizeof(Complex));
    cudaMemset(reg->d_state, 0, size * sizeof(Complex));

    Complex one = make_cuDoubleComplex(1.0, 0.0);
    cudaMemcpy(reg->d_state, &one, sizeof(Complex), cudaMemcpyHostToDevice);
}

void freeQubits(QubitRegister* reg) {
    cudaFree(reg->d_state);
    reg->d_state = nullptr;
    reg->num_qubits = 0;
}

void applySingleQubitGate(QubitRegister* reg, int target_qubit, const Complex gate[2][2]) {
    int pairs = stateVectorSize(reg->num_qubits) >> 1;
    int threads = 128;
    int blocks = (pairs + threads - 1) / threads;

    singleQubitGateKernel<<<blocks, threads>>>(reg->d_state,
                                               reg->num_qubits,
                                               target_qubit,
                                               gate[0][0],
                                               gate[0][1],
                                               gate[1][0],
                                               gate[1][1]);
    cudaDeviceSynchronize();
}

void applyCNOT(QubitRegister* reg, int control_qubit, int target_qubit) {
    int size = stateVectorSize(reg->num_qubits);
    int threads = 128;
    int blocks = (size + threads - 1) / threads;

    cnotKernel<<<blocks, threads>>>(reg->d_state, reg->num_qubits, control_qubit, target_qubit);
    cudaDeviceSynchronize();
}

void getStateVector(const QubitRegister* reg, Complex* h_state) {
    int size = stateVectorSize(reg->num_qubits);
    cudaMemcpy(h_state, reg->d_state, size * sizeof(Complex), cudaMemcpyDeviceToHost);
}

int measureQubit(QubitRegister* reg, int target_qubit) {
    int size = stateVectorSize(reg->num_qubits);
    std::vector<Complex> h_state(size);
    getStateVector(reg, h_state.data());

    double p0 = 0.0;
    for (int i = 0; i < size; ++i) {
        if (((i >> target_qubit) & 1) == 0) {
            double real = cuCreal(h_state[i]);
            double imag = cuCimag(h_state[i]);
            p0 += real * real + imag * imag;
        }
    }

    // Clamp to [0, 1] to absorb floating-point accumulation noise.
    if (p0 < 0.0) {
        p0 = 0.0;
    }
    if (p0 > 1.0) {
        p0 = 1.0;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int result = (dist(rng) < p0) ? 0 : 1;

    double kept_probability = (result == 0) ? p0 : (1.0 - p0);
    if (kept_probability < PROBABILITY_EPSILON) {
        for (int i = 0; i < size; ++i) {
            h_state[i] = make_cuDoubleComplex(0.0, 0.0);
        }
        for (int i = 0; i < size; ++i) {
            if (((i >> target_qubit) & 1) == result) {
                h_state[i] = make_cuDoubleComplex(1.0, 0.0);
                break;
            }
        }
    } else {
        double norm = sqrt(kept_probability);
        for (int i = 0; i < size; ++i) {
            if (((i >> target_qubit) & 1) == result) {
                h_state[i] = make_cuDoubleComplex(cuCreal(h_state[i]) / norm,
                                                  cuCimag(h_state[i]) / norm);
            } else {
                h_state[i] = make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }

    cudaMemcpy(reg->d_state, h_state.data(), size * sizeof(Complex), cudaMemcpyHostToDevice);
    return result;
}
