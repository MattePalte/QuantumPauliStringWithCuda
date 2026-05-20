#include <cassert>
#include <cmath>
#include <iostream>

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "../src/one_qubit_simulator.h"

static const double EPS = 1e-6;

const Complex HADAMARD[2][2] = {
    {make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0)},
    {make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0)}};

const Complex PAULI_X[2][2] = {
    {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0)},
    {make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)}};

void assertClose(Complex value, double expected_real, double expected_imag = 0.0) {
    assert(fabs(cuCreal(value) - expected_real) < EPS);
    assert(fabs(cuCimag(value) - expected_imag) < EPS);
}

void testSingleQubitRegisterFlow() {
    QubitRegister reg;
    allocateQubits(&reg, 1);

    Complex state[2];
    getStateVector(&reg, state);
    assertClose(state[0], 1.0);
    assertClose(state[1], 0.0);

    applySingleQubitGate(&reg, 0, PAULI_X);
    getStateVector(&reg, state);
    assertClose(state[0], 0.0);
    assertClose(state[1], 1.0);

    applySingleQubitGate(&reg, 0, PAULI_X);
    getStateVector(&reg, state);
    assertClose(state[0], 1.0);
    assertClose(state[1], 0.0);

    applySingleQubitGate(&reg, 0, HADAMARD);
    getStateVector(&reg, state);
    double sqrt2_inv = 1.0 / sqrt(2.0);
    assertClose(state[0], sqrt2_inv);
    assertClose(state[1], sqrt2_inv);

    int first = measureQubit(&reg, 0);
    assert(first == 0 || first == 1);
    int second = measureQubit(&reg, 0);
    assert(first == second);

    freeQubits(&reg);
    std::cout << "testSingleQubitRegisterFlow passed!" << std::endl;
}

void testLegacyOneQubitKernelsStillWork() {
    Complex* d_state;
    cudaMalloc((void**)&d_state, 2 * sizeof(Complex));

    initializeQubit<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    applyHadamard<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    applyX<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    Complex h_state[2];
    cudaMemcpy(h_state, d_state, 2 * sizeof(Complex), cudaMemcpyDeviceToHost);

    double sqrt2_inv = 1.0 / sqrt(2.0);
    assertClose(h_state[0], sqrt2_inv);
    assertClose(h_state[1], sqrt2_inv);

    cudaFree(d_state);
    std::cout << "testLegacyOneQubitKernelsStillWork passed!" << std::endl;
}

void testBellState() {
    QubitRegister reg;
    allocateQubits(&reg, 2);

    applySingleQubitGate(&reg, 0, HADAMARD);
    applyCNOT(&reg, 0, 1);

    Complex state[4];
    getStateVector(&reg, state);

    double sqrt2_inv = 1.0 / sqrt(2.0);
    assertClose(state[0], sqrt2_inv);
    assertClose(state[1], 0.0);
    assertClose(state[2], 0.0);
    assertClose(state[3], sqrt2_inv);

    freeQubits(&reg);

    int matching_measurements = 0;
    const int trials = 100;
    for (int i = 0; i < trials; ++i) {
        QubitRegister shot;
        allocateQubits(&shot, 2);
        applySingleQubitGate(&shot, 0, HADAMARD);
        applyCNOT(&shot, 0, 1);

        int m0 = measureQubit(&shot, 0);
        int m1 = measureQubit(&shot, 1);
        if (m0 == m1) {
            ++matching_measurements;
        }

        freeQubits(&shot);
    }

    assert(matching_measurements == trials);
    std::cout << "testBellState passed!" << std::endl;
}

int main() {
    testSingleQubitRegisterFlow();
    testLegacyOneQubitKernelsStillWork();
    testBellState();
    return 0;
}
