#ifndef ONE_QUBIT_SIMULATOR_H
#define ONE_QUBIT_SIMULATOR_H

#include <cuComplex.h>

typedef cuDoubleComplex Complex;

typedef struct {
    Complex* d_state;
    int num_qubits;
} QubitRegister;

void allocateQubits(QubitRegister* reg, int num_qubits);
void freeQubits(QubitRegister* reg);
void applySingleQubitGate(QubitRegister* reg, int target_qubit, const Complex gate[2][2]);
void applyCNOT(QubitRegister* reg, int control_qubit, int target_qubit);
int measureQubit(QubitRegister* reg, int target_qubit);
void getStateVector(const QubitRegister* reg, Complex* h_state);

// Backward-compatible one-qubit kernels
__global__ void initializeQubit(Complex* state);
__global__ void applyHadamard(Complex* state);
__global__ void applyX(Complex* state);

#endif // ONE_QUBIT_SIMULATOR_H
