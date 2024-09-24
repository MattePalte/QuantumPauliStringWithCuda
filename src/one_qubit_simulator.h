#ifndef ONE_QUBIT_SIMULATOR_H
#define ONE_QUBIT_SIMULATOR_H

#include <cuComplex.h>

// Assuming Complex is defined as cuDoubleComplex
typedef cuDoubleComplex Complex;

// Function declarations
__global__ void initializeQubit(Complex* state);
__global__ void applyHadamard(Complex* state);
__global__ void applyX(Complex* state);

#endif // ONE_QUBIT_SIMULATOR_H