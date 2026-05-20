# Quantum Pauli String Simulation with CUDA

This project demonstrates the simulation of quantum operations using CUDA. It now includes a future-proof qubit register API built around a state vector of size `2^n`, with support for single-qubit gates, two-qubit CNOT, and per-qubit measurement with collapse.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- NVIDIA CUDA Toolkit
- A compatible NVIDIA GPU

### Building the Project

To build and run the tests, execute the following commands:

```shell
# compile
nvcc -o exec/one_qubit_simulator_test tests/one_qubit_simulator_test.cu src/one_qubit_simulator.cu -I src
# run the tests
./exec/one_qubit_simulator_test
```

Expected output includes:
```
testSingleQubitRegisterFlow passed!
testLegacyOneQubitKernelsStillWork passed!
testBellState passed!
```

## Author

- **Matteo Paltenghi** - [personal page](https://matteopaltenghi.com/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for providing the CUDA Toolkit
- The open-source community for their contributions
