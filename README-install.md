# Installation Guide for Quantum Computing Environment

This guide documents the installation process for the quantum computing and machine learning packages in this repository.

## Prerequisites

### System Dependencies

Before installing Python packages, you need to install the following system dependencies:

```bash
# Update package list
sudo apt-get update

# Install Fortran compiler (required for building scipy from source)
sudo apt-get install -y gfortran

# Install OpenBLAS libraries (required for numerical computations)
sudo apt-get install -y libopenblas-dev
```

### Python Environment

- Python 3.11+ recommended
- pip package manager

## Installation Issues and Solutions

### Problem: Dependency Conflicts

The original `requirements.txt` file contains unpinned package versions, which can lead to dependency resolution conflicts. Specifically:

1. **tensorflow-quantum==0.7.3** pins `cirq-core==1.3.0` and `numpy~=1.16`
2. These strict version requirements conflict with newer versions of other packages
3. pip attempts to resolve conflicts by building older versions (scipy 1.9.1) from source
4. Building from source fails without proper system dependencies (gfortran, OpenBLAS)

### Solution: Staged Installation

To avoid dependency conflicts, install packages in stages:

#### Step 1: Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y gfortran libopenblas-dev
```

#### Step 2: Upgrade pip and Build Tools

```bash
pip install --upgrade pip wheel setuptools
```

#### Step 3: Install tensorflow-quantum First

Install tensorflow-quantum with its pinned dependencies before other packages:

```bash
pip install tensorflow-quantum==0.7.3
```

#### Step 4: Install Remaining Packages

```bash
pip install qiskit qiskit-aer qiskit-machine-learning pennylane \
           tensorflow torch prophet statsmodels plotly seaborn \
           jupyter ipywidgets pytest scikit-learn
```

## Alternative: Use Fixed Requirements File

A `requirements-minimal.txt` file has been created with compatible version constraints:

```bash
pip install -r requirements-minimal.txt
```

This file installs tensorflow-quantum first, then other packages with compatible version ranges.

## Verification

Verify the installation by checking key packages:

```bash
pip list | grep -E "tensorflow-quantum|qiskit|pennylane|torch|scipy|numpy"
```

Expected output should show:
- numpy 2.3.3 or 1.26.4 (depending on resolution)
- scipy 1.15.3+
- qiskit 1.4.4+
- tensorflow-quantum 0.7.3
- pennylane 0.42.3+
- torch 2.8.0+

## Troubleshooting

### Error: "No Fortran compiler found"

**Solution:** Install gfortran:
```bash
sudo apt-get install -y gfortran
```

### Error: "OpenBLAS not found"

**Solution:** Install OpenBLAS development libraries:
```bash
sudo apt-get install -y libopenblas-dev
```

### Error: "Dependency resolution taking too long"

**Solution:** This occurs when pip tries to resolve complex dependency conflicts. Use the staged installation approach described above instead of installing all packages at once.

### Error: Building scipy from source fails

**Solution:** Ensure both gfortran and libopenblas-dev are installed. If the issue persists, explicitly install a recent scipy version first:
```bash
pip install scipy>=1.11.0
```

## Package Versions

Key package versions that work together:
- tensorflow-quantum: 0.7.3
- cirq-core: 1.3.0 (pinned by tensorflow-quantum)
- numpy: 1.26.4 or 2.3.3
- scipy: 1.15.3+
- qiskit: 1.4.4+
- pennylane: 0.42.3+
- tensorflow: 2.20.0
- torch: 2.8.0

## Notes

- The dependency conflicts arise primarily from tensorflow-quantum's strict version requirements
- Future updates to tensorflow-quantum may resolve these conflicts
- Consider using virtual environments to isolate dependencies for different projects
- If you don't need tensorflow-quantum, removing it from requirements will simplify installation