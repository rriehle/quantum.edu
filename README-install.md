# Installation Guide for Quantum Computing Curriculum

## System Requirements

### Operating System Compatibility

| Library | Linux | macOS | Windows |
|---------|-------|-------|---------|
| Qiskit | ✅ Full Support | ✅ Full Support | ✅ Full Support |
| PennyLane | ✅ Full Support | ✅ Full Support | ✅ Full Support |
| Cirq | ✅ Full Support | ✅ Full Support | ✅ Full Support |
| **TensorFlow Quantum** | ✅ **Preferred** | ⚠️ **Unsupported** | ⚠️ Limited Support |

> **Important Note:** TensorFlow Quantum prefers Linux and does not officially support macOS. While other quantum libraries work across all platforms, for the best experience with TensorFlow Quantum components of this curriculum, we recommend using Linux (either natively or via Docker/WSL2).

### Hardware Requirements

- **RAM:** Minimum 8GB, recommended 16GB+
- **Storage:** 10GB free space for libraries and datasets
- **CPU:** Modern multi-core processor (2015 or newer)
- **GPU:** Optional but recommended for ML components (CUDA-compatible for TensorFlow)

## Installation Methods

### Method 1: Standard Installation (All Platforms Except macOS for TensorFlow Quantum)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/quantum.edu.git
cd quantum.edu

# 2. Create and activate virtual environment
python3 -m venv quantum_env

# Activate on Linux/macOS:
source quantum_env/bin/activate

# Activate on Windows:
# quantum_env\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install all requirements
pip install -r requirements.txt
```

### Method 2: Platform-Specific Installation

#### Linux (Recommended for Full Compatibility)

```bash
# Ubuntu/Debian users - install system dependencies first
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv
sudo apt-get install -y build-essential cmake

# Create environment and install
python3 -m venv quantum_env
source quantum_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS (Without TensorFlow Quantum)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+ via Homebrew
brew install python@3.9

# Create environment
python3.9 -m venv quantum_env
source quantum_env/bin/activate
pip install --upgrade pip

# Install requirements EXCEPT TensorFlow Quantum
pip install qiskit qiskit-aer qiskit-machine-learning
pip install pennylane cirq
pip install numpy scipy matplotlib pandas scikit-learn
pip install tensorflow torch prophet statsmodels
pip install plotly seaborn
pip install jupyter ipywidgets pytest
```

#### Windows (Via WSL2 for TensorFlow Quantum)

```powershell
# Option A: Native Windows (without TensorFlow Quantum)
python -m venv quantum_env
quantum_env\Scripts\activate
pip install --upgrade pip
# Install all except tensorflow-quantum
pip install -r requirements-windows.txt

# Option B: WSL2 (Recommended for full compatibility)
# 1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
# 2. Install Ubuntu from Microsoft Store
# 3. Follow Linux installation instructions above
```

### Method 3: Docker Installation (Recommended for macOS Users)

For macOS users who need TensorFlow Quantum, we provide a Docker container:

```bash
# Build the Docker image
docker build -t quantum-edu .

# Run the container with Jupyter
docker run -it -p 8888:8888 -v $(pwd):/workspace quantum-edu

# Or run interactive shell
docker run -it -v $(pwd):/workspace quantum-edu /bin/bash
```

**Dockerfile:**
```dockerfile
FROM ubuntu:22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
```

## Dependency Management

### Core Requirements File

The main `requirements.txt` includes all libraries. For platform-specific installations:

**requirements-core.txt** (Works on all platforms):
```txt
# Core Quantum Libraries (except TensorFlow Quantum)
qiskit>=0.39.0
qiskit-aer>=0.11.0
qiskit-machine-learning>=0.5.0
pennylane>=0.28.0
cirq>=1.0.0

# Scientific Computing
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Machine Learning
tensorflow>=2.10.0
torch>=1.12.0
prophet>=1.1.0
statsmodels>=0.13.0

# Visualization
plotly>=5.0.0
seaborn>=0.11.0

# Development
jupyter>=1.0.0
ipywidgets>=7.6.0
pytest>=7.0.0
```

**requirements-linux.txt** (Full installation):
```txt
-r requirements-core.txt
tensorflow-quantum>=0.7.0
```

## Verification

### Verify Installation

Create and run this verification script:

```python
# verify_installation.py
import sys
import importlib
from colorama import init, Fore, Style

init(autoreset=True)

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        importlib.import_module(module_name)
        version = None
        try:
            module = sys.modules[module_name]
            version = getattr(module, '__version__', 'unknown')
        except:
            version = 'unknown'
        print(f"{Fore.GREEN}✓ {package_name:<25} {version}")
        return True
    except ImportError as e:
        print(f"{Fore.RED}✗ {package_name:<25} Not installed")
        return False

def main():
    print("\n" + "="*50)
    print("Quantum Computing Curriculum - Installation Check")
    print("="*50 + "\n")

    libraries = [
        ("qiskit", "Qiskit"),
        ("qiskit_aer", "Qiskit Aer"),
        ("qiskit_machine_learning", "Qiskit ML"),
        ("pennylane", "PennyLane"),
        ("cirq", "Cirq"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("tensorflow", "TensorFlow"),
        ("torch", "PyTorch"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
        ("jupyter", "Jupyter"),
    ]

    all_success = True
    for module, name in libraries:
        if not check_import(module, name):
            all_success = False

    # Special check for TensorFlow Quantum
    print(f"\n{Fore.YELLOW}Platform-Specific Libraries:")
    print("-" * 40)

    tfq_available = check_import("tensorflow_quantum", "TensorFlow Quantum")
    if not tfq_available:
        import platform
        if platform.system() == "Darwin":
            print(f"{Fore.YELLOW}ℹ TensorFlow Quantum is not supported on macOS")
            print(f"{Fore.YELLOW}  Consider using Docker or Linux VM for TFQ exercises")
        else:
            print(f"{Fore.YELLOW}ℹ TensorFlow Quantum not installed")
            print(f"{Fore.YELLOW}  Install with: pip install tensorflow-quantum")

    print("\n" + "="*50)
    if all_success and tfq_available:
        print(f"{Fore.GREEN}{Style.BRIGHT}All libraries successfully installed!")
    elif all_success:
        print(f"{Fore.GREEN}Core libraries successfully installed!")
        print(f"{Fore.YELLOW}Some optional libraries may need attention.")
    else:
        print(f"{Fore.RED}Some required libraries are missing.")
        print("Please run: pip install -r requirements.txt")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
```

Run verification:
```bash
python verify_installation.py
```

## Troubleshooting

### Common Issues and Solutions

#### 1. TensorFlow Quantum on macOS

**Problem:** `ERROR: Could not find a version that satisfies the requirement tensorflow-quantum`

**Solution:** TensorFlow Quantum doesn't support macOS. Options:
- Use Docker (recommended)
- Use a Linux VM
- Skip TensorFlow Quantum exercises
- Use Google Colab for TFQ-specific notebooks

#### 2. Conflicting Dependencies

**Problem:** Version conflicts between packages

**Solution:**
```bash
# Create a fresh environment
python3 -m venv quantum_env_fresh
source quantum_env_fresh/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --upgrade --force-reinstall
```

#### 3. Memory Issues During Installation

**Problem:** Installation killed or hanging

**Solution:**
```bash
# Install packages one at a time
pip install qiskit
pip install pennylane
pip install tensorflow
# Continue with other packages...
```

#### 4. CUDA/GPU Issues

**Problem:** TensorFlow not recognizing GPU

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]
```

### Platform-Specific Workarounds

#### macOS Users - TensorFlow Quantum Alternatives

For exercises requiring TensorFlow Quantum on macOS:

1. **Use Google Colab** (Recommended for beginners)
   ```python
   # In Colab notebook
   !pip install tensorflow-quantum
   ```

2. **Docker Container**
   ```bash
   docker run -it -p 8888:8888 quantum-edu-tfq
   ```

3. **Alternative Libraries**
   - Replace TFQ exercises with PennyLane equivalents
   - Use Qiskit Machine Learning as substitute

#### Windows Users - WSL2 Setup

1. Enable WSL2:
   ```powershell
   wsl --install
   ```

2. Install Ubuntu:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

3. Follow Linux installation instructions

## Getting Help

### Resources

- **GitHub Issues:** [Report installation problems](https://github.com/yourusername/quantum.edu/issues)
- **Documentation:** [Full installation docs](https://docs.quantum-curriculum.org/install)
- **Community Forum:** [Get help from community](https://forum.quantum-curriculum.org)

### Quick Diagnostic

Run this command to generate a diagnostic report:

```bash
python -c "
import platform
import sys
print('System Info:')
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Architecture: {platform.machine()}')
" > diagnostic.txt

pip list >> diagnostic.txt
echo "Diagnostic report saved to diagnostic.txt"
```

## Next Steps

After successful installation:

1. **Run verification script:** `python verify_installation.py`
2. **Start Jupyter:** `jupyter lab`
3. **Open first notebook:** `week1-exercises.ipynb`
4. **Join the community:** Links in main README

---

*Last updated: November 2024*
*For latest installation instructions, check the [GitHub repository](https://github.com/yourusername/quantum.edu)*
=======
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

