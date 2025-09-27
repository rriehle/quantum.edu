"""
Week 1: Linear Algebra Essentials - Practical Exercises
Quantum Computing Curriculum for Expert Computer Scientists

This notebook contains hands-on exercises for mastering linear algebra
concepts essential for quantum computing.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import scipy.linalg as la

# Set display precision for cleaner output
np.set_printoptions(precision=4, suppress=True)

# ============================================================================
# PART 1: COMPLEX NUMBERS AND VECTORS
# ============================================================================

def complex_to_polar(z: complex) -> Tuple[float, float]:
    """
    Convert complex number to polar form (magnitude, phase).

    Args:
        z: Complex number

    Returns:
        Tuple of (magnitude, phase in radians)
    """
    magnitude = np.abs(z)
    phase = np.angle(z)
    return magnitude, phase


def normalize_state(state: np.ndarray) -> np.ndarray:
    """
    Normalize a quantum state vector.

    Args:
        state: Unnormalized state vector

    Returns:
        Normalized state vector with ||ψ|| = 1
    """
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return state / norm


def create_superposition(amplitudes: List[complex]) -> np.ndarray:
    """
    Create a superposition state with given amplitudes.

    Args:
        amplitudes: List of complex amplitudes

    Returns:
        Normalized quantum state vector
    """
    state = np.array(amplitudes, dtype=complex)
    return normalize_state(state)


def inner_product(psi: np.ndarray, phi: np.ndarray) -> complex:
    """
    Calculate inner product ⟨ψ|φ⟩.

    Args:
        psi: First state vector
        phi: Second state vector

    Returns:
        Complex inner product value
    """
    return np.vdot(psi, phi)  # Note: vdot conjugates first argument


def probability_amplitude(state: np.ndarray, basis_state: int) -> float:
    """
    Calculate probability of measuring state in given basis state.

    Args:
        state: Quantum state vector
        basis_state: Index of computational basis state

    Returns:
        Measurement probability
    """
    amplitude = state[basis_state]
    return np.abs(amplitude) ** 2


# Example 1.1: Complex number operations
print("=" * 60)
print("EXAMPLE 1.1: Complex Number Operations")
print("=" * 60)

z1 = 3 + 4j
z2 = 1 - 2j

print(f"z1 = {z1}")
print(f"z2 = {z2}")
print(f"z1 + z2 = {z1 + z2}")
print(f"z1 * z2 = {z1 * z2}")
print(f"Conjugate of z1 = {np.conj(z1)}")
print(f"Magnitude of z1 = {np.abs(z1)}")
print(f"Phase of z1 = {np.angle(z1)} radians")

mag, phase = complex_to_polar(z1)
print(f"Polar form of z1: {mag:.4f} * exp(i * {phase:.4f})")


# Example 1.2: Quantum state vectors
print("\n" + "=" * 60)
print("EXAMPLE 1.2: Quantum State Vectors")
print("=" * 60)

# Computational basis states
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

print("Computational basis states:")
print(f"|0⟩ = {ket_0}")
print(f"|1⟩ = {ket_1}")

# Create equal superposition (|+⟩ state)
ket_plus = create_superposition([1, 1])
print(f"\n|+⟩ = {ket_plus}")

# Create |−⟩ state
ket_minus = create_superposition([1, -1])
print(f"|−⟩ = {ket_minus}")

# Arbitrary superposition
alpha = 3 + 4j
beta = 4 - 3j
psi = create_superposition([alpha, beta])
print(f"\n|ψ⟩ = {psi}")
print(f"Norm of |ψ⟩ = {np.linalg.norm(psi)}")


# ============================================================================
# PART 2: MATRICES AND QUANTUM GATES
# ============================================================================

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard gate
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gate
S = np.array([[1, 0], [0, 1j]], dtype=complex)

# T gate (π/8 gate)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is unitary (U†U = I).

    Args:
        matrix: Matrix to check
        tolerance: Numerical tolerance for comparison

    Returns:
        True if matrix is unitary
    """
    n = matrix.shape[0]
    product = np.dot(np.conj(matrix.T), matrix)
    identity = np.eye(n, dtype=complex)
    return np.allclose(product, identity, atol=tolerance)


def is_hermitian(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is Hermitian (H† = H).

    Args:
        matrix: Matrix to check
        tolerance: Numerical tolerance for comparison

    Returns:
        True if matrix is Hermitian
    """
    return np.allclose(matrix, np.conj(matrix.T), atol=tolerance)


def apply_gate(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Apply a quantum gate to a state.

    Args:
        gate: Unitary gate matrix
        state: Quantum state vector

    Returns:
        New state after gate application
    """
    return np.dot(gate, state)


def rotation_gate(axis: str, angle: float) -> np.ndarray:
    """
    Create a rotation gate around specified axis.

    Args:
        axis: 'x', 'y', or 'z'
        angle: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    if axis.lower() == 'x':
        return np.cos(angle/2) * I - 1j * np.sin(angle/2) * X
    elif axis.lower() == 'y':
        return np.cos(angle/2) * I - 1j * np.sin(angle/2) * Y
    elif axis.lower() == 'z':
        return np.cos(angle/2) * I - 1j * np.sin(angle/2) * Z
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


# Example 2.1: Pauli matrices properties
print("\n" + "=" * 60)
print("EXAMPLE 2.1: Pauli Matrices Properties")
print("=" * 60)

print("Pauli X (NOT gate):")
print(X)
print(f"Is unitary? {is_unitary(X)}")
print(f"Is Hermitian? {is_hermitian(X)}")

print("\nPauli Y:")
print(Y)
print(f"Is unitary? {is_unitary(Y)}")
print(f"Is Hermitian? {is_hermitian(Y)}")

print("\nPauli Z:")
print(Z)
print(f"Is unitary? {is_unitary(Z)}")
print(f"Is Hermitian? {is_hermitian(Z)}")

# Pauli matrices square to identity
print("\nPauli matrices square to identity:")
print(f"X² = \n{np.dot(X, X)}")
print(f"Y² = \n{np.dot(Y, Y)}")
print(f"Z² = \n{np.dot(Z, Z)}")


# Example 2.2: Gate operations on states
print("\n" + "=" * 60)
print("EXAMPLE 2.2: Gate Operations on States")
print("=" * 60)

# Apply X gate (bit flip)
state_0 = ket_0.copy()
state_after_X = apply_gate(X, state_0)
print(f"X|0⟩ = {state_after_X}")

# Apply Hadamard gate
state_after_H = apply_gate(H, state_0)
print(f"H|0⟩ = {state_after_H}")

# Apply rotation gates
Rx_pi4 = rotation_gate('x', np.pi/4)
state_after_Rx = apply_gate(Rx_pi4, state_0)
print(f"Rx(π/4)|0⟩ = {state_after_Rx}")


# ============================================================================
# PART 3: EIGENVALUES AND EIGENVECTORS
# ============================================================================

def find_eigenvalues(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find eigenvalues and eigenvectors of a matrix.

    Args:
        matrix: Square matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


def spectral_decomposition(matrix: np.ndarray) -> List[Tuple[float, np.ndarray]]:
    """
    Perform spectral decomposition of Hermitian matrix.

    Args:
        matrix: Hermitian matrix

    Returns:
        List of (eigenvalue, projection_operator) pairs
    """
    if not is_hermitian(matrix):
        raise ValueError("Matrix must be Hermitian for spectral decomposition")

    eigenvalues, eigenvectors = find_eigenvalues(matrix)
    decomposition = []

    for i, eigenval in enumerate(eigenvalues):
        eigenvec = eigenvectors[:, i:i+1]
        projection = np.dot(eigenvec, np.conj(eigenvec.T))
        decomposition.append((eigenval, projection))

    return decomposition


def measure_in_basis(state: np.ndarray, observable: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Simulate measurement of state with given observable.

    Args:
        state: Quantum state vector
        observable: Hermitian observable matrix

    Returns:
        Tuple of (measurement outcome, post-measurement state)
    """
    eigenvalues, eigenvectors = find_eigenvalues(observable)
    probabilities = []

    for i in range(len(eigenvalues)):
        eigenvec = eigenvectors[:, i]
        prob = np.abs(inner_product(eigenvec, state)) ** 2
        probabilities.append(prob)

    # Simulate measurement outcome
    outcome_index = np.random.choice(len(eigenvalues), p=probabilities)
    outcome = eigenvalues[outcome_index].real
    post_state = eigenvectors[:, outcome_index]

    return outcome, post_state


# Example 3.1: Eigenvalues of Pauli matrices
print("\n" + "=" * 60)
print("EXAMPLE 3.1: Eigenvalues of Pauli Matrices")
print("=" * 60)

for name, matrix in [("X", X), ("Y", Y), ("Z", Z)]:
    eigenvals, eigenvecs = find_eigenvalues(matrix)
    print(f"\nPauli {name}:")
    print(f"Eigenvalues: {eigenvals.real}")
    print(f"Eigenvectors:")
    for i, vec in enumerate(eigenvecs.T):
        print(f"  λ={eigenvals[i].real:+.0f}: {vec}")


# Example 3.2: Spectral decomposition
print("\n" + "=" * 60)
print("EXAMPLE 3.2: Spectral Decomposition")
print("=" * 60)

# Decompose Z matrix
decomp = spectral_decomposition(Z)
print("Spectral decomposition of Z:")
reconstructed = np.zeros((2, 2), dtype=complex)
for eigenval, proj in decomp:
    print(f"\nEigenvalue {eigenval.real:+.0f}:")
    print(f"Projection operator:\n{proj}")
    reconstructed += eigenval * proj

print(f"\nReconstructed Z = Σ λᵢPᵢ:")
print(reconstructed)
print(f"Matches original? {np.allclose(reconstructed, Z)}")


# ============================================================================
# PART 4: TENSOR PRODUCTS AND ENTANGLEMENT
# ============================================================================

def tensor_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate tensor product (Kronecker product) of two matrices.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Tensor product A ⊗ B
    """
    return np.kron(A, B)


def create_bell_states() -> dict:
    """
    Create all four Bell states.

    Returns:
        Dictionary of Bell states
    """
    # Create |00⟩ and |11⟩ basis states
    ket_00 = tensor_product(ket_0, ket_0)
    ket_01 = tensor_product(ket_0, ket_1)
    ket_10 = tensor_product(ket_1, ket_0)
    ket_11 = tensor_product(ket_1, ket_1)

    bell_states = {
        "Φ+": (ket_00 + ket_11) / np.sqrt(2),  # |Φ⁺⟩
        "Φ-": (ket_00 - ket_11) / np.sqrt(2),  # |Φ⁻⟩
        "Ψ+": (ket_01 + ket_10) / np.sqrt(2),  # |Ψ⁺⟩
        "Ψ-": (ket_01 - ket_10) / np.sqrt(2),  # |Ψ⁻⟩
    }

    return bell_states


def partial_trace(state: np.ndarray, keep_qubit: int, n_qubits: int = 2) -> np.ndarray:
    """
    Calculate partial trace of a quantum state.

    Args:
        state: Quantum state vector or density matrix
        keep_qubit: Which qubit to keep (0 or 1 for 2-qubit system)
        n_qubits: Total number of qubits

    Returns:
        Reduced density matrix
    """
    if n_qubits != 2:
        raise NotImplementedError("Only 2-qubit systems supported")

    # Convert state vector to density matrix if needed
    if state.ndim == 1:
        rho = np.outer(state, np.conj(state))
    else:
        rho = state

    # Reshape to tensor form
    rho_tensor = rho.reshape((2, 2, 2, 2))

    if keep_qubit == 0:
        # Trace out qubit 1
        reduced = np.trace(rho_tensor, axis1=1, axis2=3)
    else:
        # Trace out qubit 0
        reduced = np.trace(rho_tensor, axis1=0, axis2=2)

    return reduced


def is_entangled(state: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a 2-qubit state is entangled.

    Args:
        state: 2-qubit state vector
        tolerance: Numerical tolerance

    Returns:
        True if state is entangled
    """
    # Calculate reduced density matrix
    rho_A = partial_trace(state, keep_qubit=0)

    # Check purity: Tr(ρ²)
    purity = np.trace(np.dot(rho_A, rho_A)).real

    # Pure state has purity = 1, mixed state < 1
    # For separable states, reduced density matrix is pure
    return purity < 1 - tolerance


# Example 4.1: Tensor products
print("\n" + "=" * 60)
print("EXAMPLE 4.1: Tensor Products")
print("=" * 60)

# Two-qubit basis states
print("Two-qubit computational basis:")
ket_00 = tensor_product(ket_0, ket_0)
print(f"|00⟩ = {ket_00.T}")

ket_01 = tensor_product(ket_0, ket_1)
print(f"|01⟩ = {ket_01.T}")

ket_10 = tensor_product(ket_1, ket_0)
print(f"|10⟩ = {ket_10.T}")

ket_11 = tensor_product(ket_1, ket_1)
print(f"|11⟩ = {ket_11.T}")

# Two-qubit gates
CNOT = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)

print(f"\nCNOT gate:")
print(CNOT)
print(f"Is unitary? {is_unitary(CNOT)}")


# Example 4.2: Bell states and entanglement
print("\n" + "=" * 60)
print("EXAMPLE 4.2: Bell States and Entanglement")
print("=" * 60)

bell_states = create_bell_states()

for name, state in bell_states.items():
    print(f"\n|{name}⟩ = {state.T}")
    print(f"Is entangled? {is_entangled(state)}")

    # Calculate reduced density matrix
    rho_A = partial_trace(state, keep_qubit=0)
    print(f"Reduced density matrix (qubit 0):")
    print(rho_A)
    print(f"Purity = {np.trace(np.dot(rho_A, rho_A)).real:.4f}")

# Compare with separable state
separable = tensor_product(ket_plus, ket_minus)
print(f"\nSeparable state |+⟩⊗|−⟩:")
print(f"State = {separable.T}")
print(f"Is entangled? {is_entangled(separable)}")


# ============================================================================
# PART 5: PRACTICAL QUANTUM COMPUTING APPLICATIONS
# ============================================================================

def simulate_measurement(state: np.ndarray, n_shots: int = 1000) -> dict:
    """
    Simulate multiple measurements of a quantum state.

    Args:
        state: Quantum state vector
        n_shots: Number of measurements to simulate

    Returns:
        Dictionary of measurement outcomes and counts
    """
    n_qubits = int(np.log2(len(state)))
    probabilities = np.abs(state) ** 2
    outcomes = np.arange(len(state))

    # Simulate measurements
    results = np.random.choice(outcomes, size=n_shots, p=probabilities)

    # Count outcomes
    counts = {}
    for outcome in results:
        binary = format(outcome, f'0{n_qubits}b')
        counts[binary] = counts.get(binary, 0) + 1

    return counts


def expectation_value(state: np.ndarray, observable: np.ndarray) -> float:
    """
    Calculate expectation value ⟨ψ|O|ψ⟩.

    Args:
        state: Quantum state vector
        observable: Observable operator

    Returns:
        Expectation value
    """
    return np.real(inner_product(state, apply_gate(observable, state)))


def bloch_coordinates(state: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate Bloch sphere coordinates for a qubit state.

    Args:
        state: Single-qubit state vector

    Returns:
        Tuple of (x, y, z) Bloch coordinates
    """
    x = expectation_value(state, X)
    y = expectation_value(state, Y)
    z = expectation_value(state, Z)
    return x, y, z


# Example 5.1: Measurement simulation
print("\n" + "=" * 60)
print("EXAMPLE 5.1: Measurement Simulation")
print("=" * 60)

# Measure |+⟩ state
counts = simulate_measurement(ket_plus, n_shots=1000)
print(f"Measuring |+⟩ state (1000 shots):")
for outcome, count in sorted(counts.items()):
    probability = count / 1000
    print(f"  |{outcome}⟩: {count} counts ({probability:.1%})")

# Measure Bell state
bell_phi_plus = bell_states["Φ+"]
counts = simulate_measurement(bell_phi_plus, n_shots=1000)
print(f"\nMeasuring |Φ⁺⟩ Bell state (1000 shots):")
for outcome, count in sorted(counts.items()):
    probability = count / 1000
    print(f"  |{outcome}⟩: {count} counts ({probability:.1%})")


# Example 5.2: Bloch sphere representation
print("\n" + "=" * 60)
print("EXAMPLE 5.2: Bloch Sphere Coordinates")
print("=" * 60)

states_to_test = [
    ("|0⟩", ket_0),
    ("|1⟩", ket_1),
    ("|+⟩", ket_plus),
    ("|−⟩", ket_minus),
    ("|i⟩", create_superposition([1, 1j])),
    ("|−i⟩", create_superposition([1, -1j])),
]

for name, state in states_to_test:
    x, y, z = bloch_coordinates(state)
    print(f"{name:5s}: ({x:+.3f}, {y:+.3f}, {z:+.3f})")


# ============================================================================
# EXERCISES FOR SELF-STUDY
# ============================================================================

print("\n" + "=" * 60)
print("EXERCISES FOR SELF-STUDY")
print("=" * 60)

print("""
1. WARM-UP EXERCISES:
   a) Create a function to generate arbitrary rotation gates Rx(θ), Ry(θ), Rz(θ)
   b) Verify that H² = I (Hadamard is self-inverse)
   c) Show that {I, X, Y, Z} form a basis for 2×2 Hermitian matrices

2. INTERMEDIATE EXERCISES:
   a) Implement the quantum phase estimation algorithm's unitary
   b) Create a function to test if a state is maximally entangled
   c) Implement the Schmidt decomposition for 2-qubit pure states

3. ADVANCED EXERCISES:
   a) Build a tensor network contraction function for multi-qubit systems
   b) Implement the Gram-Schmidt process for quantum states
   c) Create a function to find the nearest separable state to an entangled state

4. PROGRAMMING CHALLENGES:
   a) Build a Bloch sphere visualization tool using matplotlib
   b) Create a quantum state tomography reconstruction algorithm
   c) Implement a circuit optimizer that reduces gate count

5. THEORETICAL CONNECTIONS:
   a) Prove that measurement probabilities sum to 1 using linear algebra
   b) Show that unitary evolution preserves inner products
   c) Derive the uncertainty principle using commutator algebra
""")


# ============================================================================
# HELPER FUNCTIONS FOR VISUALIZATION
# ============================================================================

def plot_complex_plane(complex_numbers: List[complex], labels: List[str] = None):
    """
    Plot complex numbers on the complex plane.

    Args:
        complex_numbers: List of complex numbers to plot
        labels: Optional labels for each point
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, z in enumerate(complex_numbers):
        ax.scatter(z.real, z.imag, s=100, zorder=5)
        if labels:
            ax.annotate(labels[i], (z.real, z.imag),
                       xytext=(5, 5), textcoords='offset points')
        # Draw vector from origin
        ax.arrow(0, 0, z.real, z.imag, head_width=0.1,
                head_length=0.1, fc='blue', ec='blue', alpha=0.3)

    # Draw axes
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Complex Plane Visualization')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_probability_distribution(state: np.ndarray):
    """
    Plot probability distribution for quantum state measurement.

    Args:
        state: Quantum state vector
    """
    n_qubits = int(np.log2(len(state)))
    probabilities = np.abs(state) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Probability bar chart
    basis_labels = [format(i, f'0{n_qubits}b') for i in range(len(state))]
    ax1.bar(basis_labels, probabilities)
    ax1.set_xlabel('Basis State')
    ax1.set_ylabel('Probability')
    ax1.set_title('Measurement Probability Distribution')
    ax1.set_ylim([0, 1])

    # Amplitude polar plot
    ax2 = plt.subplot(122, projection='polar')
    phases = np.angle(state)
    amplitudes = np.abs(state)

    for i, (amp, phase) in enumerate(zip(amplitudes, phases)):
        ax2.plot([phase], [amp], 'o', markersize=10,
                label=f'|{basis_labels[i]}⟩')

    ax2.set_title('Amplitude and Phase Distribution')
    ax2.set_ylim([0, 1])
    ax2.legend(bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    return fig


# Save functions for import in other notebooks
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Week 1 Linear Algebra Exercises Complete!")
    print("=" * 60)
    print("\nKey functions available for import:")
    print("- normalize_state()")
    print("- create_superposition()")
    print("- inner_product()")
    print("- is_unitary()")
    print("- is_hermitian()")
    print("- apply_gate()")
    print("- tensor_product()")
    print("- create_bell_states()")
    print("- is_entangled()")
    print("- simulate_measurement()")
    print("- bloch_coordinates()")
    print("\nVisualization functions:")
    print("- plot_complex_plane()")
    print("- plot_probability_distribution()")