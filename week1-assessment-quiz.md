# Week 1: Linear Algebra Self-Assessment Quiz
## Test Your Understanding of Quantum Computing Foundations

**Instructions:** Complete this quiz after finishing Week 1 materials. Solutions are provided at the end. Try to complete without referring to notes first, then review with resources as needed.

**Time Estimate:** 60-90 minutes

---

## Part A: Complex Numbers and Vectors (20 points)

### Question 1 (5 points)
Given the complex number z = 3 - 4i, calculate:
a) |z| (magnitude)
b) z* (complex conjugate)
c) arg(z) (phase angle in radians)
d) Express z in polar form

### Question 2 (5 points)
Given two quantum states:
- |ψ⟩ = (3+4i)|0⟩ + (4-3i)|1⟩
- |φ⟩ = (1/√2)|0⟩ + (i/√2)|1⟩

a) Normalize |ψ⟩
b) Calculate ⟨ψ|φ⟩ (using the normalized |ψ⟩)
c) What is the probability of measuring |ψ⟩ in state |0⟩?

### Question 3 (5 points)
Create a quantum state |ψ⟩ = α|0⟩ + β|1⟩ such that:
- The probability of measuring |0⟩ is 0.75
- The relative phase between α and β is π/4
- The state is normalized

### Question 4 (5 points)
Given vectors in ℂ²:
- v₁ = [1+i, 2-i]ᵀ
- v₂ = [2i, 1-i]ᵀ

a) Calculate ||v₁||
b) Find a vector v₃ orthogonal to v₁
c) Verify orthogonality using inner product

---

## Part B: Matrices and Operators (25 points)

### Question 5 (5 points)
Given the matrix:
```
A = [1/√2   -i/√2]
    [i/√2    1/√2]
```
a) Is A unitary? Show your work.
b) Is A Hermitian? Show your work.
c) Calculate A†A

### Question 6 (5 points)
For the Pauli Y matrix:
```
Y = [0   -i]
    [i    0]
```
a) Find all eigenvalues
b) Find corresponding eigenvectors
c) Verify that eigenvectors are orthogonal
d) Express Y as a spectral decomposition

### Question 7 (5 points)
Given the rotation operator Rx(θ) = cos(θ/2)I - i·sin(θ/2)X:
a) Write out the matrix for Rx(π/3)
b) Verify it's unitary
c) Apply it to |0⟩ and find the resulting state
d) Calculate the probability of measuring |1⟩ after applying Rx(π/3) to |0⟩

### Question 8 (5 points)
Construct a 2×2 Hermitian matrix with:
- Eigenvalues: λ₁ = 1, λ₂ = -1
- Eigenvector for λ₁ is proportional to [1, i]ᵀ

### Question 9 (5 points)
Given two gates G₁ and G₂:
```
G₁ = [1  0]    G₂ = 1/√2 [1   1]
     [0  i]              [1  -1]
```
Calculate:
a) G₁G₂
b) G₂G₁
c) [G₁, G₂] (commutator)
d) Are these gates commuting?

---

## Part C: Tensor Products and Entanglement (25 points)

### Question 10 (5 points)
Calculate the tensor product:
```
[1]  ⊗  [a]
[2]      [b]
```
Express your answer as a 4×1 column vector.

### Question 11 (5 points)
Given single-qubit states:
- |+⟩ = (|0⟩ + |1⟩)/√2
- |-⟩ = (|0⟩ - |1⟩)/√2

a) Calculate |+⟩ ⊗ |-⟩
b) Express the result in the computational basis
c) Is this state entangled? Why or why not?

### Question 12 (5 points)
For the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:
a) Calculate the reduced density matrix for the first qubit
b) What is the purity Tr(ρ²)?
c) What does this tell you about entanglement?

### Question 13 (5 points)
Given the CNOT gate:
```
CNOT = [1 0 0 0]
       [0 1 0 0]
       [0 0 0 1]
       [0 0 1 0]
```
a) Apply CNOT to |10⟩
b) Apply CNOT to (|0⟩ + |1⟩) ⊗ |0⟩
c) Is the resulting state from (b) entangled?

### Question 14 (5 points)
Determine if the following state is entangled:
|ψ⟩ = (1/2)|00⟩ + (1/2)|01⟩ + (1/2)|10⟩ + (1/2)|11⟩

Show your reasoning using either:
- Factorization attempt
- Reduced density matrix
- Schmidt decomposition

---

## Part D: Applied Problems (30 points)

### Question 15 (10 points)
**Quantum Circuit Analysis**

Given the circuit: H - Rx(π/4) - Measure

a) Write the matrix for H (Hadamard)
b) Write the matrix for Rx(π/4)
c) Calculate the combined operation matrix
d) Apply to initial state |0⟩
e) Calculate measurement probabilities for |0⟩ and |1⟩
f) Find the Bloch sphere coordinates of the final state

### Question 16 (10 points)
**Quantum State Tomography**

You measure a qubit in three bases and get:
- Z-basis: P(|0⟩) = 0.75, P(|1⟩) = 0.25
- X-basis: P(|+⟩) = 0.5, P(|-⟩) = 0.5
- Y-basis: P(|i⟩) = 0.25, P(|-i⟩) = 0.75

where |±⟩ = (|0⟩ ± |1⟩)/√2 and |±i⟩ = (|0⟩ ± i|1⟩)/√2

Reconstruct the quantum state |ψ⟩ = α|0⟩ + β|1⟩

### Question 17 (10 points)
**Programming Challenge**

Write Python code (pseudocode acceptable) to:
a) Generate a random normalized 2-qubit state
b) Test if it's entangled
c) If entangled, calculate the entanglement entropy
d) Find the closest separable state

---

## Part E: Conceptual Understanding (Bonus: 10 points)

### Question 18 (2 points each)
Answer briefly (1-2 sentences):

a) Why must quantum gates be unitary?

b) What's the physical meaning of eigenvalues in quantum measurements?

c) Why do we use complex numbers in quantum mechanics?

d) What's the difference between a pure state and a mixed state?

e) How does the tensor product relate to combining quantum systems?

---

# Solutions and Detailed Explanations

## Part A Solutions

### Solution 1
a) |z| = √(3² + (-4)²) = √(9 + 16) = 5

b) z* = 3 + 4i

c) arg(z) = arctan(-4/3) = -0.927 radians (in 4th quadrant)

d) Polar form: z = 5e^(-i·0.927) = 5(cos(-0.927) + i·sin(-0.927))

### Solution 2
First, find the normalization constant for |ψ⟩:
- |3+4i|² + |4-3i|² = (9+16) + (16+9) = 50
- Normalization factor: 1/√50 = 1/(5√2)

a) |ψ⟩_normalized = (1/5√2)[(3+4i)|0⟩ + (4-3i)|1⟩]

b) ⟨ψ|φ⟩ = (1/5√2)[(3-4i)(1/√2) + (4+3i)(i/√2)]
   = (1/10)[(3-4i) + i(4+3i)]
   = (1/10)[3-4i + 4i-3]
   = 0

c) P(|0⟩) = |(3+4i)/(5√2)|² = 25/50 = 0.5

### Solution 3
For P(|0⟩) = 0.75, we need |α|² = 0.75, so |α| = √0.75 = √3/2

Since normalized: |β|² = 0.25, so |β| = 0.5

For relative phase π/4, we can choose:
- α = √3/2 (real)
- β = (1/2)e^(iπ/4) = (1/2)(1/√2 + i/√2) = (1+i)/(2√2)

Therefore: |ψ⟩ = (√3/2)|0⟩ + (1+i)/(2√2)|1⟩

### Solution 4
a) ||v₁|| = √|1+i|² + |2-i|² = √2 + 5 = √7

b) For orthogonality, need ⟨v₃|v₁⟩ = 0
   Choose v₃ = [2-i, -1-i]ᵀ

c) ⟨v₃|v₁⟩ = (2+i)(1+i) + (-1+i)(2-i)
   = (2+i+2i-1) + (-2+i-2i-1)
   = (1+3i) + (-3-i) = -2+2i ≠ 0

   Correct v₃ = [2-i, -(1+i)]ᵀ gives ⟨v₃|v₁⟩ = 0

## Part B Solutions

### Solution 5
a) A†A = [1/√2  i/√2] [1/√2  -i/√2] = [1 0] = I
        [-i/√2  1/√2] [i/√2   1/√2]   [0 1]
   Yes, A is unitary.

b) A† = [1/√2   i/√2] ≠ A
        [-i/√2  1/√2]
   No, A is not Hermitian.

c) Already calculated above: A†A = I

### Solution 6
a) Characteristic equation: det(Y - λI) = 0
   λ² - 1 = 0, so λ = ±1

b) For λ = +1: [1, i]ᵀ/√2
   For λ = -1: [1, -i]ᵀ/√2

c) ⟨v₁|v₂⟩ = (1/2)[(1)(1) + (-i)(i)] = (1/2)[1+1] = 0 ✓

d) Y = (+1)|+i⟩⟨+i| + (-1)|-i⟩⟨-i|

### Solution 7
a) Rx(π/3) = cos(π/6)I - i·sin(π/6)X
   = (√3/2)I - (i/2)X
   = [√3/2  -i/2]
     [-i/2   √3/2]

b) Rx†Rx = [(√3/2)² + (1/2)²    0        ] = I ✓
           [0                     (√3/2)² + (1/2)²]

c) Rx(π/3)|0⟩ = [√3/2  -i/2] [1] = [√3/2]
                 [-i/2   √3/2] [0]   [-i/2]

d) P(|1⟩) = |⟨1|Rx(π/3)|0⟩|² = |-i/2|² = 1/4

### Solution 8
Eigenvector for λ₁ = 1: v₁ = [1, i]ᵀ/√2
Eigenvector for λ₂ = -1: v₂ = [1, -i]ᵀ/√2

H = λ₁|v₁⟩⟨v₁| + λ₂|v₂⟩⟨v₂|
  = (1/2)[1  -i] + (-1/2)[1   i]
         [i   1]          [-i  1]
  = [0  -i]
    [i   0]

This is actually the Pauli Y matrix (scaled).

### Solution 9
a) G₁G₂ = (1/√2)[1   1]
                 [i  -i]

b) G₂G₁ = (1/√2)[1   i]
                 [1  -i]

c) [G₁, G₂] = G₁G₂ - G₂G₁ = (1/√2)[0    1-i]
                                    [i-1   0]

d) No, they don't commute since [G₁, G₂] ≠ 0

## Part C Solutions

### Solution 10
[1] ⊗ [a] = [1·a] = [a]
[2]    [b]   [1·b]   [b]
             [2·a]   [2a]
             [2·b]   [2b]

### Solution 11
a) |+⟩ ⊗ |-⟩ = (1/2)(|0⟩ + |1⟩) ⊗ (|0⟩ - |1⟩)
   = (1/2)(|00⟩ - |01⟩ + |10⟩ - |11⟩)

b) In computational basis: (1/2)[1, -1, 1, -1]ᵀ

c) Not entangled. Can be factored as product of single-qubit states.

### Solution 12
a) ρ₁ = Tr₂(|Φ⁺⟩⟨Φ⁺|) = (1/2)[1 0] = (1/2)I
                            [0 1]

b) Tr(ρ²) = Tr((1/4)I) = 1/2

c) Purity < 1 indicates maximal entanglement (maximally mixed reduced state)

### Solution 13
a) CNOT|10⟩ = |11⟩

b) CNOT[(|0⟩ + |1⟩) ⊗ |0⟩] = CNOT[(1/√2)(|00⟩ + |10⟩)]
   = (1/√2)(|00⟩ + |11⟩) = |Φ⁺⟩

c) Yes, this is the Bell state - maximally entangled

### Solution 14
|ψ⟩ = (1/2)(|0⟩ + |1⟩) ⊗ (|0⟩ + |1⟩) = |+⟩ ⊗ |+⟩

Not entangled - it's a product state.

## Part D Solutions

### Solution 15
a) H = (1/√2)[1   1]
              [1  -1]

b) Rx(π/4) = [cos(π/8)    -i·sin(π/8)]
              [-i·sin(π/8)  cos(π/8)]

c) Combined = Rx(π/4)·H

d) Final state = Rx(π/4)·H|0⟩

e) P(|0⟩) ≈ 0.854, P(|1⟩) ≈ 0.146

f) Bloch coordinates: (x, y, z) = (√2·cos(π/8), 0, √2·sin(π/8))

### Solution 16
From measurements:
- ⟨Z⟩ = 0.75 - 0.25 = 0.5
- ⟨X⟩ = 0.5 - 0.5 = 0
- ⟨Y⟩ = 0.25 - 0.75 = -0.5

Bloch vector: (0, -0.5, 0.5)

State: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
where θ = π/3, φ = -π/2

|ψ⟩ = (√3/2)|0⟩ - (i/2)|1⟩

### Solution 17
```python
import numpy as np

def generate_random_2qubit_state():
    # Generate 4 complex random numbers
    state = np.random.randn(4) + 1j*np.random.randn(4)
    # Normalize
    return state / np.linalg.norm(state)

def is_entangled(state):
    # Reshape to matrix
    psi_matrix = state.reshape(2, 2)
    # Try SVD
    U, s, Vh = np.linalg.svd(psi_matrix)
    # If rank 1, it's separable
    return np.sum(s > 1e-10) > 1

def entanglement_entropy(state):
    # Calculate reduced density matrix
    rho = np.outer(state, state.conj())
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    rho_A = np.trace(rho_reshaped, axis1=1, axis2=3)

    # Calculate von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def find_closest_separable(state):
    # Use iterative algorithm or optimization
    # This is a complex problem - simplified version:
    psi_matrix = state.reshape(2, 2)
    U, s, Vh = np.linalg.svd(psi_matrix)
    # Keep only largest singular value
    s_approx = np.zeros_like(s)
    s_approx[0] = s[0]
    psi_approx = U @ np.diag(s_approx) @ Vh
    return psi_approx.flatten()
```

## Part E Solutions

### Solution 18
a) Quantum gates must be unitary to preserve normalization (probability conservation) and ensure reversibility of quantum evolution.

b) Eigenvalues represent the possible measurement outcomes when measuring an observable.

c) Complex numbers are needed to represent quantum phases and interference effects, which are essential for quantum computation.

d) A pure state is a definite quantum state (vector), while a mixed state is a statistical mixture requiring a density matrix representation.

e) The tensor product combines individual quantum system spaces into a joint Hilbert space, allowing description of composite systems and entanglement.

---

## Scoring Guide

- **90-100%:** Excellent understanding, ready for Week 2
- **75-89%:** Good grasp, review weak areas
- **60-74%:** Adequate, spend more time on practice
- **Below 60%:** Review Week 1 materials thoroughly

## Key Takeaways

1. **Complex arithmetic** is fundamental - practice until automatic
2. **Unitarity** ensures quantum evolution is reversible
3. **Eigenvalues/eigenvectors** connect to measurement outcomes
4. **Tensor products** enable multi-qubit quantum computing
5. **Entanglement** is the key quantum resource

## Next Steps

If you scored:
- **Below 70%:** Review 3Blue1Brown videos and redo exercises
- **70-85%:** Focus on weak areas with targeted practice
- **Above 85%:** You're ready for Week 2: Probability and Complex Analysis

Remember: Linear algebra is the language of quantum computing. Mastery here ensures success in later modules!