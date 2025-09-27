"""
Quantum-Enhanced Prediction: Practical Labs and Exercises
For Quantum Computing Curriculum - Predictive Analytics Module

This file contains hands-on implementations of quantum-classical hybrid
prediction models for time-series forecasting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# For quantum computing (assuming Qiskit and PennyLane are installed)
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("PennyLane not installed. Some quantum features will be simulated classically.")

try:
    from qiskit import QuantumCircuit, execute, Aer
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not installed. Some quantum features will be simulated classically.")

# ============================================================================
# PART 1: DATA PREPARATION AND CLASSICAL BASELINES
# ============================================================================

class TimeSeriesData:
    """Helper class for time-series data preparation."""

    def __init__(self, data: np.ndarray, window_size: int = 10, horizon: int = 1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.normalized_data = None
        self.scaler_params = {}

    def normalize(self) -> np.ndarray:
        """Normalize time-series data to [0, 1] range."""
        self.scaler_params['min'] = np.min(self.data)
        self.scaler_params['max'] = np.max(self.data)
        self.normalized_data = (self.data - self.scaler_params['min']) / \
                              (self.scaler_params['max'] - self.scaler_params['min'])
        return self.normalized_data

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale."""
        return data * (self.scaler_params['max'] - self.scaler_params['min']) + \
               self.scaler_params['min']

    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences for training."""
        if self.normalized_data is None:
            self.normalize()

        X, y = [], []
        for i in range(len(self.normalized_data) - self.window_size - self.horizon + 1):
            X.append(self.normalized_data[i:i + self.window_size])
            y.append(self.normalized_data[i + self.window_size:i + self.window_size + self.horizon])

        return np.array(X), np.array(y)

    def train_test_split(self, test_ratio: float = 0.2) -> Dict:
        """Split data into training and testing sets."""
        X, y = self.create_sequences()
        split_idx = int(len(X) * (1 - test_ratio))

        return {
            'X_train': X[:split_idx],
            'y_train': y[:split_idx],
            'X_test': X[split_idx:],
            'y_test': y[split_idx:]
        }


def generate_synthetic_timeseries(n_points: int = 1000,
                                 pattern: str = 'sine') -> np.ndarray:
    """Generate synthetic time-series data for testing."""
    t = np.linspace(0, 4 * np.pi, n_points)

    if pattern == 'sine':
        data = np.sin(t) + 0.1 * np.random.randn(n_points)
    elif pattern == 'complex':
        data = np.sin(t) + 0.5 * np.sin(3 * t) + 0.2 * np.sin(7 * t) + \
               0.1 * np.random.randn(n_points)
    elif pattern == 'trend':
        data = 0.1 * t + np.sin(t) + 0.1 * np.random.randn(n_points)
    elif pattern == 'chaotic':
        # Logistic map
        data = np.zeros(n_points)
        data[0] = 0.5
        r = 3.9  # Chaotic regime
        for i in range(1, n_points):
            data[i] = r * data[i-1] * (1 - data[i-1])
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return data


class ClassicalLSTM:
    """Simple classical LSTM baseline for comparison."""

    def __init__(self, input_size: int, hidden_size: int = 20):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> Dict:
        """Initialize LSTM weights randomly."""
        weights = {}
        # Simplified LSTM (using tanh activation)
        weights['W_f'] = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * 0.1
        weights['W_i'] = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * 0.1
        weights['W_c'] = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * 0.1
        weights['W_o'] = np.random.randn(self.hidden_size, self.input_size + self.hidden_size) * 0.1
        weights['W_y'] = np.random.randn(1, self.hidden_size) * 0.1
        return weights

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through LSTM."""
        batch_size, seq_len = X.shape
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        for t in range(seq_len):
            x_t = np.array([X[0, t]])  # Single feature
            combined = np.concatenate([x_t, h])

            # LSTM gates (simplified)
            f_t = self._sigmoid(self.weights['W_f'] @ combined)
            i_t = self._sigmoid(self.weights['W_i'] @ combined)
            c_tilde = np.tanh(self.weights['W_c'] @ combined)
            o_t = self._sigmoid(self.weights['W_o'] @ combined)

            c = f_t * c + i_t * c_tilde
            h = o_t * np.tanh(c)

        # Output layer
        y = self.weights['W_y'] @ h
        return y

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = []
        for i in range(X.shape[0]):
            pred = self.forward(X[i:i+1])
            predictions.append(pred)
        return np.array(predictions).squeeze()


# ============================================================================
# PART 2: QUANTUM FEATURE MAPS AND ENCODINGS
# ============================================================================

if PENNYLANE_AVAILABLE:

    class QuantumFeatureMap:
        """Quantum feature map for time-series encoding."""

        def __init__(self, n_qubits: int = 4, n_layers: int = 2):
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.dev = qml.device('default.qubit', wires=n_qubits)

        def angle_encoding(self, x: np.ndarray) -> None:
            """Encode classical data using rotation angles."""
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i] * np.pi, wires=i)

        def amplitude_encoding(self, x: np.ndarray) -> None:
            """Encode classical data in quantum amplitudes."""
            # Pad and normalize
            padded = np.zeros(2**self.n_qubits)
            padded[:len(x)] = x
            normalized = padded / np.linalg.norm(padded)
            qml.QubitStateVector(normalized, wires=range(self.n_qubits))

        def iqp_encoding(self, x: np.ndarray) -> None:
            """Instantaneous Quantum Polynomial encoding."""
            # First layer of Hadamards
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Encode features
            for i in range(min(len(x), self.n_qubits)):
                qml.RZ(x[i], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])

            # Second encoding layer
            for i in range(min(len(x), self.n_qubits)):
                qml.RZ(x[i]**2, wires=i)

        @qml.qnode(device=qml.device('default.qubit', wires=4))
        def quantum_kernel_circuit(self, x1: np.ndarray, x2: np.ndarray) -> float:
            """Compute quantum kernel between two samples."""
            # Encode first sample
            self.angle_encoding(x1)
            # Inverse of second sample encoding
            qml.inv(self.angle_encoding)(x2)
            return qml.probs(wires=range(self.n_qubits))

        def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
            """Calculate quantum kernel value."""
            probs = self.quantum_kernel_circuit(x1, x2)
            return probs[0]  # Probability of all zeros state


# ============================================================================
# PART 3: VARIATIONAL QUANTUM MODELS
# ============================================================================

if PENNYLANE_AVAILABLE:

    class VariationalQuantumRegressor:
        """Variational quantum circuit for regression tasks."""

        def __init__(self, n_qubits: int = 4, n_layers: int = 3):
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self.params = self._initialize_parameters()

        def _initialize_parameters(self) -> np.ndarray:
            """Initialize variational parameters."""
            # Parameters for rotation gates and controlled operations
            return np.random.randn(self.n_layers, self.n_qubits, 3) * 0.1

        def variational_layer(self, params_layer: np.ndarray) -> None:
            """Single layer of variational circuit."""
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RX(params_layer[i, 0], wires=i)
                qml.RY(params_layer[i, 1], wires=i)
                qml.RZ(params_layer[i, 2], wires=i)

            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Wrap around
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])

        @qml.qnode(device=qml.device('default.qubit', wires=4))
        def quantum_circuit(self, x: np.ndarray, params: np.ndarray) -> float:
            """Complete variational quantum circuit."""
            # Encode input
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i] * np.pi, wires=i)

            # Variational layers
            for layer in range(self.n_layers):
                self.variational_layer(params[layer])

            # Measurement
            return qml.expval(qml.PauliZ(0))

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions using quantum circuit."""
            predictions = []
            for x in X:
                pred = self.quantum_circuit(x, self.params)
                predictions.append(pred)
            return np.array(predictions)

        def train_step(self, X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.1) -> float:
            """Single training step using parameter-shift rule."""
            # Simplified training - in practice use proper optimizer
            loss = 0
            for i in range(len(X)):
                pred = self.quantum_circuit(X[i], self.params)
                loss += (pred - y[i])**2

            # Update parameters (simplified gradient descent)
            self.params -= learning_rate * np.random.randn(*self.params.shape) * 0.01
            return loss / len(X)


# ============================================================================
# PART 4: QUANTUM RESERVOIR COMPUTING
# ============================================================================

class QuantumReservoir:
    """Quantum reservoir computing for time-series prediction."""

    def __init__(self, n_qubits: int = 6, reservoir_size: int = 100):
        self.n_qubits = n_qubits
        self.reservoir_size = reservoir_size
        self.reservoir_states = []
        self.readout_weights = None

        if QISKIT_AVAILABLE:
            self.backend = Aer.get_backend('statevector_simulator')
        else:
            self.backend = None

    def create_reservoir_circuit(self, input_val: float) -> 'QuantumCircuit':
        """Create quantum reservoir circuit."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(self.n_qubits)

        # Initialize with superposition
        for i in range(self.n_qubits):
            qc.h(i)

        # Input encoding
        qc.ry(input_val * np.pi, 0)

        # Random unitary evolution (fixed for reservoir)
        np.random.seed(42)  # Fixed seed for reproducible reservoir
        for _ in range(3):
            # Random rotations
            for i in range(self.n_qubits):
                qc.rx(np.random.rand() * np.pi, i)
                qc.rz(np.random.rand() * np.pi, i)

            # Entangling gates
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)

        return qc

    def extract_features(self, input_sequence: np.ndarray) -> np.ndarray:
        """Extract quantum features from input sequence."""
        features = []

        for val in input_sequence:
            if QISKIT_AVAILABLE and self.backend:
                qc = self.create_reservoir_circuit(val)
                job = execute(qc, self.backend)
                statevector = job.result().get_statevector()

                # Extract features from statevector
                feature = [
                    np.abs(statevector[0])**2,  # Probability of |0...0>
                    np.real(statevector[0]),     # Real part
                    np.imag(statevector[0]),     # Imaginary part
                    np.abs(statevector[-1])**2,  # Probability of |1...1>
                ]
            else:
                # Classical simulation fallback
                feature = [
                    np.sin(val),
                    np.cos(val),
                    np.tanh(val),
                    val**2
                ]

            features.append(feature)

        return np.array(features).flatten()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train readout layer."""
        # Extract features for all training samples
        X_features = []
        for x in X:
            features = self.extract_features(x)
            X_features.append(features)
        X_features = np.array(X_features)

        # Train linear readout (using pseudo-inverse)
        self.readout_weights = np.linalg.pinv(X_features) @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using quantum reservoir."""
        predictions = []
        for x in X:
            features = self.extract_features(x)
            pred = features @ self.readout_weights
            predictions.append(pred)
        return np.array(predictions)


# ============================================================================
# PART 5: HYBRID QUANTUM-CLASSICAL MODELS
# ============================================================================

class HybridQuantumLSTM:
    """Hybrid model combining classical LSTM with quantum enhancement."""

    def __init__(self, input_size: int, hidden_size: int = 10,
                 n_qubits: int = 4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        # Classical LSTM component
        self.lstm = ClassicalLSTM(input_size, hidden_size)

        # Quantum enhancement component
        if PENNYLANE_AVAILABLE:
            self.quantum_processor = VariationalQuantumRegressor(n_qubits, n_layers=2)
        else:
            self.quantum_processor = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through hybrid model."""
        # Classical LSTM processing
        lstm_output = self.lstm.predict(X)

        # Quantum enhancement
        if self.quantum_processor:
            # Use last few time steps as quantum input
            quantum_input = X[:, -self.n_qubits:]
            quantum_enhancement = self.quantum_processor.predict(quantum_input)

            # Combine classical and quantum predictions
            combined = 0.7 * lstm_output + 0.3 * quantum_enhancement
        else:
            combined = lstm_output

        return combined

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)


# ============================================================================
# PART 6: EVALUATION AND BENCHMARKING
# ============================================================================

class PredictionEvaluator:
    """Evaluate and compare prediction models."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred)**2)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    @staticmethod
    def plot_predictions(y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                        title: str = "Model Comparison"):
        """Plot predictions from multiple models."""
        plt.figure(figsize=(12, 6))

        # Plot actual values
        plt.plot(y_true, label='Actual', color='black', linewidth=2)

        # Plot predictions
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (name, y_pred) in enumerate(predictions.items()):
            plt.plot(y_pred, label=name, color=colors[i % len(colors)],
                    alpha=0.7, linestyle='--')

        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def benchmark_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Benchmark multiple models."""
        results = {}

        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            results[name] = {
                'MSE': PredictionEvaluator.mse(y_test, y_pred),
                'MAE': PredictionEvaluator.mae(y_test, y_pred),
                'MAPE': PredictionEvaluator.mape(y_test, y_pred),
                'predictions': y_pred
            }

        return results


# ============================================================================
# LAB EXERCISES
# ============================================================================

def lab1_classical_baseline():
    """Lab 1: Implement and evaluate classical baseline models."""
    print("=" * 60)
    print("LAB 1: Classical Baseline for Time-Series Prediction")
    print("=" * 60)

    # Generate synthetic data
    data = generate_synthetic_timeseries(500, pattern='complex')
    ts = TimeSeriesData(data, window_size=10, horizon=1)

    # Prepare data
    split = ts.train_test_split(test_ratio=0.2)
    X_train, y_train = split['X_train'], split['y_train'].squeeze()
    X_test, y_test = split['X_test'], split['y_test'].squeeze()

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train classical LSTM
    lstm = ClassicalLSTM(input_size=1, hidden_size=20)

    # Make predictions
    y_pred = lstm.predict(X_test)

    # Evaluate
    mse = PredictionEvaluator.mse(y_test, y_pred)
    print(f"\nClassical LSTM MSE: {mse:.4f}")

    # Plot results
    PredictionEvaluator.plot_predictions(
        y_test[:50],
        {'Classical LSTM': y_pred[:50]},
        "Classical LSTM Predictions"
    )

    return lstm, (X_test, y_test)


def lab2_quantum_features():
    """Lab 2: Extract quantum features for time-series."""
    print("\n" + "=" * 60)
    print("LAB 2: Quantum Feature Extraction")
    print("=" * 60)

    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping quantum features lab.")
        return None

    # Create quantum feature map
    qfm = QuantumFeatureMap(n_qubits=4, n_layers=2)

    # Generate sample data
    x1 = np.array([0.1, 0.2, 0.3, 0.4])
    x2 = np.array([0.1, 0.2, 0.3, 0.5])
    x3 = np.array([0.9, 0.8, 0.7, 0.6])

    # Calculate quantum kernels
    k11 = qfm.quantum_kernel(x1, x1)
    k12 = qfm.quantum_kernel(x1, x2)
    k13 = qfm.quantum_kernel(x1, x3)

    print(f"K(x1, x1) = {k11:.4f} (self-similarity)")
    print(f"K(x1, x2) = {k12:.4f} (similar samples)")
    print(f"K(x1, x3) = {k13:.4f} (different samples)")

    return qfm


def lab3_variational_quantum_prediction():
    """Lab 3: Implement variational quantum predictor."""
    print("\n" + "=" * 60)
    print("LAB 3: Variational Quantum Regression")
    print("=" * 60)

    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping variational quantum lab.")
        return None

    # Generate simple prediction task
    n_samples = 20
    X = np.random.rand(n_samples, 4)
    y = np.sum(X, axis=1) / 4  # Average of inputs

    # Create and train VQR
    vqr = VariationalQuantumRegressor(n_qubits=4, n_layers=3)

    # Training loop (simplified)
    print("Training variational quantum circuit...")
    for epoch in range(10):
        loss = vqr.train_step(X, y, learning_rate=0.1)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # Make predictions
    y_pred = vqr.predict(X)
    mse = PredictionEvaluator.mse(y, y_pred)
    print(f"\nFinal MSE: {mse:.4f}")

    return vqr


def lab4_quantum_reservoir():
    """Lab 4: Quantum reservoir computing."""
    print("\n" + "=" * 60)
    print("LAB 4: Quantum Reservoir Computing")
    print("=" * 60)

    # Generate chaotic time-series
    data = generate_synthetic_timeseries(200, pattern='chaotic')
    ts = TimeSeriesData(data, window_size=5, horizon=1)

    # Prepare data
    split = ts.train_test_split(test_ratio=0.3)
    X_train, y_train = split['X_train'], split['y_train'].squeeze()
    X_test, y_test = split['X_test'], split['y_test'].squeeze()

    print(f"Training on chaotic time-series")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Create quantum reservoir
    qr = QuantumReservoir(n_qubits=4, reservoir_size=20)

    # Train readout layer
    print("\nTraining quantum reservoir...")
    qr.fit(X_train[:20], y_train[:20])  # Use subset for speed

    # Make predictions
    y_pred = qr.predict(X_test[:10])

    # Evaluate
    mse = PredictionEvaluator.mse(y_test[:10], y_pred)
    print(f"Quantum Reservoir MSE: {mse:.4f}")

    return qr


def lab5_hybrid_model():
    """Lab 5: Build and evaluate hybrid quantum-classical model."""
    print("\n" + "=" * 60)
    print("LAB 5: Hybrid Quantum-Classical Model")
    print("=" * 60)

    # Generate data with trend
    data = generate_synthetic_timeseries(300, pattern='trend')
    ts = TimeSeriesData(data, window_size=8, horizon=1)

    # Prepare data
    split = ts.train_test_split(test_ratio=0.2)
    X_test, y_test = split['X_test'], split['y_test'].squeeze()

    print(f"Testing hybrid model on trending time-series")
    print(f"Test samples: {len(X_test)}")

    # Create models
    models = {
        'Classical LSTM': ClassicalLSTM(input_size=1, hidden_size=15),
        'Hybrid Quantum-LSTM': HybridQuantumLSTM(input_size=1, hidden_size=10, n_qubits=4)
    }

    # Benchmark
    results = PredictionEvaluator.benchmark_models(models, X_test[:30], y_test[:30])

    # Print results
    print("\nBenchmark Results:")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  MSE:  {metrics['MSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")

    # Plot comparison
    predictions = {name: metrics['predictions'] for name, metrics in results.items()}
    PredictionEvaluator.plot_predictions(
        y_test[:30],
        predictions,
        "Hybrid vs Classical Model Comparison"
    )

    return results


# ============================================================================
# ADVANCED EXERCISES
# ============================================================================

def advanced_exercise_1():
    """Advanced Exercise 1: Multi-step ahead prediction."""
    print("\n" + "=" * 60)
    print("ADVANCED EXERCISE 1: Multi-Step Ahead Forecasting")
    print("=" * 60)

    # TODO: Implement multi-step prediction
    # Hint: Use recursive prediction or direct multi-output approach

    print("Exercise: Implement multi-step ahead prediction")
    print("1. Modify models to predict multiple future time steps")
    print("2. Compare recursive vs direct prediction strategies")
    print("3. Analyze error propagation in multi-step forecasting")


def advanced_exercise_2():
    """Advanced Exercise 2: Real-world data application."""
    print("\n" + "=" * 60)
    print("ADVANCED EXERCISE 2: Real-World Application")
    print("=" * 60)

    print("Exercise: Apply quantum prediction to real data")
    print("1. Load stock market or energy consumption data")
    print("2. Implement data preprocessing pipeline")
    print("3. Train and evaluate quantum-enhanced models")
    print("4. Perform statistical significance testing")
    print("5. Analyze when quantum enhancement provides benefit")


def advanced_exercise_3():
    """Advanced Exercise 3: Custom quantum architecture."""
    print("\n" + "=" * 60)
    print("ADVANCED EXERCISE 3: Design Custom Quantum Architecture")
    print("=" * 60)

    print("Exercise: Design problem-specific quantum circuit")
    print("1. Analyze your time-series characteristics")
    print("2. Design quantum encoding matching data properties")
    print("3. Create problem-inspired ansatz")
    print("4. Implement and benchmark your architecture")
    print("5. Compare with standard variational circuits")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM-ENHANCED PREDICTION LABS")
    print("=" * 60)

    # Check available libraries
    print("\nLibrary Status:")
    print(f"PennyLane: {'Available' if PENNYLANE_AVAILABLE else 'Not Available'}")
    print(f"Qiskit: {'Available' if QISKIT_AVAILABLE else 'Not Available'}")

    # Run labs
    print("\n" + "=" * 60)
    print("RUNNING LAB EXERCISES")
    print("=" * 60)

    # Lab 1: Classical baseline
    lstm_model, test_data = lab1_classical_baseline()

    # Lab 2: Quantum features
    quantum_features = lab2_quantum_features()

    # Lab 3: Variational quantum
    vqr_model = lab3_variational_quantum_prediction()

    # Lab 4: Quantum reservoir
    reservoir_model = lab4_quantum_reservoir()

    # Lab 5: Hybrid model
    hybrid_results = lab5_hybrid_model()

    # Advanced exercises (descriptions only)
    advanced_exercise_1()
    advanced_exercise_2()
    advanced_exercise_3()

    print("\n" + "=" * 60)
    print("LABS COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Quantum features can capture complex patterns")
    print("2. Hybrid models combine strengths of both paradigms")
    print("3. Current quantum advantage is problem-specific")
    print("4. NISQ devices require careful circuit design")
    print("5. Benchmarking against classical is essential")

    print("\nNext Steps:")
    print("- Implement exercises with real data")
    print("- Explore different quantum encodings")
    print("- Optimize circuit depth for NISQ devices")
    print("- Test on actual quantum hardware")
    print("- Contribute to open-source quantum ML libraries")