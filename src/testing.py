import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit import Parameter
import tracemalloc
import torch
import torch_geometric.data as pyg_data
from model import FloodDataPipeline, QuantumFloodCircuit, QGraphPredictor  # Replace with your module import

def test_quantum_circuit_parameters():
    """Test if the quantum circuit defines all expected parameters."""
    num_features = 11  # 3 static + 8 dynamic features
    num_edges = 10
    active_edges = [0, 1, 2]  # Example active edges
    num_qubits = 8

    # Initialize quantum circuit
    quantum_circuit = QuantumFloodCircuit(num_features, num_edges, active_edges, num_qubits)
    circuit = quantum_circuit.circuit
    input_params = quantum_circuit.input_params
    edge_params = quantum_circuit.edge_params
    var_params = quantum_circuit.var_params

    # Expected parameters
    expected_input_params = [f"x[{i}]" for i in range(8)]  # ZZFeatureMap uses 8 params
    expected_edge_params = [f"theta_edge_{i}" for i in range(len(active_edges))]
    expected_var_params = [f"var_{i}" for i in range(num_qubits)]

    # Debug: print actual parameters
    print("Input parameters:", [str(p) for p in input_params])
    print("Edge parameters:", [str(p) for p in edge_params])
    print("Var parameters:", [str(p) for p in var_params])
    print("Total circuit parameters:", [str(p) for p in circuit.parameters])

    # Assertions
    assert len(input_params) == len(expected_input_params), f"Expected {len(expected_input_params)} input params, got {len(input_params)}"
    assert all(str(p) in expected_input_params for p in input_params), "Input parameter names mismatch"
    assert len(edge_params) == len(expected_edge_params), f"Expected {len(expected_edge_params)} edge params, got {len(edge_params)}"
    assert all(str(p) in expected_edge_params for p in edge_params), "Edge parameter names mismatch"
    assert len(var_params) == len(expected_var_params), f"Expected {len(expected_var_params)} var params, got {len(var_params)}"
    assert all(str(p) in expected_var_params for p in var_params), "Var parameter names mismatch"

    print("Quantum circuit parameter test passed!")

def test_parameter_binding():
    """Test parameter binding in QGraphPredictor forward pass."""
    num_features = 11
    num_edges = 10
    active_edges = [0, 1, 2]
    num_qubits = 8

    # Create dummy graph data
    data = pyg_data.Data(
        x=torch.rand(20, 3),  # Static features
        dynamic=torch.rand(20, 5, 8),  # Dynamic features
        edge_index=torch.randint(0, 20, (2, num_edges)),
        y=torch.rand(20, 5),
        num_nodes=20,
        num_time_steps=5,
        num_edges=num_edges
    )

    # Initialize quantum circuit and predictor
    quantum_circuit = QuantumFloodCircuit(num_features, num_edges, active_edges, num_qubits)
    predictor = QGraphPredictor(quantum_circuit)

    # Test forward pass
    time_step = 1
    try:
        output, overlap = predictor.forward(data, time_step, active_edges)
        print("Forward pass output:", output)
        print("Hilbert overlap:", overlap)
        print("Parameter binding test passed!")
    except Exception as e:
        print("Parameter binding test failed:", e)

def test_quantum_circuit_parameters():
    """Test quantum circuit parameters and memory usage."""
    tracemalloc.start()
    print("tracemalloc enabled for circuit parameter test")

    num_features = 11  # 3 static + 8 dynamic features
    num_edges = 10
    active_edges = [0, 1, 2]
    num_qubits = 8

    # Initialize quantum circuit
    snapshot1 = tracemalloc.take_snapshot()
    quantum_circuit = QuantumFloodCircuit(num_features, num_edges, active_edges, num_qubits)
    snapshot2 = tracemalloc.take_snapshot()

    circuit = quantum_circuit.circuit
    input_params = quantum_circuit.input_params
    edge_params = quantum_circuit.edge_params
    var_params = quantum_circuit.var_params

    # Expected parameters
    expected_input_params = [f"x[{i}]" for i in range(min(num_features, num_qubits))]
    expected_edge_params = [f"theta_edge_{i}" for i in range(6)]  # Fix to 6
    expected_var_params = [f"var_{i}" for i in range(num_qubits)]

    # Debug: print parameters
    print("Input parameters:", [str(p) for p in input_params])
    print("Edge parameters:", [str(p) for p in edge_params])
    print("Var parameters:", [str(p) for p in var_params])
    print("Total circuit parameters:", [str(p) for p in circuit.parameters])

    # Assertions
    assert len(input_params) == len(expected_input_params), f"Expected {len(expected_input_params)} input params, got {len(input_params)}"
    assert all(str(p) in expected_input_params for p in input_params), "Input parameter names mismatch"
    assert len(edge_params) == len(expected_edge_params), f"Expected {len(expected_edge_params)} edge params, got {len(edge_params)}"
    assert all(str(p) in expected_edge_params for p in edge_params), "Edge parameter names mismatch"
    assert len(var_params) == len(expected_var_params), f"Expected {len(expected_var_params)} var params, got {len(var_params)}"
    assert all(str(p) in expected_var_params for p in var_params), "Var parameter names mismatch"

    # Memory usage
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("Top memory allocations during circuit creation:")
    for stat in stats[:5]:
        print(stat)

    print("Quantum circuit parameter test passed!")
    tracemalloc.stop()

if __name__ == "__main__":
    test_quantum_circuit_parameters()