import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.data as pyg_data
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.transforms import ToUndirected
from torch_cluster import grid_cluster
import networkx as nx
from qiskit import QuantumCircuit, quantum_info
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_algorithms.gradients import ParamShiftSamplerGradient
from qiskit_machine_learning.circuit.library import QNNCircuit
import asyncio
import platform
from typing import Dict, List, Tuple
from collections import deque

# Constants
NUM_STATIC_FEATURES = 3  # DEM, distance to stream, Manning's coefficient
NUM_DYNAMIC_FEATURES = 8  # Water depth, in/out velocities, norms, rainfall
NUM_QUBITS = 8  # Reduced for feasibility
EDGE_THRESHOLD = 0.05  # Lowered to increase active edge detection
PARAMETER_SHIFT = np.pi / 2  # For parameter-shift rule
FPS = 60  # For async control in Pyodide
SUBGRAPH_SIZE = 50  # Target size for subgraphs
MEMORY_SIZE = 10  # Size of replay memory for continual learning
MIN_EDGE_PARAMS = 1  # Minimum number of edge parameters

class FloodDataPipeline:
    """Data pipeline for FloodGNN-GRU dataset with edge-adaptive clustering"""
    def __init__(self):
        self.transform = ToUndirected()
        self.memory = deque(maxlen=MEMORY_SIZE)  # Replay memory for continual learning

    def load_flood_data(self, file_path: str) -> List[pyg_data.Data]:
        """Load FloodGNN-GRU dataset and create subgraphs"""
        try:
            data = np.load(file_path, allow_pickle=True)['harvey']
            graph_list = []

            for sample_idx in range(min(5, len(data))):
                sample = data[sample_idx]
                static_features = torch.tensor(sample['static'], dtype=torch.float32)
                sequence_data = torch.tensor(sample['data'], dtype=torch.float32)
                edge_index = torch.tensor(sample['s_edges'], dtype=torch.long).t()
                binary_data = torch.tensor(sample['bin'], dtype=torch.float32)

                num_nodes = static_features.shape[0]
                num_time_steps = sequence_data.shape[1]
                num_edges = edge_index.shape[1]

                graph_data = pyg_data.Data(
                    x=static_features,
                    edge_index=edge_index,
                    y=sequence_data[:, :, 0],
                    dynamic=sequence_data,
                    binary=binary_data,
                    num_nodes=num_nodes,
                    num_time_steps=num_time_steps,
                    num_edges=num_edges
                )
                graph_data = self.transform(graph_data)
                subgraphs = self.cluster_graph(graph_data, sequence_data)
                graph_list.extend(subgraphs)
                self.memory.extend(subgraphs[:2])  # Store initial subgraphs in memory

            print(f"Loaded {len(graph_list)} subgraphs from {file_path}")
            return graph_list
        except Exception as e:
            print(f"Failed to load data: {e}. Using synthetic data.")
            return self.create_synthetic_flood_data(n_samples=5)

    def create_synthetic_flood_data(self, n_samples: int) -> List[pyg_data.Data]:
        """Create synthetic flood data"""
        graph_list = []
        for _ in range(n_samples):
            num_nodes = np.random.randint(40, 60)
            num_time_steps = np.random.randint(10, 20)
            num_edges = min(200, num_nodes * 4)
            static_features = torch.rand(num_nodes, NUM_STATIC_FEATURES) * 10
            sequence_data = torch.rand(num_nodes, num_time_steps, NUM_DYNAMIC_FEATURES) * 5
            binary_data = torch.randint(0, 2, (num_nodes, num_time_steps, 1)).float()
            edges = torch.randint(0, num_nodes, (num_edges, 2))
            mask = edges[:, 0] != edges[:, 1]
            edge_index = edges[mask].t().contiguous()

            graph_data = pyg_data.Data(
                x=static_features,
                edge_index=edge_index,
                y=sequence_data[:, :, 0],
                dynamic=sequence_data,
                binary=binary_data,
                num_nodes=num_nodes,
                num_time_steps=num_time_steps,
                num_edges=edge_index.shape[1]
            )
            graph_data = self.transform(graph_data)
            subgraphs = self.cluster_graph(graph_data, sequence_data)
            graph_list.extend(subgraphs)
            self.memory.extend(subgraphs[:2])
        print(f"Created {len(graph_list)} synthetic subgraphs")
        return graph_list

    def cluster_graph(self, data: pyg_data.Data, sequence_data: torch.Tensor) -> List[pyg_data.Data]:
        """Cluster graph prioritizing active edges"""
        if data.num_nodes <= SUBGRAPH_SIZE:
            return [data]

        edge_activity = torch.zeros(data.num_edges)
        for t in range(1, min(5, data.num_time_steps)):
            current_dynamic = sequence_data[:, t, :]
            prev_dynamic = sequence_data[:, t - 1, :]
            for edge_idx, (i, j) in enumerate(data.edge_index.t().tolist()):
                rainfall_diff = abs(current_dynamic[i, -1] - prev_dynamic[i, -1])
                edge_activity[edge_idx] += rainfall_diff

        pos = torch.rand(data.num_nodes, 2) * 10
        cluster_ids = grid_cluster(pos, torch.tensor([2.0, 2.0])).long()
        subgraphs = []
        unique_clusters = torch.unique(cluster_ids)

        for cluster in unique_clusters:
            node_mask = cluster_ids == cluster
            num_cluster_nodes = node_mask.sum().item()
            if num_cluster_nodes < 5 or num_cluster_nodes > SUBGRAPH_SIZE * 1.5:
                continue

            subset = torch.where(node_mask)[0]
            subgraph_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
            if subgraph_edge_index.shape[1] == 0:
                continue

            subgraph_data = pyg_data.Data(
                x=data.x[subset],
                edge_index=subgraph_edge_index,
                y=data.y[subset],
                dynamic=data.dynamic[subset],
                binary=data.binary[subset],
                num_nodes=num_cluster_nodes,
                num_time_steps=data.num_time_steps,
                num_edges=subgraph_edge_index.shape[1]
            )
            subgraphs.append(subgraph_data)

        return subgraphs if subgraphs else [data]

    def get_replay_data(self) -> List[pyg_data.Data]:
        """Retrieve data from replay memory"""
        return list(self.memory)

    def to_networkx_graph(self, data: pyg_data.Data) -> nx.Graph:
        """Convert to NetworkX graph"""
        return to_networkx(data, to_undirected=True)

class QuantumFloodCircuit:
    """Quantum circuit with edge-adaptive subgraph encoding"""
    def __init__(self, num_features: int, num_edges: int, active_edges: List[int], num_qubits: int = 8):
        self.num_features = min(num_features, num_qubits)
        self.num_edges = num_edges
        self.active_edges = active_edges
        self.num_qubits = num_qubits
        self.circuit, self.input_params, self.edge_params, self.var_params = self._build_circuit()

    def _build_circuit(self):
        """Build quantum circuit with dynamic edge encoding using QNNCircuit"""
        feature_map = ZZFeatureMap(self.num_features, reps=1)
        ansatz = QuantumCircuit(self.num_qubits)
        
        # Define edge parameters (always 6 for consistency)
        edge_params = [Parameter(f'theta_edge_{i}') for i in range(6)]
        active_edges = self.active_edges if self.active_edges else [0]

        # Apply operations for active edges, up to 6
        for idx in range(6):
            if idx < len(active_edges):
                control_qubit = idx % self.num_qubits
                target_qubit = (idx + 1) % self.num_qubits
                ansatz.cx(control_qubit, target_qubit)
                ansatz.rz(edge_params[idx], target_qubit)
                ansatz.cx(control_qubit, target_qubit)
            else:
                # Dummy operation to include unused edge parameters
                ansatz.rz(edge_params[idx], 0)

        # Apply variational parameters
        var_params = [Parameter(f'var_{i}') for i in range(self.num_qubits)]
        for i, param in enumerate(var_params):
            ansatz.ry(param, i)

        # Use QNNCircuit to combine feature map and ansatz
        qnn_circuit = QNNCircuit(num_qubits=self.num_qubits, feature_map=feature_map, ansatz=ansatz)
        
        # Extract parameters from QNNCircuit
        input_params = qnn_circuit.input_parameters
        weight_params = qnn_circuit.weight_parameters
        edge_params = [p for p in weight_params if p.name.startswith('theta_edge_')]
        var_params = [p for p in weight_params if p.name.startswith('var_')]
        
        return qnn_circuit, input_params, edge_params, var_params

    def compute_state_overlap(self, param_dict: Dict) -> float:
        """Compute overlap with initial state"""
        initial_state = quantum_info.Statevector.from_instruction(QuantumCircuit(self.num_qubits))
        bound_circuit = self.circuit.assign_parameters(param_dict)
        final_state = quantum_info.Statevector.from_instruction(bound_circuit)
        return float(np.abs(initial_state.inner(final_state)))

class QGraphPredictor(nn.Module):
    """Quantum Graph Neural Network for Flood Prediction"""
    def __init__(self, quantum_circuit: QuantumFloodCircuit):
        super().__init__()
        self.quantum_circuit = quantum_circuit
        self.circuit = quantum_circuit.circuit
        self.input_params = quantum_circuit.input_params
        self.edge_params = quantum_circuit.edge_params
        self.var_params = quantum_circuit.var_params
        self.weights = nn.Parameter(torch.randn(len(self.edge_params) + len(self.var_params)) * 0.1)
        self.qnn = None  # Initialize QNN in forward pass to ensure fresh parameters
        input_size = len(self.input_params)
        self.classical_backup = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.fisher_information = {}
        self.optimal_params = {}
        self.lambda_ewc = 0.1
        self.task_count = 0
        self.transparency_engine = None  # Reference to QVISMTransparencyEngine for updating initial_params

    def set_transparency_engine(self, transparency_engine):
        """Set reference to QVISMTransparencyEngine for parameter updates"""
        self.transparency_engine = transparency_engine

    def forward(self, data: pyg_data.Data, time_step: int, active_edges: List[int]) -> Tuple[torch.Tensor, float]:
        """Forward pass with dynamic circuit"""
        static_features = data.x
        dynamic_features = data.dynamic[:, time_step, :]
        node_features = torch.cat([static_features, dynamic_features], dim=-1)
        inputs = node_features.mean(dim=0)[:len(self.input_params)]

        # Rebuild circuit for current active edges to ensure parameter consistency
        self.quantum_circuit = QuantumFloodCircuit(
            num_features=len(node_features[0]),
            num_edges=data.num_edges,
            active_edges=active_edges,
            num_qubits=NUM_QUBITS
        )
        self.circuit = self.quantum_circuit.circuit
        self.input_params = self.quantum_circuit.input_params
        self.edge_params = self.quantum_circuit.edge_params
        self.var_params = self.quantum_circuit.var_params

        # Resize weights if necessary
        expected_weight_size = len(self.edge_params) + len(self.var_params)
        if expected_weight_size != len(self.weights):
            old_weights = self.weights.data.clone()
            self.weights = nn.Parameter(torch.randn(expected_weight_size) * 0.1)
            min_len = min(len(old_weights), len(self.weights))
            self.weights.data[:min_len] = old_weights[:min_len]
            old_fisher = self.fisher_information.copy()
            old_optimal = self.optimal_params.copy()
            self.fisher_information = {i: old_fisher.get(i, 0.0) for i in range(len(self.weights))}
            self.optimal_params = {i: old_optimal.get(i, 0.0) for i in range(len(self.weights))}
            if self.transparency_engine and hasattr(self.transparency_engine, 'initial_params'):
                old_initial_params = self.transparency_engine.initial_params.copy()
                self.transparency_engine.initial_params = np.zeros(len(self.weights))
                min_len_params = min(len(old_initial_params), len(self.weights))
                self.transparency_engine.initial_params[:min_len_params] = old_initial_params[:min_len_params]

        # Initialize SamplerQNN with fresh circuit
        try:
            self.qnn = SamplerQNN(
                circuit=self.circuit,
                sampler=StatevectorSampler(),
                input_gradients=True,
                gradient=ParamShiftSamplerGradient(sampler=StatevectorSampler()),
                sparse=False
            )
        except Exception as e:
            print(f"SamplerQNN initialization failed: {e}. Using classical backup.")
            self.qnn = None

        if self.qnn:
            try:
                input_values = np.array([inputs[:len(self.input_params)].detach().numpy()])
                weight_values = np.array([self.weights[:len(self.edge_params + self.var_params)].detach().numpy()])
                
                result = self.qnn.forward(input_values, weight_values)

                # Create parameter dictionary for state overlap
                param_dict = {p: v for p, v in zip(self.input_params, input_values[0])}
                param_dict.update({p: v for p, v in zip(self.edge_params + self.var_params, weight_values[0])})
                
                overlap = self.quantum_circuit.compute_state_overlap(param_dict)
                return torch.tensor(result[0, 0], dtype=torch.float32), overlap

            except Exception as e:
                print(f"Quantum forward failed: {e}. Using classical backup.")

        return self.classical_backup(inputs).squeeze(), 0.0

    def parameter_shift_gradient(self, data: pyg_data.Data, time_step: int, param_idx: int, active_edges: List[int]) -> float:
        """Compute gradient using parameter-shift rule"""
        if param_idx >= len(self.weights):
            return 0.0
        original_weight = self.weights[param_idx].item()
        targets = data.y[:, time_step].mean()

        self.weights.data[param_idx] = original_weight + PARAMETER_SHIFT
        with torch.no_grad():
            output_plus, _ = self.forward(data, time_step, active_edges)
            loss_plus = (output_plus - targets).pow(2).item()

        self.weights.data[param_idx] = original_weight - PARAMETER_SHIFT
        with torch.no_grad():
            output_minus, _ = self.forward(data, time_step, active_edges)
            loss_minus = (output_minus - targets).pow(2).item()

        self.weights.data[param_idx] = original_weight
        return (loss_plus - loss_minus) / (2 * PARAMETER_SHIFT)

    def identify_active_edges(self, data: pyg_data.Data, time_step: int) -> List[int]:
        """Identify edges with significant dynamic changes"""
        if time_step == 0:
            return list(range(min(data.num_edges, NUM_QUBITS - 1)))
        active_edges = []
        current_dynamic = data.dynamic[:, time_step, :]
        prev_dynamic = data.dynamic[:, time_step - 1, :]
        for edge_idx, (i, j) in enumerate(data.edge_index.t().tolist()):
            rainfall_diff = abs(current_dynamic[i, -1] - prev_dynamic[i, -1])
            if rainfall_diff > EDGE_THRESHOLD:
                active_edges.append(edge_idx)
        # Ensure at least one active edge to prevent empty circuit
        if not active_edges and data.num_edges > 0:
            active_edges = [0]
        return active_edges

    def graph_update_routine(self, data: pyg_data.Data, time_step: int, learning_rate: float = 0.01) -> Tuple[torch.Tensor, float]:
        """Edge-adaptive update routine with Q-EWC"""
        active_edges = self.identify_active_edges(data, time_step)
        for edge_idx in active_edges:
            if edge_idx < len(self.weights):
                grad = self.parameter_shift_gradient(data, time_step, edge_idx, active_edges)
                importance = self.fisher_information.get(edge_idx, 0.0)
                optimal = self.optimal_params.get(edge_idx, 0.0)
                with torch.no_grad():
                    self.weights[edge_idx] -= learning_rate * (grad + self.lambda_ewc * importance * (self.weights[edge_idx] - optimal))
        return self.forward(data, time_step, active_edges)

    def update_continual_learning(self, data_list: List[pyg_data.Data]):
        """Update Fisher Information and optimal parameters for continual learning"""
        old_fisher = self.fisher_information.copy()
        self.compute_fisher_information(data_list)
        for i in range(len(self.weights)):
            self.fisher_information[i] = (self.task_count * old_fisher.get(i, 0.0) + self.fisher_information.get(i, 0.0)) / (self.task_count + 1)
        self.store_optimal_parameters()
        self.task_count += 1

    def compute_fisher_information(self, data_list: List[pyg_data.Data]):
        """Compute Fisher Information for Q-EWC"""
        fisher = {i: 0.0 for i in range(len(self.weights))}
        n_samples = 0
        for data in data_list:
            for t in range(min(3, data.num_time_steps)):
                active_edges = self.identify_active_edges(data, t)
                for param_idx in range(len(self.weights)):
                    grad = self.parameter_shift_gradient(data, t, param_idx, active_edges)
                    fisher[param_idx] += grad ** 2
                n_samples += 1
        for param_idx in fisher:
            fisher[param_idx] /= n_samples if n_samples > 0 else 1
        self.fisher_information = fisher

    def store_optimal_parameters(self):
        """Store current parameters for Q-EWC"""
        self.optimal_params = {i: self.weights[i].item() for i in range(len(self.weights))}

class QVISMTransparencyEngine:
    """QVISM: Quantum Variational Interpretability and Sensitivity Mapping"""
    def __init__(self, predictor: QGraphPredictor):
        self.predictor = predictor
        self.predictor.set_transparency_engine(self)  # Set reference in predictor

    def generate_transparency_report(self, data: pyg_data.Data, time_step: int) -> Dict:
        """Generate QVISM transparency report"""
        edge_influences = self._compute_edge_influences(data, time_step)
        node_sensitivities = self._compute_node_sensitivities(data, time_step)
        temporal_drift = self._analyze_temporal_drift()
        predictions, hilbert_overlap = self.predictor.graph_update_routine(data, time_step)
        avg_flood_risk = predictions.item() if predictions.numel() == 1 else predictions.mean().item()

        return {
            "timestamp": f"Time step {time_step}",
            "average_flood_risk": avg_flood_risk,
            "risk_level": "HIGH" if avg_flood_risk > 0.7 else "MEDIUM" if avg_flood_risk > 0.4 else "LOW",
            "edge_influences": edge_influences,
            "node_sensitivities": node_sensitivities,
            "temporal_drift": temporal_drift,
            "active_edges": len(self.predictor.identify_active_edges(data, time_step)),
            "hilbert_space_overlap": hilbert_overlap
        }

    def _compute_edge_influences(self, data: pyg_data.Data, time_step: int) -> Dict[int, float]:
        """Compute edge influences"""
        influences = {}
        active_edges = self.predictor.identify_active_edges(data, time_step)
        baseline_pred, _ = self.predictor.graph_update_routine(data, time_step)
        baseline_pred = baseline_pred.item()
        for edge_idx in range(min(5, len(self.predictor.edge_params))):
            original_weight = self.predictor.weights[edge_idx].item()
            self.predictor.weights.data[edge_idx] = 0.0
            modified_pred, _ = self.predictor.graph_update_routine(data, time_step)
            influences[edge_idx] = abs(baseline_pred - modified_pred.item())
            self.predictor.weights.data[edge_idx] = original_weight
        return influences

    def _compute_node_sensitivities(self, data: pyg_data.Data, time_step: int) -> Dict[int, float]:
        """Compute node sensitivities"""
        sensitivities = {}
        active_edges = self.predictor.identify_active_edges(data, time_step)
        for node_idx in range(min(5, data.num_nodes)):
            original_dynamic = data.dynamic[node_idx, time_step, :].clone()
            data.dynamic[node_idx, time_step, :] += 0.1
            perturbed_pred, _ = self.predictor.graph_update_routine(data, time_step)
            original_pred, _ = self.predictor.graph_update_routine(data, time_step)
            sensitivities[node_idx] = abs(perturbed_pred.item() - original_pred.item()) / 0.1
            data.dynamic[node_idx, time_step, :] = original_dynamic
        return sensitivities

    def _analyze_temporal_drift(self) -> float:
        """Analyze parameter drift"""
        if not hasattr(self, 'initial_params'):
            self.initial_params = self.predictor.weights.detach().numpy().copy()
            return 0.0
        current_weights = self.predictor.weights.detach().numpy()
        min_len = min(len(current_weights), len(self.initial_params))
        drift = np.mean(np.abs(current_weights[:min_len] - self.initial_params[:min_len]))
        return float(drift)

async def main():
    """Main execution for Q-Graph"""
    data_pipeline = FloodDataPipeline()
    train_data = data_pipeline.load_flood_data('/Users/santoshramesh/Desktop/NASA-BTA-ESTO/data/FloodGNN-GRU/train.npz')
    
    for sample_idx, data in enumerate(train_data):
        num_features = data.x.shape[1] + data.dynamic.shape[2]
        quantum_circuit = QuantumFloodCircuit(num_features, data.num_edges, active_edges=[])
        predictor = QGraphPredictor(quantum_circuit)
        qvism = QVISMTransparencyEngine(predictor)
        
        print(f"\nProcessing Sample {sample_idx} (Nodes: {data.num_nodes}, Edges: {data.num_edges}, Time Steps: {data.num_time_steps})")
        for t in range(min(20, data.num_time_steps)):
            report = qvism.generate_transparency_report(data, t)
            print(f"Time step {t}: Flood Risk = {report['average_flood_risk']:.4f} ({report['risk_level']}), "
                  f"Active Edges = {report['active_edges']}, Hilbert Overlap = {report['hilbert_space_overlap']:.4f}")

        predictor.update_continual_learning([data] + data_pipeline.get_replay_data())
        data_pipeline.memory.append(data)

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())