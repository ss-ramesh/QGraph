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

class EnhancedQGraphPredictor(QGraphPredictor):
    """Enhanced Q-Graph with specific flood timing and depth prediction"""
    
    def __init__(self, quantum_circuit: QuantumFloodCircuit, flood_threshold: float = 2.0):
        super().__init__(quantum_circuit)
        self.flood_threshold = flood_threshold  # meters
        # Add depth prediction head with improved architecture
        input_size = len(self.input_params) if hasattr(self, 'input_params') else 8
        self.depth_predictor = nn.Sequential(
            nn.Linear(input_size, 64),  # Increased capacity
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # Ensure positive depth
        )
        
        # Enhanced confidence calculation components
        self.confidence_weights = nn.Parameter(torch.ones(3) * 0.33)  # Ensemble weights
        self.temporal_consistency_factor = 0.8  # Weight for temporal consistency
        
    def predict_flood_event(self, data: pyg_data.Data, current_time: int, 
                           node_id: int, max_lookahead: int = 30) -> Dict:
        """Predict specific flood event for a node with enhanced confidence"""
        
        # Ensure node_id is within bounds
        if node_id >= data.num_nodes:
            node_id = min(node_id, data.num_nodes - 1)
        
        # Multi-step prediction with ensemble
        predicted_depths = []
        flood_probabilities = []
        quantum_confidences = []
        
        for future_step in range(current_time, min(current_time + max_lookahead, data.num_time_steps)):
            # Get active edges for this time step
            active_edges = self.identify_active_edges(data, future_step)
            
            # Ensemble prediction for flood probability
            quantum_prob, overlap = self.forward(data, future_step, active_edges)
            
            # Classical prediction for comparison
            static_features = data.x[node_id]
            dynamic_features = data.dynamic[node_id, future_step, :]
            node_features = torch.cat([static_features, dynamic_features])
            
            # Ensure input size matches
            input_size = self.classical_backup[0].in_features
            classical_input = node_features[:input_size]
            if len(classical_input) < input_size:
                padding = torch.zeros(input_size - len(classical_input))
                classical_input = torch.cat([classical_input, padding])
            
            classical_prob = torch.sigmoid(self.classical_backup(classical_input))
            
            # Ensemble combination with learned weights
            weights = torch.softmax(self.confidence_weights, dim=0)
            ensemble_prob = (weights[0] * torch.sigmoid(quantum_prob) + 
                           weights[1] * classical_prob + 
                           weights[2] * overlap)
            
            flood_probabilities.append(ensemble_prob.item())
            
            # Enhanced depth prediction with node-specific features
            depth_input = node_features[:self.depth_predictor[0].in_features]
            if len(depth_input) < self.depth_predictor[0].in_features:
                padding = torch.zeros(self.depth_predictor[0].in_features - len(depth_input))
                depth_input = torch.cat([depth_input, padding])
            
            predicted_depth = self.depth_predictor(depth_input)
            predicted_depths.append(predicted_depth.item())
            
            # Store quantum confidence metrics
            quantum_confidences.append(overlap)
        
        # Enhanced confidence calculation
        confidence = self._calculate_enhanced_confidence(
            flood_probabilities, predicted_depths, quantum_confidences, current_time
        )
        
        # Find when flooding occurs with enhanced criteria
        flood_time = None
        flood_depth = None
        
        for i, (depth, prob) in enumerate(zip(predicted_depths, flood_probabilities)):
            # Enhanced flood detection criteria
            depth_threshold = depth > self.flood_threshold
            prob_threshold = prob > 0.3  # Lowered threshold
            confidence_threshold = confidence > 0.2  # Additional confidence check
            
            if depth_threshold and prob_threshold and confidence_threshold:
                flood_time = i  # Time steps from current
                flood_depth = depth
                break
        
        return {
            'node_id': node_id,
            'current_time_step': current_time,
            'will_flood': flood_time is not None,
            'time_to_flood_steps': flood_time,
            'time_to_flood_minutes': flood_time * 5 if flood_time else None,  # Assuming 5 min per step
            'predicted_depth_meters': flood_depth,
            'confidence': confidence,
            'depth_progression': predicted_depths,
            'probability_progression': flood_probabilities,
            'quantum_overlap_progression': quantum_confidences
        }
    
    def _calculate_enhanced_confidence(self, probabilities: List[float], depths: List[float], 
                                     quantum_overlaps: List[float], current_time: int) -> float:
        """Calculate enhanced confidence score using multiple factors"""
        if not probabilities or not depths:
            return 0.0
        
        # Factor 1: Prediction consistency (temporal smoothness)
        if len(probabilities) > 1:
            prob_consistency = 1.0 - np.mean([abs(probabilities[i] - probabilities[i-1]) 
                                            for i in range(1, len(probabilities))])
            depth_consistency = 1.0 - np.mean([abs(depths[i] - depths[i-1]) 
                                             for i in range(1, len(depths))])
        else:
            prob_consistency = depth_consistency = 0.5
        
        # Factor 2: Quantum coherence (average overlap)
        quantum_coherence = np.mean(quantum_overlaps) if quantum_overlaps else 0.0
        
        # Factor 3: Prediction magnitude (higher values = more confident)
        max_prob = max(probabilities)
        max_depth = max(depths)
        magnitude_factor = min(1.0, (max_prob + max_depth / 10.0) / 2.0)
        
        # Factor 4: Early prediction bonus (earlier predictions are less certain)
        time_factor = min(1.0, current_time / 10.0)  # Confidence increases with time
        
        # Weighted combination
        confidence = (0.3 * prob_consistency + 
                     0.2 * depth_consistency + 
                     0.2 * quantum_coherence + 
                     0.2 * magnitude_factor + 
                     0.1 * time_factor)
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

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

class EnhancedQVISMTransparencyEngine(QVISMTransparencyEngine):
    """Enhanced QVISM with specific flood event reporting"""
    
    def __init__(self, predictor: EnhancedQGraphPredictor):
        super().__init__(predictor)
        
    def generate_flood_alert(self, data: pyg_data.Data, current_time: int, 
                           node_id: int = None) -> str:
        """Generate human-readable flood alert"""
        
        if node_id is None:
            # Find highest risk node
            risk_scores = []
            for i in range(min(10, data.num_nodes)):  # Check first 10 nodes
                prediction = self.predictor.predict_flood_event(data, current_time, i)
                risk_score = prediction['confidence'] if prediction['will_flood'] else 0.0
                risk_scores.append((i, risk_score))
            
            if not risk_scores or max(risk_scores, key=lambda x: x[1])[1] == 0:
                return "No immediate flood risk detected in monitored areas."
            
            node_id = max(risk_scores, key=lambda x: x[1])[0]
        
        prediction = self.predictor.predict_flood_event(data, current_time, node_id)
        
        if prediction['will_flood']:
            return (f"**Grid cell {node_id}** is expected to flood in "
                   f"**{prediction['time_to_flood_minutes']} minutes** with a predicted "
                   f"water depth of **{prediction['predicted_depth_meters']:.1f} meters**. "
                   f"(Confidence: {prediction['confidence']:.2f})")
        else:
            current_depth = prediction['depth_progression'][0] if prediction['depth_progression'] else 0
            return (f"**Grid cell {node_id}** is not expected to flood in the next "
                   f"{len(prediction['depth_progression']) * 5} minutes. "
                   f"Current predicted depth: {current_depth:.1f} meters.")
    
    def generate_detailed_flood_report(self, data: pyg_data.Data, current_time: int) -> Dict:
        """Generate detailed flood report for all high-risk nodes"""
        high_risk_nodes = []
        
        for node_id in range(min(data.num_nodes, 20)):  # Check up to 20 nodes
            prediction = self.predictor.predict_flood_event(data, current_time, node_id)
            if prediction['will_flood'] or prediction['confidence'] > 0.3:
                high_risk_nodes.append(prediction)
        
        # Sort by urgency (time to flood, then confidence)
        high_risk_nodes.sort(key=lambda x: (
            x['time_to_flood_minutes'] if x['will_flood'] else float('inf'),
            -x['confidence']
        ))
        
        return {
            'current_time_step': current_time,
            'total_monitored_nodes': min(data.num_nodes, 20),
            'high_risk_count': len(high_risk_nodes),
            'immediate_threats': [n for n in high_risk_nodes if n['will_flood'] and n['time_to_flood_minutes'] <= 30],
            'emerging_risks': [n for n in high_risk_nodes if not n['will_flood'] and n['confidence'] > 0.5],
            'all_predictions': high_risk_nodes
        }

async def main():
    """Main execution for Enhanced Q-Graph with improved detailed analysis"""
    data_pipeline = FloodDataPipeline()
    file_path = input ("Input the full path to the FloodGNN-GRU dataset (e.g., '/path/to/train.npz'): ")
    train_data = data_pipeline.load_flood_data(file_path)
    
    for sample_idx, data in enumerate(train_data):
        num_features = data.x.shape[1] + data.dynamic.shape[2]
        quantum_circuit = QuantumFloodCircuit(num_features, data.num_edges, active_edges=[])
        predictor = EnhancedQGraphPredictor(quantum_circuit, flood_threshold=2.0)
        qvism = EnhancedQVISMTransparencyEngine(predictor)
        
        print(f"\n{'='*80}")
        print(f"PROCESSING SAMPLE {sample_idx}")
        print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Time Steps: {data.num_time_steps}")
        print(f"{'='*80}")
        
        # Generate specific flood alerts every few time steps
        for t in range(0, min(20, data.num_time_steps), 3):
            print(f"\n--- TIME STEP {t} ---")
            
            # Generate main flood alert
            alert = qvism.generate_flood_alert(data, t)
            print(f"FLOOD ALERT: {alert}")
            
            # Generate detailed report for high-risk areas
            detailed_report = qvism.generate_detailed_flood_report(data, t)
            
            if detailed_report['immediate_threats']:
                print(f"\n⚠️  IMMEDIATE THREATS ({len(detailed_report['immediate_threats'])}):")
                for threat in detailed_report['immediate_threats'][:3]:  # Show top 3
                    print(f"   • Grid cell {threat['node_id']}: {threat['time_to_flood_minutes']}min, "
                          f"{threat['predicted_depth_meters']:.1f}m depth")
            
            if detailed_report['emerging_risks']:
                print(f"\n📊 EMERGING RISKS ({len(detailed_report['emerging_risks'])}):")
                for risk in detailed_report['emerging_risks'][:3]:  # Show top 3
                    current_depth = risk['depth_progression'][0] if risk['depth_progression'] else 0
                    print(f"   • Grid cell {risk['node_id']}: {risk['confidence']:.2f} confidence, "
                          f"{current_depth:.1f}m current depth")
            
            # Show detailed prediction for specific node (node 4 if exists) - NOW SHOWS AT EVERY TIME STEP
            if data.num_nodes > 4:
                print(f"\n🔍 DETAILED ANALYSIS - Grid Cell 4:")
                detailed_pred = predictor.predict_flood_event(data, t, node_id=4)
                if detailed_pred['will_flood']:
                    print(f"   **Grid cell 4** is expected to flood in "
                          f"**{detailed_pred['time_to_flood_minutes']} minutes** with a predicted "
                          f"water depth of **{detailed_pred['predicted_depth_meters']:.1f} meters**.")
                else:
                    current_depth = detailed_pred['depth_progression'][0] if detailed_pred['depth_progression'] else 0
                    print(f"   Grid cell 4 is not expected to flood in the next "
                          f"{len(detailed_pred['depth_progression']) * 5} minutes. "
                          f"Current predicted depth: {current_depth:.1f} meters.")
                
                print(f"   Enhanced Confidence: {detailed_pred['confidence']:.3f}")
                print(f"   Depth progression (next 6 steps): {[f'{d:.1f}m' for d in detailed_pred['depth_progression'][:6]]}")
                print(f"   Probability progression: {[f'{p:.3f}' for p in detailed_pred['probability_progression'][:6]]}")
                print(f"   Quantum coherence: {[f'{q:.3f}' for q in detailed_pred['quantum_overlap_progression'][:6]]}")
            
            # Original QVISM transparency report
            transparency_report = qvism.generate_transparency_report(data, t)
            print(f"\n📈 QVISM ANALYSIS:")
            print(f"   Risk Level: {transparency_report['risk_level']}")
            print(f"   Active Edges: {transparency_report['active_edges']}")
            print(f"   Hilbert Overlap: {transparency_report['hilbert_space_overlap']:.4f}")
            print(f"   Temporal Drift: {transparency_report['temporal_drift']:.4f}")

        # Update continual learning
        predictor.update_continual_learning([data] + data_pipeline.get_replay_data())
        data_pipeline.memory.append(data)
        
        print(f"\n✅ Sample {sample_idx} processing complete. Updated continual learning parameters.")

async def enhanced_main():
    """Enhanced main execution with specific flood predictions"""
    data_pipeline = FloodDataPipeline()
    file_path = input ("Input the full path to the FloodGNN-GRU dataset (e.g., '/path/to/train.npz'): ")
    train_data = data_pipeline.load_flood_data(file_path)
    
    for sample_idx, data in enumerate(train_data):
        num_features = data.x.shape[1] + data.dynamic.shape[2]
        quantum_circuit = QuantumFloodCircuit(num_features, data.num_edges, active_edges=[])
        predictor = EnhancedQGraphPredictor(quantum_circuit, flood_threshold=2.0)
        qvism = EnhancedQVISMTransparencyEngine(predictor)
        
        print(f"\nProcessing Sample {sample_idx} (Nodes: {data.num_nodes}, Edges: {data.num_edges})")
        
        # Generate specific flood alerts every few time steps with detailed analysis
        for t in range(0, min(20, data.num_time_steps), 5):
            alert = qvism.generate_flood_alert(data, t)
            print(f"Time step {t}: {alert}")
            
            # Show detailed prediction for specific node at every time step
            node_to_check = min(4, data.num_nodes - 1)
            detailed_pred = predictor.predict_flood_event(data, t, node_id=node_to_check)
            if detailed_pred['will_flood']:
                print(f"  Detailed: Node {node_to_check} flood in {detailed_pred['time_to_flood_minutes']}min, "
                      f"depth {detailed_pred['predicted_depth_meters']:.1f}m, confidence {detailed_pred['confidence']:.3f}")
            else:
                current_depth = detailed_pred['depth_progression'][0] if detailed_pred['depth_progression'] else 0
                print(f"  Detailed: Node {node_to_check} no flood expected, "
                      f"current depth {current_depth:.1f}m, confidence {detailed_pred['confidence']:.3f}")

        predictor.update_continual_learning([data] + data_pipeline.get_replay_data())
        data_pipeline.memory.append(data)

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())