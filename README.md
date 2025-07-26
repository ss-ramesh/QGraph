# QGraph
NASA ESTO Beyond the Algorithm Challenge

# Overview
Q-Graph combines quantum computing with graph neural networks to predict flood events with unprecedented precision. The system can generate specific predictions like:
"Grid cell 4 is expected to flood in 22 minutes with a predicted water depth of 5.1 meters."

# Data Download
The data can be found at "https://zenodo.org/records/10787632?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjRmZjFlYjM5LWI0NTItNDVlMy04OTRjLWNiY2M4YTgwZmFlNSIsImRhdGEiOnt9LCJyYW5kb20iOiJmOGI0NDMxNTg2NmMxNzg3YWE3YmQxZmYzMTE2ODRiZSJ9.CFvVViSDO4_Q8CR7mZ5zPzl0qTTYNlRvLs1Li1hbwh80Sz_C1F8pViPXvuToHRJuIK6McjoMuU631q64h-TXtw" and should be placed in the 'data' folder (created by user) after download.

# Quantum Circuit Design
The quantum circuit employs:

ZZFeatureMap: Encodes input features into quantum states
Edge-adaptive encoding: Dynamic circuit topology based on active edges
Variational parameters: Trainable quantum gates
Parameter-shift rule: Quantum gradient computation

# Data Flow
Input: Graph data with static (DEM, Manning's coefficient) and dynamic (rainfall, water depth) features
Preprocessing: Graph clustering and edge activity analysis
Quantum Processing: Feature encoding and variational circuit execution
Prediction: Multi-step flood probability and depth forecasting
Output: Human-readable flood alerts with specific timing and depths

# Core Components
┌─────────────────────────────────────────────────────────────┐
│                    Q-Graph Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ FloodDataPipeline│    │QuantumFloodCircuit              │
│  │                 │    │                 │                │
│  │ • Data loading  │    │ • ZZFeatureMap  │                │
│  │ • Graph clustering   │ • Edge encoding │                │
│  │ • Replay memory │    │ • Variational   │                │
│  │ • Synthetic data│    │   parameters    │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                        │                       │
│           ▼                        ▼                       │
│  ┌─────────────────────────────────────────────────────────┤
│  │           EnhancedQGraphPredictor                       │
│  │                                                         │
│  │ • Quantum circuit execution                             │
│  │ • Classical backup model                                │
│  │ • Multi-step flood prediction                           │
│  │ • Depth prediction head                                 │
│  │ • Continual learning (Q-EWC)                            │
│  └─────────────────────────────────────────────────────────┤
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┤
│  │         EnhancedQVISMTransparencyEngine                 │
│  │                                                         │
│  │ • Flood alert generation                                │
│  │ • Risk assessment and prioritization                    │
│  │ • Edge influence analysis                               │
│  │ • Node sensitivity mapping                              │
│  │ • Temporal drift analysis                               │
│  └─────────────────────────────────────────────────────────┘
│                                                             │
└─────────────────────────────────────────────────────────────┘

# Requirements
pip install torch torch-geometric
pip install qiskit qiskit-machine-learning
pip install numpy networkx
pip install torch-cluster 

# Usage
Run model.py