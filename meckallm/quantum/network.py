import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import networkx as nx

@dataclass
class QuantumNode:
    id: int
    state: np.ndarray
    connections: List[int]
    coherence: float
    entanglement: float

@dataclass
class QuantumEdge:
    source: int
    target: int
    weight: float
    phase: float
    entanglement: float

class QuantumNetwork:
    def __init__(
        self,
        num_nodes: int,
        connection_probability: float = 0.3,
        max_connections: int = 5
    ):
        self.num_nodes = num_nodes
        self.connection_probability = connection_probability
        self.max_connections = max_connections
        
        # Initialize network
        self.nodes: Dict[int, QuantumNode] = {}
        self.edges: Dict[Tuple[int, int], QuantumEdge] = {}
        self.graph = nx.Graph()
        
        self.initialize_network()
        
    def initialize_network(self):
        """
        Initialize quantum network with nodes and connections
        """
        # Create nodes
        for i in range(self.num_nodes):
            self.nodes[i] = QuantumNode(
                id=i,
                state=np.zeros(2**self.num_nodes, dtype=np.complex128),
                connections=[],
                coherence=1.0,
                entanglement=0.0
            )
            self.graph.add_node(i)
            
        # Create edges
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if (
                    np.random.random() < self.connection_probability and
                    len(self.nodes[i].connections) < self.max_connections and
                    len(self.nodes[j].connections) < self.max_connections
                ):
                    self.add_edge(i, j)
                    
    def add_edge(self, source: int, target: int):
        """
        Add quantum edge between nodes
        """
        # Create edge
        edge = QuantumEdge(
            source=source,
            target=target,
            weight=np.random.random(),
            phase=np.random.random() * 2 * np.pi,
            entanglement=0.0
        )
        
        # Add to network
        self.edges[(source, target)] = edge
        self.edges[(target, source)] = edge
        
        # Update node connections
        self.nodes[source].connections.append(target)
        self.nodes[target].connections.append(source)
        
        # Add to graph
        self.graph.add_edge(source, target, weight=edge.weight)
        
    def update_node_state(
        self,
        node_id: int,
        new_state: np.ndarray
    ):
        """
        Update quantum state of node
        """
        self.nodes[node_id].state = new_state
        
        # Update coherence
        self.nodes[node_id].coherence = np.abs(
            np.sum(new_state * np.conj(new_state))
        )
        
    def update_edge_entanglement(
        self,
        source: int,
        target: int,
        entanglement: float
    ):
        """
        Update entanglement of edge
        """
        edge = self.edges[(source, target)]
        edge.entanglement = entanglement
        
        # Update node entanglement
        self.nodes[source].entanglement = max(
            self.nodes[source].entanglement,
            entanglement
        )
        self.nodes[target].entanglement = max(
            self.nodes[target].entanglement,
            entanglement
        )
        
    def propagate_quantum_state(
        self,
        source: int,
        target: int,
        state: np.ndarray
    ) -> np.ndarray:
        """
        Propagate quantum state through network
        """
        edge = self.edges[(source, target)]
        
        # Apply phase shift
        phase = np.exp(1j * edge.phase)
        state = state * phase
        
        # Apply entanglement
        if edge.entanglement > 0:
            state = self.apply_entanglement(state, edge.entanglement)
            
        return state
        
    def apply_entanglement(
        self,
        state: np.ndarray,
        entanglement: float
    ) -> np.ndarray:
        """
        Apply entanglement to quantum state
        """
        # Create entanglement operator
        operator = np.eye(len(state), dtype=np.complex128)
        operator[0, 0] = np.cos(entanglement)
        operator[0, 1] = np.sin(entanglement)
        operator[1, 0] = -np.sin(entanglement)
        operator[1, 1] = np.cos(entanglement)
        
        # Apply operator
        return np.matmul(operator, state)
        
    def optimize_network(self):
        """
        Optimize network topology
        """
        # Calculate current metrics
        current_metrics = self.calculate_network_metrics()
        
        # Try different connection patterns
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if (i, j) not in self.edges:
                    # Try adding edge
                    self.add_edge(i, j)
                    new_metrics = self.calculate_network_metrics()
                    
                    # If metrics improved, keep edge
                    if new_metrics['coherence'] > current_metrics['coherence']:
                        current_metrics = new_metrics
                    else:
                        # Remove edge
                        self.remove_edge(i, j)
                        
    def remove_edge(self, source: int, target: int):
        """
        Remove edge from network
        """
        # Remove from edges
        del self.edges[(source, target)]
        del self.edges[(target, source)]
        
        # Update node connections
        self.nodes[source].connections.remove(target)
        self.nodes[target].connections.remove(source)
        
        # Remove from graph
        self.graph.remove_edge(source, target)
        
    def calculate_network_metrics(self) -> dict:
        """
        Calculate network-wide metrics
        """
        metrics = {
            'coherence': 0.0,
            'entanglement': 0.0,
            'connectivity': 0.0,
            'efficiency': 0.0
        }
        
        # Calculate average coherence
        metrics['coherence'] = np.mean([
            node.coherence for node in self.nodes.values()
        ])
        
        # Calculate average entanglement
        metrics['entanglement'] = np.mean([
            edge.entanglement for edge in self.edges.values()
        ])
        
        # Calculate network connectivity
        metrics['connectivity'] = nx.average_clustering(self.graph)
        
        # Calculate network efficiency
        metrics['efficiency'] = nx.global_efficiency(self.graph)
        
        return metrics
        
    def get_network_state(self) -> dict:
        """
        Get current state of network
        """
        return {
            'nodes': {
                node_id: {
                    'coherence': node.coherence,
                    'entanglement': node.entanglement,
                    'connections': node.connections
                }
                for node_id, node in self.nodes.items()
            },
            'edges': {
                f"{source}-{target}": {
                    'weight': edge.weight,
                    'phase': edge.phase,
                    'entanglement': edge.entanglement
                }
                for (source, target), edge in self.edges.items()
            },
            'metrics': self.calculate_network_metrics()
        } 