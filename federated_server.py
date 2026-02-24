
import numpy as np
import tensorflow as tf
from typing import List

class FederatedServer:
    def __init__(self, model):
        self.global_model = model
        self.round_history = []

    def get_global_weights(self):
        return self.global_model.get_weights()

    def federated_averaging(self, client_weights_list: List, client_data_sizes: List):
        total_samples = sum(client_data_sizes)
        num_layers = len(client_weights_list[0])
        aggregated_weights = []
        for layer_idx in range(num_layers):
            layer_weights = np.zeros_like(client_weights_list[0][layer_idx], dtype=np.float64)
            for client_idx, client_weights in enumerate(client_weights_list):
                weight = client_data_sizes[client_idx] / total_samples
                layer_weights += weight * client_weights[layer_idx]
            aggregated_weights.append(layer_weights)
        return aggregated_weights

    def aggregate_and_update(self, client_weights_list: List, client_data_sizes: List, round_num: int):
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Aggregating {len(client_weights_list)} hospital updates")
        print(f"{'='*60}")
        aggregated_weights = self.federated_averaging(client_weights_list, client_data_sizes)
        self.global_model.set_weights(aggregated_weights)
        print("Global model updated")

    def evaluate_global_model(self, test_data, round_num: int):
        results = self.global_model.evaluate(test_data, verbose=0)
        metrics = {'round': round_num, 'loss': results[0], 'accuracy': results[1]}
        self.round_history.append(metrics)
        print(f"Global Model Performance:")
        print(f"  Loss: {results[0]:.4f}")
        print(f"  Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
        return metrics
