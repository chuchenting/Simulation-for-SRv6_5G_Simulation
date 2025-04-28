import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import yaml
from typing import List

class PathPredictor:
    def __init__(self, rf_model_path: str, lstm_model_path: str, gru_model_path: str, traffic_config_path: str, lstm_val_loss: float, gru_val_loss: float):
        self.rf_model = pickle.load(open(rf_model_path, 'rb'))
        self.lstm_model = load_model(lstm_model_path)
        self.gru_model = load_model(gru_model_path)
        with open(traffic_config_path, 'r') as f:
            self.traffic_config = yaml.safe_load(f)
        total_loss = lstm_val_loss + gru_val_loss + 1
        self.rf_weight = 1 / total_loss
        self.lstm_weight = (1 / lstm_val_loss) / total_loss
        self.gru_weight = (1 / gru_val_loss) / total_loss

    def predict_best_path(self, metrics: List[dict], time_steps: int = 1, reliability_threshold: float = 0.85) -> dict:
        """Predict the best SRv6 path using a weighted ensemble of ML models with QoS weighting and reliability filter."""
        X = pd.DataFrame(
            [[m['delay'], m['jitter'], m['packet_loss'], m['throughput'], 
              m['latency_variance'], m['reliability'], m['avg_congestion'], m['qos_violations']] for m in metrics],
            columns=['delay', 'jitter', 'packet_loss', 'throughput', 'latency_variance', 'reliability', 'avg_congestion', 'qos_violations']
        )
        valid_indices = X[X['reliability'] >= reliability_threshold].index
        if len(valid_indices) == 0:
            valid_indices = X.index
        X_filtered = X.loc[valid_indices]
        metrics_filtered = [metrics[i] for i in valid_indices]
        
        if X_filtered.empty:
            return {'path_id': 'none', 'score': 0, 'slice_type': metrics[0]['slice_type'], 'routing_type': 'none', 'metrics': {}, 'ensemble_score': 0}

        slice_types = [m['slice_type'] for m in metrics_filtered]
        
        rf_scores = self.rf_model.predict(X_filtered)
        X_lstm = X_filtered.to_numpy().reshape((X_filtered.shape[0], time_steps, X_filtered.shape[1]))
        lstm_scores = self.lstm_model.predict(X_lstm, verbose=0).flatten()
        gru_scores = self.gru_model.predict(X_lstm, verbose=0).flatten()
        
        combined_scores = (self.rf_weight * rf_scores + 
                          self.lstm_weight * lstm_scores + 
                          self.gru_weight * gru_scores)

        # Apply QoS weighting and penalize QoS violations
        weighted_scores = []
        for i, (score, slice_type) in enumerate(zip(combined_scores, slice_types)):
            config = self.traffic_config[slice_type]
            latency_weight = config['latency_sensitivity']
            throughput_weight = config['throughput_sensitivity']
            delay = X_filtered.iloc[i]['delay']
            throughput = X_filtered.iloc[i]['throughput']
            reliability = X_filtered.iloc[i]['reliability']
            qos_violations = X_filtered.iloc[i]['qos_violations']
            
            # Define QoS thresholds (same as in network_sim.py)
            qos_thresholds = {
                'URLLC': {'max_delay': 5, 'min_throughput': 200},
                'game_streaming': {'max_delay': 10, 'min_throughput': 300},
                'autonomous_vehicles': {'max_delay': 3, 'min_throughput': 200},
                'healthcare_monitoring': {'max_delay': 8, 'min_throughput': 150},
                'default': {'max_delay': 50, 'min_throughput': 100}
            }
            threshold = qos_thresholds.get(slice_type, qos_thresholds['default'])
            
            # Base score adjustment
            adjusted_score = (score * (1 - latency_weight - throughput_weight) +
                            (100 / (1 + delay)) * latency_weight +
                            throughput * throughput_weight +
                            reliability * 0.1)
            
            # Apply penalty for QoS violations
            if delay > threshold['max_delay']:
                adjusted_score *= 0.8  # 20% penalty
            if throughput < threshold['min_throughput']:
                adjusted_score *= 0.9  # 10% penalty
            adjusted_score -= qos_violations * 5  # Additional penalty per violation
            
            weighted_scores.append(adjusted_score)

        best_path_idx = np.argmax(weighted_scores)
        return {
            'path_id': metrics_filtered[best_path_idx]['path_id'],
            'score': weighted_scores[best_path_idx],
            'slice_type': metrics_filtered[best_path_idx]['slice_type'],
            'routing_type': metrics_filtered[best_path_idx]['routing_type'],
            'metrics': metrics_filtered[best_path_idx],
            'ensemble_score': combined_scores[best_path_idx]
        }