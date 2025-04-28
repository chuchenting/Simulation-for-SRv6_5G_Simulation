import pandas as pd
import time
from typing import Dict, List

class MetricsCollector:
    def __init__(self):
        self.metrics = []

    def collect_metrics(self, path_id: str, metrics: Dict, routing_type: str, slice_type: str, packet_size: float, avg_congestion: float):
        """Collect metrics for a path and compute path_score."""
        delay = metrics['delay']
        jitter = metrics['jitter']
        packet_loss = metrics['packet_loss']
        latency_variance = metrics['latency_variance']
        qos_violations = metrics['qos_violations']
        # Simple cost function: higher delay/jitter/loss/variance/violations -> lower score
        cost = (0.3 * delay) + (0.2 * jitter) + (0.2 * packet_loss * 100) + (0.1 * latency_variance) + (0.2 * qos_violations)
        path_score = 100 / (1 + cost)  # Higher score for lower cost
        throughput = (packet_size * 8) / (delay / 1000) / 1e6 if delay > 0 else 0
        reliability = 1 - packet_loss if packet_loss < 1 else 0
        metrics['path_score'] = path_score
        metrics['throughput'] = throughput
        metrics['latency_variance'] = latency_variance
        metrics['reliability'] = reliability
        metrics['qos_violations'] = qos_violations
        metrics['avg_congestion'] = avg_congestion
        metrics['path_id'] = path_id
        metrics['routing_type'] = routing_type
        metrics['slice_type'] = slice_type
        metrics['timestamp'] = time.time()
        self.metrics.append(metrics)

    def save_metrics(self, output_path: str):
        """Save collected metrics to CSV."""
        df = pd.DataFrame(self.metrics)
        df.to_csv(output_path, index=False)