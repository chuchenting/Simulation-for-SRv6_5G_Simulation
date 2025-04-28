import numpy as np
import pandas as pd
import yaml
from typing import Dict

class TrafficGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.profiles = yaml.safe_load(f)
        self.traffic_data = []

    def generate_traffic(self, num_packets: int, slice_type: str) -> Dict:
        """Generate traffic for a specific slice (eMBB, URLLC, mMTC)."""
        profile = self.profiles[slice_type]
        packets = []
        for _ in range(num_packets):
            packet = {
                'timestamp': np.random.uniform(0, 100),
                'size': np.random.normal(profile['packet_size_mean'], profile['packet_size_std']),
                'priority': profile['priority'],
                'slice_type': slice_type
            }
            packets.append(packet)
        self.traffic_data.extend(packets)
        return packets

    def save_traffic_data(self, output_path: str):
        """Save generated traffic data to CSV."""
        df = pd.DataFrame(self.traffic_data)
        df.to_csv(output_path, index=False)