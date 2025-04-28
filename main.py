import numpy as np  # Added import for numpy
from src.simulation.network_sim import NetworkSimulator
from src.simulation.traffic_gen import TrafficGenerator
from src.simulation.metrics_collector import MetricsCollector
from src.ml.train_model import MLPathOptimizer
from src.ml.predict_path import PathPredictor
from src.visualization.plot_results import plot_performance_comparison

def main():
    # Initialize components
    sim = NetworkSimulator('configs/simulation_config.yaml')
    traffic_gen = TrafficGenerator('configs/traffic_profiles.yaml')
    metrics_collector = MetricsCollector()
    selected_paths = []

    # Define slice types
    slices = ['eMBB', 'URLLC', 'mMTC', 'video_streaming', 'game_streaming', 'metaverse', 'iot', 'ar_vr', 'autonomous_vehicles', 'smart_grid', 'healthcare_monitoring']
    
    # Simulate traffic and collect metrics for IGP, SRv6
    for iteration in range(10):  # Increased to 10 iterations
        sim.simulate_link_failure(failure_rate=0.03)
        sim.simulate_node_failure(failure_rate=0.01)
        sim.simulate_congestion(congestion_increase=0.15)
        sim.simulate_bandwidth_fluctuation(fluctuation_rate=0.2)
        
        for slice_type in slices:
            traffic = traffic_gen.generate_traffic(1000, slice_type)
            avg_packet_size = sum(p['size'] for p in traffic) / len(traffic)
            
            for i in range(250):  # Increased to 250 paths per slice per iteration
                # IGP: Shortest path without SRv6
                path_id = f"igp_path_{iteration}_{i}_{slice_type}"
                segment_list = sim.define_segment_list(0, 19, path_id)
                metrics = sim.simulate_packet_forwarding(segment_list, slice_type)
                if segment_list:
                    congestion = [sim.G.edges[segment_list[j], segment_list[j+1]]['congestion'] 
                                 for j in range(len(segment_list)-1) if (segment_list[j], segment_list[j+1]) in sim.G.edges]
                    avg_congestion = np.mean(congestion) if congestion else 0
                else:
                    avg_congestion = 0
                metrics_collector.collect_metrics(path_id, metrics, 'IGP', slice_type, avg_packet_size, avg_congestion)

                # SRv6: Use segment list
                path_id = f"srv6_path_{iteration}_{i}_{slice_type}"
                segment_list = sim.define_segment_list(0, 19, path_id)
                metrics = sim.simulate_packet_forwarding(segment_list, slice_type)
                if segment_list:
                    congestion = [sim.G.edges[segment_list[j], segment_list[j+1]]['congestion'] 
                                 for j in range(len(segment_list)-1) if (segment_list[j], segment_list[j+1]) in sim.G.edges]
                    avg_congestion = np.mean(congestion) if congestion else 0
                else:
                    avg_congestion = 0
                metrics_collector.collect_metrics(path_id, metrics, 'SRv6', slice_type, avg_packet_size, avg_congestion)

    # Save initial data
    traffic_gen.save_traffic_data('data/traffic_data.csv')
    metrics_collector.save_metrics('data/ml_dataset.csv')

    # Train ML models
    ml_optimizer = MLPathOptimizer('data/ml_dataset.csv')
    ml_optimizer.train_rf()
    ml_optimizer.train_lstm()
    ml_optimizer.train_gru()

    # Get validation losses for ensemble weighting
    lstm_val_loss = ml_optimizer.lstm_history['val_loss'][-1]
    gru_val_loss = ml_optimizer.gru_history['val_loss'][-1]

    # Predict optimal paths and simulate ML-SRv6
    predictor = PathPredictor('rf_model.pkl', 'lstm_model.keras', 'gru_model.keras', 'configs/traffic_profiles.yaml', lstm_val_loss, gru_val_loss)
    for slice_type in slices:
        for iteration in range(10):
            slice_metrics = [m for m in metrics_collector.metrics 
                           if m['slice_type'] == slice_type and m['path_id'].startswith(f"srv6_path_{iteration}")]
            if not slice_metrics:
                continue
            best_path = predictor.predict_best_path(slice_metrics, reliability_threshold=0.85)
            selected_paths.append(best_path)
            
            if best_path['path_id'] != 'none':
                path_id = f"ml_path_{iteration}_{slice_type}_{best_path['path_id']}"
                segment_list = sim.define_segment_list(0, 19, path_id)
                metrics = sim.simulate_packet_forwarding(segment_list, slice_type)
                avg_packet_size = sum(p['size'] for p in traffic) / len(traffic)
                if segment_list:
                    congestion = [sim.G.edges[segment_list[j], segment_list[j+1]]['congestion'] 
                                 for j in range(len(segment_list)-1) if (segment_list[j], segment_list[j+1]) in sim.G.edges]
                    avg_congestion = np.mean(congestion) if congestion else 0
                else:
                    avg_congestion = 0
                metrics_collector.collect_metrics(path_id, metrics, 'ML-SRv6', slice_type, avg_packet_size, avg_congestion)

    # Update metrics with ML-SRv6
    metrics_collector.save_metrics('data/ml_dataset.csv')

    # Visualize results
    history = {'lstm_history': ml_optimizer.lstm_history, 'gru_history': ml_optimizer.gru_history}
    plot_performance_comparison('data/ml_dataset.csv', 'results', history, selected_paths)
    print(f"Best predicted paths for each slice: {[p['path_id'] for p in selected_paths]}")

if __name__ == "__main__":
    main()