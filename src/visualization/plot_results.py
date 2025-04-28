import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict
import plotly.graph_objects as go
from math import pi
from pandas.plotting import parallel_coordinates

def plot_performance_comparison(data_path: str, output_dir: str, ml_history: dict, selected_paths: List[dict]):
    """Generate various plots for performance analysis and path selection."""
    data = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize timestamps
    if 'timestamp' in data.columns:
        data['timestamp'] = data['timestamp'] - data['timestamp'].min()
    else:
        data['timestamp'] = data.index

    # 1-2. Test-Train Loss Curves for LSTM and GRU
    plt.figure(figsize=(10, 6))
    plt.plot(ml_history['lstm_history']['loss'], label='LSTM Training Loss', color='blue')
    plt.plot(ml_history['lstm_history']['val_loss'], label='LSTM Validation Loss', color='cyan')
    plt.title('LSTM Test-Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'lstm_test_train_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ml_history['gru_history']['loss'], label='GRU Training Loss', color='green')
    plt.plot(ml_history['gru_history']['val_loss'], label='GRU Validation Loss', color='lime')
    plt.title('GRU Test-Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'gru_test_train_loss.png'))
    plt.close()

    # 3-6. Routing Performance: Delay, Jitter, Packet Loss, Throughput
    metrics = ['delay', 'jitter', 'packet_loss', 'throughput']
    routing_types = ['IGP', 'SRv6', 'ML-SRv6']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for rtype in routing_types:
            subset = data[data['routing_type'] == rtype]
            plt.plot(subset['timestamp'], subset[metric], label=f'{rtype} {metric.capitalize()}')
        plt.title(f'{metric.capitalize()} Comparison Across Routing Types')
        plt.xlabel('Time (s)')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'routing_{metric}.png'))
        plt.close()

    # 7-9. Split by Slice Type: Delay, Jitter, Throughput
    slice_types = data['slice_type'].unique()
    for metric in ['delay', 'jitter', 'throughput']:
        plt.figure(figsize=(12, 6))
        for stype in slice_types:
            subset = data[data['slice_type'] == stype]
            plt.plot(subset['timestamp'], subset[metric], label=f'{stype} {metric.capitalize()}')
        plt.title(f'{metric.capitalize()} Across Slice Types')
        plt.xlabel('Time (s)')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'slice_{metric}.png'))
        plt.close()

    # 10. Box Plot: Delay Distribution by Routing Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='routing_type', y='delay', data=data, hue='routing_type')
    plt.title('Delay Distribution by Routing Type')
    plt.xlabel('Routing Type')
    plt.ylabel('Delay (ms)')
    plt.savefig(os.path.join(output_dir, 'boxplot_delay_routing.png'))
    plt.close()

    # 11. Box Plot: Delay Distribution by Slice Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='slice_type', y='delay', data=data, hue='slice_type')
    plt.title('Delay Distribution by Slice Type')
    plt.xlabel('Slice Type')
    plt.ylabel('Delay (ms)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'boxplot_delay_slice.png'))
    plt.close()

    # 12. Histogram: Delay Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['delay'], bins=30, alpha=0.7, color='blue')
    plt.title('Delay Distribution Histogram')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'histogram_delay.png'))
    plt.close()

    # 13. Scatter Plot: Delay vs Throughput
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='delay', y='throughput', hue='slice_type', style='routing_type', size='avg_congestion', data=data)
    plt.title('Delay vs Throughput by Slice and Routing Type')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Throughput (Mbps)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'scatter_delay_throughput.png'), bbox_inches='tight')
    plt.close()

    # 14. Violin Plot: Jitter by Slice Type
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='slice_type', y='jitter', hue='routing_type', data=data)
    plt.title('Jitter Distribution by Slice Type and Routing')
    plt.xlabel('Slice Type')
    plt.ylabel('Jitter (ms)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'violin_jitter_slice.png'))
    plt.close()

    # 15. Pair Plot: Metrics Relationships
    sns.pairplot(data[['delay', 'jitter', 'throughput', 'reliability', 'slice_type']], hue='slice_type')
    plt.savefig(os.path.join(output_dir, 'pairplot_metrics.png'))
    plt.close()

    # 16. Heatmap: Correlation Between Metrics
    plt.figure(figsize=(10, 8))
    corr = data[['delay', 'jitter', 'packet_loss', 'throughput', 'latency_variance', 'reliability', 'avg_congestion', 'qos_violations']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Network Metrics')
    plt.savefig(os.path.join(output_dir, 'heatmap_metrics.png'))
    plt.close()

    # 17. 3D Scatter Plot: Delay, Throughput, Reliability
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data['delay'], data['throughput'], data['reliability'], c=data['avg_congestion'], cmap='viridis')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_zlabel('Reliability')
    plt.colorbar(scatter, label='Avg Congestion')
    plt.title('3D Scatter: Delay, Throughput, Reliability')
    plt.savefig(os.path.join(output_dir, '3d_scatter_metrics.png'))
    plt.close()

    # 18. Bar Plot: Selected Paths by Slice Type
    selected_df = pd.DataFrame(selected_paths)
    plt.figure(figsize=(12, 6))
    sns.countplot(x='slice_type', hue='routing_type', data=selected_df)
    plt.title('Selected Paths by Slice Type and Routing')
    plt.xlabel('Slice Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'bar_selected_paths.png'))
    plt.close()

    # 19. Heatmap: Path Scores by Slice and Routing Type
    pivot = selected_df.pivot_table(values='score', index='slice_type', columns='routing_type', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title('Average Path Scores by Slice and Routing Type')
    plt.xlabel('Routing Type')
    plt.ylabel('Slice Type')
    plt.savefig(os.path.join(output_dir, 'heatmap_path_scores.png'))
    plt.close()

    # 20. Time-Series Heatmap: Congestion Over Time by Slice Type
    pivot_congestion = data.pivot_table(values='avg_congestion', index='slice_type', columns='timestamp', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_congestion, cmap='Reds')
    plt.title('Congestion Over Time by Slice Type')
    plt.xlabel('Time (s)')
    plt.ylabel('Slice Type')
    plt.savefig(os.path.join(output_dir, 'heatmap_congestion_time.png'))
    plt.close()

    # 21. Sankey Diagram: Path Selection Transitions
    if not selected_df.empty:
        nodes = list(selected_df['slice_type'].unique()) + list(selected_df['path_id'].unique())
        node_dict = {node: idx for idx, node in enumerate(nodes)}
        sources = [node_dict[slice_type] for slice_type in selected_df['slice_type']]
        targets = [node_dict[path_id] for path_id in selected_df['path_id']]
        values = [1 for _ in range(len(sources))]
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            )
        )])
        fig.update_layout(title_text="Path Selection Transitions (Slice Type to Path)", font_size=10)
        fig.write_html(os.path.join(output_dir, 'sankey_path_selection.html'))

    # 22. Radar Chart: QoS Metrics for Selected Paths
    if not selected_df.empty:
        categories = ['delay', 'jitter', 'packet_loss', 'throughput', 'reliability']
        slice_types = selected_df['slice_type'].unique()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        for stype in slice_types:
            subset = selected_df[selected_df['slice_type'] == stype].iloc[0]
            values = [
                subset['metrics']['delay'],
                subset['metrics']['jitter'],
                subset['metrics']['packet_loss'] * 100,
                subset['metrics']['throughput'] / 10,
                subset['metrics']['reliability'] * 100
            ]
            values += values[:1]
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            ax.plot(angles, values, label=stype)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        plt.title('QoS Metrics for Selected Paths by Slice Type')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(os.path.join(output_dir, 'radar_qos_metrics.png'))
        plt.close()

    # 23. Time-Series Plot: Ensemble Scores Over Iterations
    if not selected_df.empty:
        # Extract iteration, handle NaN by filtering out invalid path_ids
        extracted_iterations = selected_df['path_id'].str.extract(r'ml_path_(\d+)_')
        selected_df['iteration'] = extracted_iterations[0].fillna(-1).astype(int)  # Use -1 for invalid paths
        valid_df = selected_df[selected_df['iteration'] != -1]  # Exclude invalid paths
        if not valid_df.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='iteration', y='ensemble_score', hue='slice_type', data=valid_df)
            plt.title('Ensemble Scores of Selected Paths Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Ensemble Score')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'timeseries_ensemble_scores.png'))
            plt.close()

    # 24. Stacked Bar Chart: QoS Violations by Slice Type and Routing
    plt.figure(figsize=(12, 6))
    pivot_violations = data.pivot_table(values='qos_violations', index='slice_type', columns='routing_type', aggfunc='sum')
    pivot_violations.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('QoS Violations by Slice Type and Routing')
    plt.xlabel('Slice Type')
    plt.ylabel('Total QoS Violations')
    plt.xticks(rotation=45)
    plt.legend(title='Routing Type')
    plt.savefig(os.path.join(output_dir, 'stacked_bar_qos_violations.png'))
    plt.close()

    # 25. Parallel Coordinates Plot: Metrics Comparison
    plt.figure(figsize=(12, 6))
    metrics_to_plot = data[['delay', 'jitter', 'throughput', 'reliability', 'qos_violations', 'slice_type']].copy()
    for col in ['delay', 'jitter', 'throughput', 'reliability', 'qos_violations']:
        metrics_to_plot[col] = (metrics_to_plot[col] - metrics_to_plot[col].min()) / (metrics_to_plot[col].max() - metrics_to_plot[col].min())
    parallel_coordinates(metrics_to_plot, 'slice_type', colormap='viridis')
    plt.title('Parallel Coordinates Plot of Metrics by Slice Type')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'parallel_coordinates_metrics.png'))
    plt.close()

    # 26. Bubble Chart: Path Scores by Slice Type and Routing
    if not selected_df.empty:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            x='slice_type', 
            y='score', 
            size='ensemble_score', 
            hue='routing_type', 
            data=selected_df,
            sizes=(50, 500)
        )
        plt.title('Path Scores by Slice Type and Routing (Bubble Size = Ensemble Score)')
        plt.xlabel('Slice Type')
        plt.ylabel('Path Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(output_dir, 'bubble_path_scores.png'), bbox_inches='tight')
        plt.close()

    # 27. Gantt Chart: Path Usage Over Iterations
    if not selected_df.empty and 'iteration' in selected_df.columns:
        valid_df['duration'] = 1  # Each path is used for 1 iteration
        fig = plt.figure(figsize=(12, 8))
        colors = sns.color_palette("husl", len(valid_df['slice_type'].unique()))
        color_dict = dict(zip(valid_df['slice_type'].unique(), colors))
        
        for idx, row in valid_df.iterrows():
            plt.barh(
                y=row['path_id'], 
                width=row['duration'], 
                left=row['iteration'], 
                color=color_dict[row['slice_type']], 
                alpha=0.6, 
                label=row['slice_type'] if row['slice_type'] not in plt.gca().get_legend_handles_labels()[1] else ""
            )
        
        plt.title('Gantt Chart: Path Usage Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Path ID')
        plt.legend(title='Slice Type')
        plt.grid(True, axis='x')
        plt.savefig(os.path.join(output_dir, 'gantt_path_usage.png'))
        plt.close()