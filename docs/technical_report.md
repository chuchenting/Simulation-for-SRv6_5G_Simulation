# Technical Report: SRv6 5G+ Network Simulation with ML Optimization

## 1. Introduction
This project simulates a 5G+ mobile transport network with SRv6, supporting network slicing (eMBB, URLLC, mMTC) and ML-based path optimization. It compares static SR-TE policies against ML-driven dynamic routing.

## 2. System Design
- **Topology**: Simulated using NetworkX with adjustable nodes and edges.
- **SRv6 Simulation**: Segment lists are lists of node IDs, guiding packet forwarding.
- **Traffic Generation**: Synthetic traffic for slices with distinct QoS profiles.
- **ML Models**: Random Forest and LSTM predict optimal paths based on delay, jitter, and packet loss.
- **Visualization**: Matplotlib plots compare IGP, SRv6, and ML-SRv6 performance.

## 3. Implementation
- **Modules**: Separated into simulation, traffic generation, ML, and visualization.
- **Parameters**: Configurable via YAML files (node count, failure rate, traffic profiles).
- **Dataset**: Simulated metrics saved as CSV for ML training.

## 4. Results
- ML-SRv6 reduces delay by X% compared to static SRv6 (based on simulated data).
- Visualizations show performance trends across routing types.

## 5. Future Directions
- **Reinforcement Learning**: Use RL for adaptive SRv6 path selection.
- **Quantum ML**: Explore QML for 6G network optimization.
- **GNS3 Integration**: Extend to real Cisco XRv9k setups.

## 6. Conclusion
The project demonstrates the feasibility of ML-optimized SRv6 in 5G+ networks, with modular code and clear documentation for further research.