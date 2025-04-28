# Setup Instructions

## Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Running the Simulation
1. Configure parameters in `configs/simulation_config.yaml` and `configs/traffic_profiles.yaml`.
2. Run the main script: `python main.py`.
3. Check outputs in `data/` (traffic and metrics) and `results/` (plots).

## GNS3 Mode (Optional)
- **Requirements**: GNS3, Cisco XRv9k/QEMU images.
- **Setup**:
  1. Create topology with Access, Aggregation, Core routers.
  2. Configure SRv6 policies using XRv9k CLI.
  3. Use TRex for traffic generation (see TRex YAML configs).
- **Troubleshooting**:
  - Ensure QEMU nodes have sufficient RAM (4GB+).
  - Check IPv6 routing tables if connectivity fails (`show ipv6 route`).
  - Verify SRv6 SID allocation.

## Notes
- Adjust `num_nodes` and `failure_rate` in `simulation_config.yaml` for larger simulations.
- ML models require sufficient data; run multiple iterations for robust training.