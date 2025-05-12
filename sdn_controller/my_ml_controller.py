import os
import sys

# --- 1. Ensure Python can find the src directory ---
# Get the directory containing this script (sdn_controller)
controller_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Repo root)
repo_root = os.path.dirname(controller_dir)
# Construct the absolute path to the 'src' directory
src_path = os.path.join(repo_root, 'src')
# Add the 'src' directory to Python's search path (sys.path)
if src_path not in sys.path:
    sys.path.insert(0, src_path) # Insert at the beginning for higher priority

# --- Now we can import modules from src ---
try:
    # Since src is in sys.path, we can directly import from the 'ml' package
    from ml.predict_path import PathPredictor
except ImportError as e:
    print(f"Error importing PathPredictor: {e}")
    print(f"Ensure 'predict_path.py' exists in '{os.path.join(src_path, 'ml')}'")
    sys.exit(1)

# --- 2. Set paths for models and config files (relative to Repo root) ---
# NOTE: You need to verify where main.py actually saves the model files
# (e.g., in the repo root, or a 'data/' or 'models/' folder) and adjust paths below!
MODEL_DIR = repo_root # Assuming models are in the repo root for now
CONFIG_DIR = os.path.join(repo_root, 'configs')

RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl') # Verify actual location!
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras') # Verify actual location!
GRU_MODEL_PATH = os.path.join(MODEL_DIR, 'gru_model.keras') # Verify actual location!
TRAFFIC_CONFIG_PATH = os.path.join(CONFIG_DIR, 'traffic_profiles.yaml') # This should be correct

# --- 3. Import Ryu related modules ---
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub # Ryu's GreenThread library for background tasks

# --- 4. Example validation loss values (you should use values from actual training) ---
# These values affect the weighting in model ensembling. Replace with real values.
EXAMPLE_LSTM_VAL_LOSS = 0.1
EXAMPLE_GRU_VAL_LOSS = 0.1

class MyMlController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MyMlController, self).__init__(*args, **kwargs)
        self.topology_api_app = self # Ryu's topology API app
        self.predictor = None       # To hold the PathPredictor instance
        self.logger.info("My ML Controller Initializing...")

        # --- 5. Load ML models and PathPredictor ---
        try:
            self.logger.info("Loading ML models and PathPredictor...")
            # Check if required files exist
            required_files = [RF_MODEL_PATH, LSTM_MODEL_PATH, GRU_MODEL_PATH, TRAFFIC_CONFIG_PATH]
            for f_path in required_files:
                if not os.path.exists(f_path):
                     # Raise an error if any required file is missing
                     raise FileNotFoundError(f"Required file not found: {f_path}. Please check paths in my_ml_controller.py.")

            # Instantiate the PathPredictor
            self.predictor = PathPredictor(
                rf_model_path=RF_MODEL_PATH,
                lstm_model_path=LSTM_MODEL_PATH,
                gru_model_path=GRU_MODEL_PATH,
                traffic_config_path=TRAFFIC_CONFIG_PATH,
                lstm_val_loss=EXAMPLE_LSTM_VAL_LOSS, # Use example or actual value
                gru_val_loss=EXAMPLE_GRU_VAL_LOSS    # Use example or actual value
            )
            self.logger.info("PathPredictor loaded successfully.")
        except FileNotFoundError as e:
             # Log error if a file was not found
             self.logger.error(f"Initialization failed: {e}")
             self.predictor = None # Ensure predictor is None if loading failed
        except Exception as e:
            # Catch other potential errors during model loading (e.g., TensorFlow issues)
            self.logger.error(f"Error loading PathPredictor: {e}", exc_info=True)
            self.predictor = None

        # --- 6. Start a background task for periodic prediction ---
        if self.predictor:
            # Spawn a green thread that runs the _monitor method
            self.monitor_thread = hub.spawn(self._monitor)
            self.logger.info("Monitor thread started.")
        else:
            # Log a warning if the predictor couldn't be loaded
            self.logger.warning("Predictor not loaded, monitor thread not started.")

    def _monitor(self):
        """Background task for periodic prediction execution"""
        self.logger.info("Monitor task running...")
        while True:
            self.logger.info("Monitor loop iteration.")
            # Trigger the prediction logic here
            self._perform_prediction_with_dummy_data()
            # Wait for a specified interval before the next iteration
            hub.sleep(30) # Execute every 30 seconds (adjustable)

    def _perform_prediction_with_dummy_data(self):
        """Use dummy data to test the predict_best_path function"""
        # Check if the predictor was loaded successfully
        if not self.predictor:
            self.logger.warning("Predictor is not available, skipping prediction.")
            return

        self.logger.info("Performing prediction with dummy data...")

        # --- 7. Create dummy data matching the input format for predict_best_path ---
        #     Simulate three candidate paths, each with the required metrics.
        #     Replace these with actual metrics gathered from the network later.
        dummy_metrics_list = [
            { # Path 1
              'delay': 15.5, 'jitter': 2.1, 'packet_loss': 0.01, 'throughput': 550.0,
              'latency_variance': 1.8, 'reliability': 0.99, 'avg_congestion': 0.15,
              'qos_violations': 0, 'path_id': 'dummy_path_1', 'slice_type': 'eMBB',
              'routing_type': 'SRv6'
            },
            { # Path 2
              'delay': 8.2, 'jitter': 0.8, 'packet_loss': 0.001, 'throughput': 300.0,
              'latency_variance': 0.5, 'reliability': 0.999, 'avg_congestion': 0.05,
              'qos_violations': 0, 'path_id': 'dummy_path_2', 'slice_type': 'URLLC',
              'routing_type': 'SRv6'
            },
            { # Path 3
              'delay': 25.0, 'jitter': 5.5, 'packet_loss': 0.05, 'throughput': 800.0,
              'latency_variance': 6.0, 'reliability': 0.95, 'avg_congestion': 0.3,
              'qos_violations': 1, 'path_id': 'dummy_path_3', 'slice_type': 'eMBB',
              'routing_type': 'SRv6'
            }
        ]

        try:
            # --- 8. Call the prediction function ---
            best_path_info = self.predictor.predict_best_path(dummy_metrics_list)

            # --- 9. Process the prediction result ---
            if best_path_info and best_path_info.get('path_id') != 'none':
                self.logger.info(f"Prediction Result: Best path is '{best_path_info['path_id']}' "
                                 f"with score {best_path_info.get('score', 'N/A')}")
                # TODO: Add code here later to translate best_path_info['path_id']
                #       into an actual SRv6 policy (SID list) and program it
                #       onto the network device (e.g., using NETCONF, BGP-LS, PCEP).
            else:
                # Log a warning if the prediction didn't yield a valid path
                self.logger.warning("Prediction returned no best path or an error condition.")

        except Exception as e:
            # Log any unexpected errors during the prediction call
            self.logger.error(f"Error during prediction: {e}", exc_info=True)

    # --- Ryu Event Handlers (basic setup for now) ---
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch features event when a switch connects."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.logger.info(f"Switch connected: {datapath.id:016x}") # Log switch DPID

        # Install table-miss flow entry
        # This sends packets that don't match any other flow entry to the controller.
        match = parser.OFPMatch() # Match all packets
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, # Send to controller
                                          ofproto.OFPCML_NO_BUFFER)] # Don't buffer packet
        self.add_flow(datapath, 0, match, actions) # Add flow with priority 0 (lowest)

    def add_flow(self, datapath, priority, match, actions):
        """Helper function to add a flow entry to the switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Construct flow modification message
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        # Send the message to the switch
        datapath.send_msg(mod)
