import os
import sys
import random # Added for dynamic dummy data

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
# NOTE: Ensure your model files (rf_model.pkl, lstm_model.keras, gru_model.keras)
# are in the location specified by MODEL_DIR (e.g., repo root or a 'data/' folder).
# Adjust MODEL_DIR if your models are saved elsewhere by main.py.
MODEL_DIR = repo_root # Assuming models are in the repo root for now
CONFIG_DIR = os.path.join(repo_root, 'configs')

RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')
GRU_MODEL_PATH = os.path.join(MODEL_DIR, 'gru_model.keras')
TRAFFIC_CONFIG_PATH = os.path.join(CONFIG_DIR, 'traffic_profiles.yaml')

# --- 3. Import Ryu related modules ---
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER # MAIN_DISPATCHER might be needed later
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub # Ryu's GreenThread library for background tasks

# --- 4. Example validation loss values (you should use values from actual training) ---
# These values affect the weighting in model ensembling. Replace with real values if available.
# If you regenerate models, main.py calculates these. For now, placeholders are fine.
EXAMPLE_LSTM_VAL_LOSS = 0.1 # Placeholder, get actual value if possible
EXAMPLE_GRU_VAL_LOSS = 0.1  # Placeholder, get actual value if possible

class MyMlController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MyMlController, self).__init__(*args, **kwargs)
        self.topology_api_app = self # For topology discovery later
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
                     raise FileNotFoundError(f"Required file not found: {f_path}. "
                                             "Please ensure models are generated (e.g., by running main.py) "
                                             "and paths in my_ml_controller.py are correct.")

            # Instantiate the PathPredictor
            self.predictor = PathPredictor(
                rf_model_path=RF_MODEL_PATH,
                lstm_model_path=LSTM_MODEL_PATH,
                gru_model_path=GRU_MODEL_PATH,
                traffic_config_path=TRAFFIC_CONFIG_PATH,
                lstm_val_loss=EXAMPLE_LSTM_VAL_LOSS,
                gru_val_loss=EXAMPLE_GRU_VAL_LOSS
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
            self.logger.warning("Predictor not loaded, so the monitor thread (for periodic predictions) was not started.")

    def _monitor(self):
        """Background task for periodic prediction execution"""
        self.logger.info("Monitor task running...")
        while True:
            self.logger.info("Monitor loop iteration.")
            # Trigger the prediction logic here
            self._perform_prediction_with_dynamic_dummy_data() # Changed method name
            # Wait for a specified interval before the next iteration
            hub.sleep(10) # Execute every 10 seconds (adjustable)

    def _perform_prediction_with_dynamic_dummy_data(self): # Changed method name
        """Use DYNAMIC dummy data to test the predict_best_path function"""
        # Check if the predictor was loaded successfully
        if not self.predictor:
            self.logger.warning("Predictor is not available, skipping prediction.")
            return

        self.logger.info("Performing prediction with DYNAMIC dummy data...")

        # --- 7. Create DYNAMIC dummy data matching the input format for predict_best_path ---
        #     Metrics will change slightly in each iteration.
        dummy_metrics_list = [
            { # Path 1
              'delay': random.uniform(10.0, 20.0),
              'jitter': random.uniform(1.0, 3.0),
              'packet_loss': random.uniform(0.005, 0.02),
              'throughput': random.uniform(400.0, 600.0),
              'latency_variance': random.uniform(1.0, 2.5),
              'reliability': random.uniform(0.98, 0.995),
              'avg_congestion': random.uniform(0.1, 0.2),
              'qos_violations': random.choice([0, 1]),
              'path_id': 'dummy_path_1',
              'slice_type': 'eMBB', # Keep slice_type fixed for easier comparison for now
              'routing_type': 'SRv6'
            },
            { # Path 2
              'delay': random.uniform(5.0, 15.0),
              'jitter': random.uniform(0.5, 1.5),
              'packet_loss': random.uniform(0.0, 0.005),
              'throughput': random.uniform(250.0, 350.0),
              'latency_variance': random.uniform(0.2, 0.8),
              'reliability': random.uniform(0.995, 0.9999),
              'avg_congestion': random.uniform(0.01, 0.1),
              'qos_violations': random.choice([0, 1]),
              'path_id': 'dummy_path_2',
              'slice_type': 'URLLC', # Keep slice_type fixed for easier comparison for now
              'routing_type': 'SRv6'
            },
            { # Path 3
              'delay': random.uniform(20.0, 30.0),
              'jitter': random.uniform(4.0, 7.0),
              'packet_loss': random.uniform(0.03, 0.08),
              'throughput': random.uniform(700.0, 900.0),
              'latency_variance': random.uniform(5.0, 8.0),
              'reliability': random.uniform(0.90, 0.98),
              'avg_congestion': random.uniform(0.25, 0.4),
              'qos_violations': random.choice([0, 1, 2]),
              'path_id': 'dummy_path_3',
              'slice_type': 'eMBB', # Keep slice_type fixed for easier comparison for now
              'routing_type': 'SRv6'
            }
        ]

        try:
            # Call the prediction function; time_steps defaults to 1 in PathPredictor
            best_path_info = self.predictor.predict_best_path(dummy_metrics_list)

            # Process the prediction result
            if best_path_info and best_path_info.get('path_id') != 'none':
                self.logger.info(f"Prediction Result: Best path is '{best_path_info['path_id']}' "
                                 f"with score {best_path_info.get('score', 'N/A')}, "
                                 f"Slice: {best_path_info.get('slice_type', 'N/A')}, "
                                 f"Metrics (delay): {best_path_info.get('metrics', {}).get('delay', 'N/A'):.2f}, " # Format delay
                                 f"Metrics (loss): {best_path_info.get('metrics', {}).get('packet_loss', 'N/A'):.4f}") # Format loss
            else:
                # Log a warning if the prediction didn't yield a valid path
                self.logger.warning("Prediction returned no best path or an error condition.")

        except Exception as e:
            # Log any unexpected errors during the prediction call
            self.logger.error(f"Error during prediction: {e}", exc_info=True)

    # --- Ryu Event Handlers (basic setup for now, will be expanded later) ---
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
