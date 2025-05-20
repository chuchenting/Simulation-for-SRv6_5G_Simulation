# sdn_controller/my_ml_controller.py

import os
import sys
import random # For dynamic dummy data

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
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
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

        # Define the slice types to simulate optimization for, in a cycle
        # You can get these from your 'configs/traffic_profiles.yaml' keys
        # Or from the 'slices' list in your main.py
        self.slice_types_to_simulate = [
            'eMBB', 'URLLC', 'mMTC', 'video_streaming', 'game_streaming',
            'metaverse', 'iot', 'ar_vr', 'autonomous_vehicles',
            'smart_grid', 'healthcare_monitoring'
        ] # Using the list from your main.py
        self.current_slice_index = 0 #
