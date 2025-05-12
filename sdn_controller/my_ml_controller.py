# my_ml_controller.py

import os
import sys
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub # Ryu 的協程庫，用於背景任務

# --- 1. 確保 Python 找得到你的 predict_path 模組 ---
#    如果 predict_path.py 和 my_ml_controller.py 在同一個資料夾，這行可能不用
#    如果放在不同地方，需要調整路徑
# sys.path.append('/path/to/your/Simulation-for-SRv6_5G_Simulation/src/ml')
try:
    from predict_path import PathPredictor
except ImportError as e:
    print(f"Error importing PathPredictor: {e}")
    print("Please ensure predict_path.py is in the Python path or the same directory.")
    sys.exit(1)

# --- 2. 設定模型和設定檔的路徑 ---
#    請修改成你實際存放檔案的路徑
MODEL_DIR = './' # 假設檔案都放在跟 my_ml_controller.py 同個資料夾
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')
GRU_MODEL_PATH = os.path.join(MODEL_DIR, 'gru_model.keras')
TRAFFIC_CONFIG_PATH = os.path.join(MODEL_DIR, 'traffic_profiles.yaml')

# --- 3. 從 main.py 取得的範例損失值 (你需要用實際訓練得到的值) ---
#    這些值影響模型融合的權重，先用範例值代替
EXAMPLE_LSTM_VAL_LOSS = 0.1
EXAMPLE_GRU_VAL_LOSS = 0.1

class MyMlController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MyMlController, self).__init__(*args, **kwargs)
        self.topology_api_app = self # Ryu 的拓撲 API
        self.predictor = None       # 用來存放 PathPredictor 物件
        self.logger.info("My ML Controller Initializing...")

        # --- 4. 載入模型和 PathPredictor ---
        try:
            self.logger.info("Loading ML models and PathPredictor...")
            # 檢查檔案是否存在
            required_files = [RF_MODEL_PATH, LSTM_MODEL_PATH, GRU_MODEL_PATH, TRAFFIC_CONFIG_PATH]
            for f_path in required_files:
                if not os.path.exists(f_path):
                     raise FileNotFoundError(f"Required file not found: {f_path}")

            self.predictor = PathPredictor(
                rf_model_path=RF_MODEL_PATH,
                lstm_model_path=LSTM_MODEL_PATH,
                gru_model_path=GRU_MODEL_PATH,
                traffic_config_path=TRAFFIC_CONFIG_PATH,
                lstm_val_loss=EXAMPLE_LSTM_VAL_LOSS, # 使用範例值或實際值
                gru_val_loss=EXAMPLE_GRU_VAL_LOSS    # 使用範例值或實際值
            )
            self.logger.info("PathPredictor loaded successfully.")
        except FileNotFoundError as e:
             self.logger.error(f"Initialization failed: {e}")
             # 在這種情況下，控制器可能無法正常工作，可以選擇退出或設定標記
             self.predictor = None # 確保 predictor 是 None
        except Exception as e:
            # 捕捉載入模型時可能發生的其他錯誤 (例如 TensorFlow 問題)
            self.logger.error(f"Error loading PathPredictor: {e}", exc_info=True)
            self.predictor = None

        # --- 5. 啟動一個背景任務，定期觸發預測 ---
        if self.predictor:
            self.monitor_thread = hub.spawn(self._monitor)
            self.logger.info("Monitor thread started.")
        else:
            self.logger.warning("Predictor not loaded, monitor thread not started.")

    def _monitor(self):
        """定期執行預測的背景任務"""
        self.logger.info("Monitor task running...")
        while True:
            self.logger.info("Monitor loop iteration.")
            # 在這裡觸發預測邏輯
            self._perform_prediction_with_dummy_data()
            hub.sleep(30) # 每 30 秒執行一次 (可以調整)

    def _perform_prediction_with_dummy_data(self):
        """使用假數據來測試 predict_best_path 函數"""
        if not self.predictor:
            self.logger.warning("Predictor is not available, skipping prediction.")
            return

        self.logger.info("Performing prediction with dummy data...")

        # --- 6. 建立符合 predict_best_path 輸入格式的假數據 ---
        #     模擬有三條候選路徑，每條路徑都有所需的指標
        dummy_metrics_list = [
            { # 路徑 1
              'delay': 15.5, 'jitter': 2.1, 'packet_loss': 0.01, 'throughput': 550.0,
              'latency_variance': 1.8, 'reliability': 0.99, 'avg_congestion': 0.15,
              'qos_violations': 0, 'path_id': 'dummy_path_1', 'slice_type': 'eMBB',
              'routing_type': 'SRv6'
            },
            { # 路徑 2
              'delay': 8.2, 'jitter': 0.8, 'packet_loss': 0.001, 'throughput': 300.0,
              'latency_variance': 0.5, 'reliability': 0.999, 'avg_congestion': 0.05,
              'qos_violations': 0, 'path_id': 'dummy_path_2', 'slice_type': 'URLLC',
              'routing_type': 'SRv6'
            },
            { # 路徑 3
              'delay': 25.0, 'jitter': 5.5, 'packet_loss': 0.05, 'throughput': 800.0,
              'latency_variance': 6.0, 'reliability': 0.95, 'avg_congestion': 0.3,
              'qos_violations': 1, 'path_id': 'dummy_path_3', 'slice_type': 'eMBB',
              'routing_type': 'SRv6'
            }
        ]

        try:
            # --- 7. 呼叫預測函數 ---
            best_path_info = self.predictor.predict_best_path(dummy_metrics_list)

            # --- 8. 處理預測結果 ---
            if best_path_info and best_path_info.get('path_id') != 'none':
                self.logger.info(f"Prediction Result: Best path is '{best_path_info['path_id']}' "
                                 f"with score {best_path_info.get('score', 'N/A')}")
                # 在這裡，之後會加入將 best_path_info['path_id'] 轉換成 SRv6 策略
                # 並下發到網路設備的程式碼
            else:
                self.logger.warning("Prediction returned no best path or an error.")

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)

    # --- Ryu 的事件處理函數 (目前是空的，之後會用到) ---
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.logger.info(f"Switch connected: {datapath.id}")
        # 設定 Table-miss Flow Entry (讓不知道怎麼處理的封包送到控制器)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        """輔助函數，用來新增 Flow Entry"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)
