import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import pickle

class MLPathOptimizer:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.rf_model = None
        self.lstm_model = None
        self.gru_model = None
        self.lstm_history = None
        self.gru_history = None

    def preprocess_data(self):
        """Prepare data for ML training."""
        X = self.data[['delay', 'jitter', 'packet_loss', 'throughput', 'latency_variance', 'reliability', 'avg_congestion', 'qos_violations']]
        y = self.data['path_score']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_rf(self):
        """Train Random Forest model."""
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)

    def train_lstm(self, time_steps: int = 1):
        """Train LSTM model with adjusted time_steps."""
        X_train, X_test, y_train, y_test = self.preprocess_data()
        X_train = np.array(X_train).reshape((X_train.shape[0], time_steps, X_train.shape[1]))
        X_test = np.array(X_test).reshape((X_test.shape[0], time_steps, X_test.shape[1]))
        
        self.lstm_model = Sequential([
            Input(shape=(time_steps, X_train.shape[2])),
            LSTM(50, activation='relu', return_sequences=False),
            Dense(1)
        ])
        self.lstm_model.compile(optimizer='adam', loss=MeanSquaredError())
        history = self.lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self.lstm_history = history.history
        self.lstm_model.save('lstm_model.keras')

    def train_gru(self, time_steps: int = 1):
        """Train GRU model with adjusted time_steps."""
        X_train, X_test, y_train, y_test = self.preprocess_data()
        X_train = np.array(X_train).reshape((X_train.shape[0], time_steps, X_train.shape[1]))
        X_test = np.array(X_test).reshape((X_test.shape[0], time_steps, X_test.shape[1]))
        
        self.gru_model = Sequential([
            Input(shape=(time_steps, X_train.shape[2])),
            GRU(50, activation='relu', return_sequences=False),
            Dense(1)
        ])
        self.gru_model.compile(optimizer='adam', loss=MeanSquaredError())
        history = self.gru_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        self.gru_history = history.history
        self.gru_model.save('gru_model.keras')