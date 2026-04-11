import os
import glob
import numpy as np
import pandas as pd
import joblib

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


class HybridAirWritingModel:
    def __init__(self, target_length=100):
        self.target_length = target_length

        # Time-series model
        self.ts_model = MLPClassifier(
            # hidden_layer_sizes=(128, 64),
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            learning_rate_init=1e-4,
            max_iter=400,
            verbose=False
        )

        # Feature-based model
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        self.label_encoder = LabelEncoder()

    # -------------------------------
    # Preprocessing (time-series)
    # -------------------------------

    def remove_gravity(self, stroke):
        return stroke - np.mean(stroke, axis=0)

    def normalize(self, stroke):
        std = np.std(stroke, axis=0)
        std[std == 0] = 1
        return (stroke - np.mean(stroke, axis=0)) / std

    def resample_stroke(self, stroke):
        return resample(stroke, self.target_length)

    def preprocess_stroke(self, stroke):
        stroke = self.remove_gravity(stroke)
        stroke = self.normalize(stroke)

        # mag = np.linalg.norm(stroke, axis=1, keepdims=True) # adding the magnitude of x y z
        velocity = np.cumsum(stroke, axis=0)    # add velocity
        # stroke = np.hstack([stroke, mag, velocity])
        # stroke = np.hstack([stroke, mag])

        stroke = self.resample_stroke(stroke)   # resample very last
        return stroke

    # -------------------------------
    # Feature extraction (RF)
    # -------------------------------

    def extract_features(self, df):
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        num_strokes = df['stroke_id'].nunique()

        features = {
            'duration_sec': duration,
            'num_strokes': num_strokes,
        }

        for axis in ['x', 'y', 'z']:
            features[f'{axis}_mean'] = df[axis].mean()
            features[f'{axis}_std'] = df[axis].std()
            features[f'{axis}_min'] = df[axis].min()
            features[f'{axis}_max'] = df[axis].max()
            features[f'{axis}_range'] = df[axis].max() - df[axis].min()

        return np.array(list(features.values()), dtype=np.float32)

    # -------------------------------
    # Data loading
    # -------------------------------

    def load_dataset(self, root_dir):
        X_ts = []
        X_rf = []
        y = []

        for label_folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, label_folder)
            if not os.path.isdir(folder_path):
                continue

            for csv_file in glob.glob(os.path.join(folder_path, "*.csv")):
                df = pd.read_csv(csv_file)

                if len(df) < 5:
                    continue

                # ---- Time-series ----
                stroke = df[['x', 'y', 'z']].values.astype(np.float32)
                stroke = self.preprocess_stroke(stroke)
                X_ts.append(stroke.flatten())

                # ---- Feature-based ----
                feats = self.extract_features(df)
                X_rf.append(feats)

                y.append(label_folder)

        X_ts = np.array(X_ts)
        X_rf = np.array(X_rf)
        y = np.array(y)

        print(f"Loaded {len(y)} samples")

        return X_ts, X_rf, y

    # -------------------------------
    # Training
    # -------------------------------

    def train(self, X_ts, X_rf, y):
        y_enc = self.label_encoder.fit_transform(y)

        Xts_train, Xts_test, Xrf_train, Xrf_test, y_train, y_test = train_test_split(
            X_ts, X_rf, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        print("Training time-series model...")
        self.ts_model.fit(Xts_train, y_train)

        print("Training random forest...")
        self.rf_model.fit(Xrf_train, y_train)

        self.Xts_test = Xts_test
        self.Xrf_test = Xrf_test
        self.y_test = y_test

    # -------------------------------
    # Voting prediction
    # -------------------------------

    def predict(self, stroke_df):
        # ---- Time-series ----
        stroke = stroke_df[['x', 'y', 'z']].values.astype(np.float32)
        stroke = self.preprocess_stroke(stroke)
        ts_input = stroke.flatten().reshape(1, -1)

        # ---- RF features ----
        rf_input = self.extract_features(stroke_df).reshape(1, -1)

        # ---- Probabilities ----
        ts_probs = self.ts_model.predict_proba(ts_input)[0]
        rf_probs = self.rf_model.predict_proba(rf_input)[0]

        # ---- Combine (average) ----
        combined_probs = (ts_probs + rf_probs) / 2

        pred_idx = np.argmax(combined_probs)

        return self.label_encoder.inverse_transform([pred_idx])[0]

    # -------------------------------
    # Evaluation
    # -------------------------------
    def evaluate(self):
        correct_hybrid = 0
        correct_ts = 0
        correct_rf = 0

        for i in range(len(self.y_test)):
            # Get test sample
            X_ts_sample = self.Xts_test[i].reshape(1, -1)
            X_rf_sample = self.Xrf_test[i].reshape(1, -1)
            y_true = self.y_test[i]

            # Predict probabilities
            ts_probs = self.ts_model.predict_proba(X_ts_sample)[0]
            rf_probs = self.rf_model.predict_proba(X_rf_sample)[0]

            # Hybrid (average probs)
            combined = (ts_probs + rf_probs) / 2
            pred_hybrid = np.argmax(combined)

            # Individual model predictions
            pred_ts = np.argmax(ts_probs)
            pred_rf = np.argmax(rf_probs)

            # Count correct
            if pred_hybrid == y_true:
                correct_hybrid += 1
            if pred_ts == y_true:
                correct_ts += 1
            if pred_rf == y_true:
                correct_rf += 1

        # Compute accuracies
        total = len(self.y_test)
        acc_hybrid = correct_hybrid / total
        acc_ts = correct_ts / total
        acc_rf = correct_rf / total

        print(f"\nHybrid Model Accuracy: {acc_hybrid * 100:.2f}%")
        print(f"Time-Series Model Accuracy: {acc_ts * 100:.2f}%")
        print(f"Random Forest Model Accuracy: {acc_rf * 100:.2f}%")


    def save_model(self, filename="hybrid_wii_model.pkl"):
        """Packages the trained models and encoder into a single file."""
        joblib.dump({
            'ts_model': self.ts_model,
            'rf_model': self.rf_model,
            'label_encoder': self.label_encoder,
            'target_length': self.target_length
        }, filename)
        print(f"\n[SUCCESS] Hybrid model saved to {filename}")

    @classmethod
    def load_model(cls, filename="hybrid_wii_model.pkl"):
        """Loads a pre-trained model from a file."""
        data = joblib.load(filename)
        
        # Create a blank instance of the class
        instance = cls(target_length=data['target_length'])
        
        # Inject the trained weights and encoder
        instance.ts_model = data['ts_model']
        instance.rf_model = data['rf_model']
        instance.label_encoder = data['label_encoder']
        
        return instance


if __name__ == "__main__":
    data_path = "./wii_dataset_local"

    model = HybridAirWritingModel()

    X_ts, X_rf, y = model.load_dataset(data_path)

    model.train(X_ts, X_rf, y)

    model.evaluate()

    model.save_model("hybrid_wii_model.pkl")