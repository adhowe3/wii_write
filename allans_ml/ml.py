import os
import glob
import numpy as np
import pandas as pd

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


class AirWritingModel:
    def __init__(self, target_length=100):
        self.target_length = target_length
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            verbose=True
        )
        self.label_encoder = LabelEncoder()

    # -------------------------------
    # Preprocessing functions
    # -------------------------------

    def remove_gravity(self, stroke):
        """
        Subtract mean from each axis (gravity removal approximation)
        stroke: (T, 3)
        """
        return stroke - np.mean(stroke, axis=0)

    def normalize(self, stroke):
        """
        Normalize to zero mean and unit variance
        """
        std = np.std(stroke, axis=0)
        std[std == 0] = 1  # avoid divide by zero
        return (stroke - np.mean(stroke, axis=0)) / std

    def resample_stroke(self, stroke):
        """
        Resample to fixed length
        """
        return resample(stroke, self.target_length)

    def preprocess_stroke(self, stroke):
        stroke = self.remove_gravity(stroke)
        stroke = self.normalize(stroke)
        stroke = self.resample_stroke(stroke)
        return stroke

    # -------------------------------
    # Data loading
    # -------------------------------

    def load_dataset(self, root_dir):
        X = []
        y = []

        for label_folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, label_folder)

            if not os.path.isdir(folder_path):
                continue

            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)

                # Group by stroke_id
                for stroke_id, group in df.groupby("stroke_id"):
                    stroke = group[["x", "y", "z"]].values.astype(np.float32)

                    if len(stroke) < 5:
                        continue  # skip tiny strokes

                    stroke = self.preprocess_stroke(stroke)

                    # Flatten for MLP
                    stroke_flat = stroke.flatten()

                    X.append(stroke_flat)
                    y.append(label_folder)

        X = np.array(X)
        y = np.array(y)

        print(f"Loaded dataset: {X.shape}, labels: {len(set(y))}")

        return X, y

    # -------------------------------
    # Training
    # -------------------------------

    def train(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        print("Training model...")
        self.model.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = self.model.predict(X_test)

        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

    # -------------------------------
    # Inference
    # -------------------------------

    def predict(self, stroke):
        """
        stroke: (T, 3)
        """
        stroke = self.preprocess_stroke(stroke)
        stroke = stroke.flatten().reshape(1, -1)

        pred = self.model.predict(stroke)[0]
        return self.label_encoder.inverse_transform([pred])[0]


# -------------------------------
# Main entry point
# -------------------------------

if __name__ == "__main__":
    data_path = "../wii_dataset_local"

    model = AirWritingModel(target_length=100)

    X, y = model.load_dataset(data_path)

    model.train(X, y)