import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the features
DATA_FILE = "extracted_features.csv"
print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Separate Features (X) from Labels (y)
# We drop 'label' because that's what we're guessing, and 'filename' because it's not a math feature.
X = df.drop(columns=['label', 'filename'])
y = df['label']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.\n")

# Initialize and Train the Random Forest
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Take the Pop Quiz (Predict on the test set)
y_pred = rf_model.predict(X_test)

# Grade the Test
accuracy = accuracy_score(y_test, y_pred)
print("="*40)
print(f"OVERALL ACCURACY: {accuracy * 100:.2f}%")
print("="*40)

# Print detailed stats per letter
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.1f}%)", fontsize=16)
plt.ylabel('Actual Letter Written')
plt.xlabel('Letter Predicted by ML Model')
plt.tight_layout()
plt.show()

EXPORT_FILE = "wii_rf_model.pkl"
joblib.dump(rf_model, EXPORT_FILE)
print(f"\n[+] Successfully exported trained model to {EXPORT_FILE}")