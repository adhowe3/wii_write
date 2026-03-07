import os
import pandas as pd
import numpy as np

# Point this to your master dataset folder
DATASET_PATH = "./wii_dataset_local"
OUTPUT_FILE = "extracted_features.csv"

print(f"Scanning directory: {DATASET_PATH}...")

all_features = []

# Loop through every folder (A, B, C...)
for label in os.listdir(DATASET_PATH):
    label_dir = os.path.join(DATASET_PATH, label)
    
    # Skip if it's not a folder
    if not os.path.isdir(label_dir):
        continue
        
    # Loop through every CSV file inside the letter folder
    for csv_file in os.listdir(label_dir):
        if not csv_file.endswith(".csv"):
            continue
            
        file_path = os.path.join(label_dir, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            
            # Skip empty files or files that are too short to calculate variance
            if len(df) < 2:
                print(f"Skipping {file_path}: Not enough data points.")
                continue
                
            # --- FEATURE CALCULATION ---
            # 1. Temporal & Stroke Features
            duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
            num_strokes = df['stroke_id'].nunique()
            
            # 2. Statistical Features (per axis)
            features = {
                'label': label,           # The target we want to predict (e.g., 'A')
                'filename': csv_file,     # Kept for debugging/tracking
                'duration_sec': duration,
                'num_strokes': num_strokes,
                
                # X-Axis Stats
                'x_mean': df['x'].mean(),
                'x_std': df['x'].std(),   # Standard deviation (variance)
                'x_min': df['x'].min(),
                'x_max': df['x'].max(),
                'x_range': df['x'].max() - df['x'].min(),
                
                # Y-Axis Stats
                'y_mean': df['y'].mean(),
                'y_std': df['y'].std(),
                'y_min': df['y'].min(),
                'y_max': df['y'].max(),
                'y_range': df['y'].max() - df['y'].min(),
                
                # Z-Axis Stats
                'z_mean': df['z'].mean(),
                'z_std': df['z'].std(),
                'z_min': df['z'].min(),
                'z_max': df['z'].max(),
                'z_range': df['z'].max() - df['z'].min(),
            }
            
            all_features.append(features)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Compile everything into a single master DataFrame
features_df = pd.DataFrame(all_features)

# Clean up any potential NaN values (e.g., if a standard deviation calculation failed)
features_df.fillna(0, inplace=True)

# Save to disk
features_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n[SUCCESS] Extracted features from {len(features_df)} files.")
print(f"Saved dataset to: {OUTPUT_FILE}")