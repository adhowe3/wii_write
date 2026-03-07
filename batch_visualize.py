import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Check if a directory was passed in the command line
if len(sys.argv) < 2:
    print("Usage: python batch_visualize.py <path_to_dataset_folder>")
    sys.exit(1)

dataset_path = sys.argv[1]
if not os.path.exists(dataset_path):
    print(f"Error: Could not find {dataset_path}")
    sys.exit(1)

# Set up the master output directory
OUTPUT_DIR = "wii_local_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Scanning dataset: {dataset_path}")
print(f"Saving graphs to: ./{OUTPUT_DIR}/")

file_count = 0

# Loop through every letter folder (A, B, C...)
for label in os.listdir(dataset_path):
    label_dir = os.path.join(dataset_path, label)
    
    # Skip if it's not a folder
    if not os.path.isdir(label_dir):
        continue
        
    # Create the corresponding output subfolder (e.g., wii_local_graphs/A)
    out_label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)
    
    # Loop through every CSV file inside the letter folder
    for csv_file in os.listdir(label_dir):
        if not csv_file.endswith(".csv"):
            continue
            
        file_path = os.path.join(label_dir, csv_file)
        out_file_path = os.path.join(out_label_dir, csv_file.replace('.csv', '.png'))
        
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Skip empty or corrupted files
            if len(df) < 2:
                print(f"Skipping {csv_file}: Not enough data.")
                continue

            # Normalize timestamps
            df['time_sec'] = df['timestamp'] - df['timestamp'].iloc[0]
            
            # --- Calculate the exact number of strokes ---
            num_strokes = df['stroke_id'].nunique()

            # Create the figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
            # --- NEW: Add the stroke count to the main title ---
            fig.suptitle(f"Wii Remote Accelerometer Signature\n{csv_file}  |  Total Strokes: {num_strokes}", fontsize=16, fontweight='bold')

            # Plot X-Axis
            ax1.plot(df['time_sec'], df['x'], color='#e74c3c', linewidth=2, label='X-Axis')
            ax1.set_ylabel('Raw X')
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.6)

            # Plot Y-Axis
            ax2.plot(df['time_sec'], df['y'], color='#2ecc71', linewidth=2, label='Y-Axis')
            ax2.set_ylabel('Raw Y')
            ax2.legend(loc='upper right')
            ax2.grid(True, linestyle='--', alpha=0.6)

            # Plot Z-Axis
            ax3.plot(df['time_sec'], df['z'], color='#3498db', linewidth=2, label='Z-Axis')
            ax3.set_ylabel('Raw Z')
            ax3.set_xlabel('Time (seconds)')
            ax3.legend(loc='upper right')
            ax3.grid(True, linestyle='--', alpha=0.6)

            # Find where the stroke_id changes and draw vertical dashed lines
            stroke_changes = df[df['stroke_id'].diff() != 0]['time_sec'].tolist()[1:]
            for ax in (ax1, ax2, ax3):
                for t in stroke_changes:
                    ax.axvline(x=t, color='black', linestyle='--', alpha=0.8)
                    
                # Shade the background for alternating strokes
                for i in range(len(stroke_changes) + 1):
                    start = stroke_changes[i-1] if i > 0 else 0
                    end = stroke_changes[i] if i < len(stroke_changes) else df['time_sec'].iloc[-1]
                    if i % 2 == 0:
                        ax.axvspan(start, end, facecolor='gray', alpha=0.1)

            plt.tight_layout()
            
            # Save the figure to the new directory and CLOSE it to save RAM
            plt.savefig(out_file_path, dpi=100)
            plt.close(fig)
            
            file_count += 1
            # Print a progress dot so you know it hasn't frozen
            print(".", end="", flush=True)

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")

print(f"\n[SUCCESS] Generated {file_count} graphs in ./{OUTPUT_DIR}/")