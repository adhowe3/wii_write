import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Check if a file was passed in the command line
if len(sys.argv) < 2:
    print("Usage: python visualize_wii.py <path_to_csv>")
    sys.exit(1)

file_path = sys.argv[1]
if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}")
    sys.exit(1)

# Load the data using Pandas
df = pd.read_csv(file_path)

# Normalize timestamps so the graph always starts exactly at 0 seconds
df['time_sec'] = df['timestamp'] - df['timestamp'].iloc[0]

# Calculate the exact number of strokes
num_strokes = df['stroke_id'].nunique()

# Create a figure with 3 stacked subplots (one for each axis)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Add the stroke count to the main title
csv_filename = os.path.basename(file_path)
fig.suptitle(f"Wii Remote Accelerometer Signature\n{csv_filename}  |  Total Strokes: {num_strokes}", fontsize=16, fontweight='bold')

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

# Find where the stroke_id changes and draw vertical dashed lines to mark pen lifts
stroke_changes = df[df['stroke_id'].diff() != 0]['time_sec'].tolist()[1:]
for ax in (ax1, ax2, ax3):
    for t in stroke_changes:
        ax.axvline(x=t, color='black', linestyle='--', alpha=0.8)
        
    # Optional: shade the background for alternating strokes
    for i in range(len(stroke_changes) + 1):
        start = stroke_changes[i-1] if i > 0 else 0
        end = stroke_changes[i] if i < len(stroke_changes) else df['time_sec'].iloc[-1]
        if i % 2 == 0:
            ax.axvspan(start, end, facecolor='gray', alpha=0.1)

plt.tight_layout()

# --- SAVE INSTEAD OF SHOW ---
# Extract the letter from the filename (e.g., "N" from "N_4.csv")
letter_dir = csv_filename.split('_')[0]
out_dir = os.path.join("wii_local_graphs", letter_dir)
os.makedirs(out_dir, exist_ok=True) # Ensure the subfolder exists

out_file_path = os.path.join(out_dir, csv_filename.replace('.csv', '.png'))

plt.savefig(out_file_path, dpi=100)
plt.close(fig)

print(f"[SUCCESS] Saved graph to: {out_file_path}")