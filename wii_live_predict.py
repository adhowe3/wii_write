import evdev
import select
import time
import pandas as pd
import joblib
import warnings

# Suppress scikit-learn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

# Load the Machine Learning Model
MODEL_FILE = "wii_rf_model.pkl"
try:
    print(f"Loading AI model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    print("[SUCCESS] Brain loaded!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    quit()

#Connect to the Wii Remote
print("Searching for Wii Remote devices...")
time.sleep(1)
accel_device, button_device = None, None

for path in evdev.list_devices():
    try:
        dev = evdev.InputDevice(path)
        if "Nintendo" in dev.name:
            if "Accelerometer" in dev.name:
                accel_device = dev
            elif "IR" not in dev.name and "Motion Plus" not in dev.name:
                button_device = dev
    except OSError:
        pass

if not accel_device or not button_device:
    print("Error: Could not find Wii Remote.")
    quit()

print("\n--- LIVE INFERENCE READY ---")
print("1. HOLD 'A' to draw strokes.")
print("2. PRESS 'D-Pad UP' to predict the letter.")
print("3. PRESS 'B' to clear the buffer if you mess up.")
print("4. Press CTRL+C to exit.\n")

# State variables
is_recording = False
current_letter_data = []
stroke_counter = 1
current_x = current_y = current_z = 0

devices = {dev.fd: dev for dev in [accel_device, button_device]}

try:
    while True:
        r, w, x = select.select(devices, [], [])
        
        for fd in r:
            for event in devices[fd].read():
                
                # --- HANDLE BUTTONS ---
                if event.type == evdev.ecodes.EV_KEY:
                    
                    # Button A (Pen Down / Up)
                    if event.code == evdev.ecodes.BTN_A:
                        if event.value == 1:
                            is_recording = True
                        elif event.value == 0:
                            is_recording = False
                            stroke_counter += 1
                            
                    # D-Pad UP (Predict Letter)
                    elif event.code == evdev.ecodes.BTN_DPAD_UP or event.code == evdev.ecodes.KEY_UP:
                        if event.value == 1 and len(current_letter_data) > 2:
                            
                            # Convert raw data to a DataFrame
                            df = pd.DataFrame(current_letter_data, columns=['timestamp', 'stroke_id', 'x', 'y', 'z'])
                            
                            # Extract Features
                            duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
                            num_strokes = df['stroke_id'].nunique()
                            
                            features = {
                                'duration_sec': [duration],
                                'num_strokes': [num_strokes],
                                'x_mean': [df['x'].mean()], 'x_std': [df['x'].std()], 'x_min': [df['x'].min()], 'x_max': [df['x'].max()], 'x_range': [df['x'].max() - df['x'].min()],
                                'y_mean': [df['y'].mean()], 'y_std': [df['y'].std()], 'y_min': [df['y'].min()], 'y_max': [df['y'].max()], 'y_range': [df['y'].max() - df['y'].min()],
                                'z_mean': [df['z'].mean()], 'z_std': [df['z'].std()], 'z_min': [df['z'].min()], 'z_max': [df['z'].max()], 'z_range': [df['z'].max() - df['z'].min()]
                            }
                            
                            # Clean up potential NaN values for single-point strokes
                            feature_df = pd.DataFrame(features).fillna(0)

                            print("\n" + "="*40)
                            print("DEBUG: WHAT THE AI SEES")
                            print(feature_df.iloc[0].to_string())
                            
                            # Ask the Model for a Prediction
                            prediction = model.predict(feature_df)[0]
                            
                            print("="*40)
                            print(f"PREDICTED LETTER:  {prediction}")
                            print("="*40)
                            
                            # Reset for the next letter
                            current_letter_data = []
                            stroke_counter = 1

                    # Button B (Erase / Clear Buffer)
                    elif event.code == evdev.ecodes.BTN_B:
                        if event.value == 1:
                            current_letter_data = []
                            stroke_counter = 1
                            print("[-] Buffer cleared. Restart your letter.\n")

                # --- HANDLE ACCELEROMETER ---
                elif event.type == evdev.ecodes.EV_ABS:
                    if event.code == evdev.ecodes.ABS_RX: current_x = event.value
                    elif event.code == evdev.ecodes.ABS_RY: current_y = event.value
                    elif event.code == evdev.ecodes.ABS_RZ: current_z = event.value
                        
                    if is_recording:
                        current_letter_data.append([time.time(), stroke_counter, current_x, current_y, current_z])

except KeyboardInterrupt:
    print("\nExiting Live Inference.")
