import evdev
import select
import time
import pandas as pd
import warnings
# Import the class from Allan's file
from ml import HybridAirWritingModel 

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# LOAD THE HYBRID MODEL
MODEL_FILE = "hybrid_wii_model.pkl"
try:
    print(f"Loading Hybrid AI model from {MODEL_FILE}...")
    # Use the custom load method Allan wrote
    model = HybridAirWritingModel.load_model(MODEL_FILE)
except Exception as e:
    print(f"Error loading model: {e}")
    quit()

try:
    while True:
        r, w, x = select.select(devices, [], [])
        for fd in r:
            for event in devices[fd].read():
                
                # --- HANDLE BUTTONS ---
                if event.type == evdev.ecodes.EV_KEY:
                    if event.code == evdev.ecodes.BTN_A:
                        if event.value == 1:
                            is_recording = True
                        elif event.value == 0:
                            is_recording = False
                            stroke_counter += 1
                            
                    # D-Pad UP (Predict Letter)
                    elif event.code == evdev.ecodes.BTN_DPAD_UP or event.code == evdev.ecodes.KEY_UP:
                        if event.value == 1 and len(current_letter_data) > 5: # Changed to 5 for Allan's min-check
                            
                            # Create the raw DataFrame
                            df = pd.DataFrame(current_letter_data, columns=['timestamp', 'stroke_id', 'x', 'y', 'z'])
                            
                            print("\n" + "="*40)
                            print("HYBRID INFERENCE IN PROGRESS...")
                            
                            # 3. NEW PREDICTION PATH
                            # We don't extract features here anymore! 
                            # The model.predict() method handles preprocessing and extraction internally.
                            prediction = model.predict(df)
                            
                            print("="*40)
                            print(f"PREDICTED LETTER:  {prediction}")
                            print("="*40)
                            
                            # Reset
                            current_letter_data = []
                            stroke_counter = 1

                    # Button B (Erase)
                    elif event.code == evdev.ecodes.BTN_B:
                        if event.value == 1:
                            current_letter_data = []
                            stroke_counter = 1
                            print("[-] Buffer cleared.\n")

                # --- HANDLE ACCELEROMETER ---
                elif event.type == evdev.ecodes.EV_ABS:
                    if event.code == evdev.ecodes.ABS_RX: current_x = event.value
                    elif event.code == evdev.ecodes.ABS_RY: current_y = event.value
                    elif event.code == evdev.ecodes.ABS_RZ: current_z = event.value
                        
                    if is_recording:
                        # Append raw values exactly as the model expects
                        current_letter_data.append([time.time(), stroke_counter, current_x, current_y, current_z])

except KeyboardInterrupt:
    print("\nExiting Live Inference.")