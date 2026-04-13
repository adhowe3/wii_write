import evdev
import time
import select

# --- CONFIGURATION ---
EXPECTED_FREQ = 100.0  # Wii Remote sends ~100 samples per second

def find_wii_devices():
    accel, buttons = None, None
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            if "Nintendo" in dev.name:
                if "Accelerometer" in dev.name:
                    accel = dev
                # Grab the main button interface (ignoring IR and Motion Plus)
                elif "IR" not in dev.name and "Motion Plus" not in dev.name:
                    buttons = dev
        except OSError:
            continue
    return accel, buttons

accel_dev, button_dev = find_wii_devices()

if not accel_dev or not button_dev:
    print("Error: Could not find both Wii Remote Accelerometer and Buttons.")
    print("Is the remote awake and connected?")
    quit()

print(f"Connected to Motion:  {accel_dev.name}")
print(f"Connected to Buttons: {button_dev.name}")
print("\n--- PACKET LOSS TESTER ---")
print("Instructions: Hold 'A' for a few seconds, then release.")
print("The script will compare actual samples vs. elapsed time.")
print("Press CTRL+C to stop.\n")

# Create the multiplexer dictionary
devices = {dev.fd: dev for dev in [accel_dev, button_dev]}

samples = []
start_time = 0
is_recording = False

try:
    while True:
        # Wait for data from EITHER the motion sensor or the buttons
        r, w, x = select.select(devices, [], [])
        
        for fd in r:
            for event in devices[fd].read():
                
                # --- BUTTON EVENTS ---
                if event.type == evdev.ecodes.EV_KEY and event.code == evdev.ecodes.BTN_A:
                    if event.value == 1:  # Button Pressed
                        is_recording = True
                        start_time = time.time()
                        samples = []
                        print("Recording... ", end='\r', flush=True)
                        
                    elif event.value == 0:  # Button Released
                        is_recording = False
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        actual_count = len(samples)
                        
                        # Calculate the actual Hz achieved during this burst
                        actual_hz = actual_count / duration if duration > 0 else 0
                        
                        print(f"\n--- TEST RESULTS ---")
                        print(f"Duration:     {duration:.2f}s")
                        print(f"Received:     {actual_count} physical packets")
                        print(f"Average Rate: {actual_hz:.2f} Hz")
                        print(f"--------------------\n")

                # --- ACCELEROMETER EVENTS ---
                # Count EV_SYN instead of EV_ABS to count full packets, not individual axes
                if is_recording and event.type == evdev.ecodes.EV_SYN and event.code == evdev.ecodes.SYN_REPORT:
                    samples.append(event.timestamp())

except KeyboardInterrupt:
    print("\nTest finished.")