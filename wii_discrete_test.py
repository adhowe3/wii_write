import evdev
import select

def find_wii_buttons():
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            # Find the main button interface, not the accelerometer or IR
            if "Nintendo" in dev.name and "IR" not in dev.name and "Motion Plus" not in dev.name and "Accelerometer" not in dev.name:
                return dev
        except OSError:
            continue
    return None

device = find_wii_buttons()
if not device:
    print("Error: Could not find Wii Remote Buttons.")
    quit()

print(f"Connected to: {device.name}")
print("\n--- DISCRETE EVENT TESTER ---")
print("Instructions: Press the 'A' button exactly 50 times at a normal clicking speed.")
print("The script will count how many presses actually survived the TDM blind spot.")
print("Press CTRL+C when you are done to see the results.\n")

press_count = 0

try:
    while True:
        r, w, x = select.select([device.fd], [], [])
        for event in device.read():
            # We only count the moment the button goes DOWN (value == 1)
            if event.type == evdev.ecodes.EV_KEY and event.code == evdev.ecodes.BTN_A:
                if event.value == 1:
                    press_count += 1
                    print(f"Press registered! Count: {press_count}", end='\r')

except KeyboardInterrupt:
    print("\n\n--- TEST RESULTS ---")
    print(f"Total Registered Presses: {press_count}")
    print("If you pressed it 50 times, subtract your count from 50 to find your Blind Spot Drop Rate.")
    print("--------------------\n")