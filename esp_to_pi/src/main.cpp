#include <Arduino.h>

// Standard onboard LED for most ESP32 WROOM-32 boards
#define LED_PIN     LED_BUILTIN  // typically GPIO2
#define BLINK_DELAY 1000          // ms

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("ESP32 WROOM-32E — LED blink started");
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("LED ON");
  delay(BLINK_DELAY);

  digitalWrite(LED_PIN, LOW);
  Serial.println("LED OFF");
  delay(BLINK_DELAY);
}