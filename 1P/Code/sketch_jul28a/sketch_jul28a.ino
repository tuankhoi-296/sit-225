#include <DHT.h>

#define PIR_PIN   2      // PIR D2
#define DHT_PIN   3      // DHT22 D3
#define DHT_TYPE  DHT22

DHT dht(DHT_PIN, DHT_TYPE);

const unsigned long DHT_INTERVAL = 2000;  // 2 s / record
unsigned long lastDhtRead = 0;
int lastPirState = LOW;

void setup() {
  Serial.begin(9600);
  while (!Serial) { ; }        

  pinMode(PIR_PIN, INPUT);
  dht.begin();

  // Print header (with tags for script parsing) 
  Serial.println("PIR,timestamp_ms,state");
  Serial.println("DHT,timestamp_ms,humidity,temperature");

  lastPirState = digitalRead(PIR_PIN);
}

void loop() {
  unsigned long now = millis();   //  timestamp in milliseconds since startup

  // --- 1. PIR Sensor Logic: State-Change Detection ---
  // --- PIR: only log on state CHANGE  ---
  int pirState = digitalRead(PIR_PIN);

  // Only log data if the state has CHANGED.
  // This prevents spamming the serial port with repetitive data.
  if (pirState != lastPirState) {
    Serial.print("PIR,");
    Serial.print(now);
    Serial.print(",");
    Serial.println(pirState == HIGH ? "HIGH" : "LOW");
    lastPirState = pirState;
  }
  // --- 2. DHT22 Sensor Logic: Interval Polling ---
  // --- log every 2 seconds ---
  if (now - lastDhtRead >= DHT_INTERVAL) {
    lastDhtRead = now;
    float h = dht.readHumidity();
    float t = dht.readTemperature();   // Celsius degree

    // Data Quality Check: Verify if the reads failed and return NaN
    Serial.print("DHT,");
    Serial.print(now);
    Serial.print(",");
    if (isnan(h) || isnan(t)) {
      Serial.println("NaN,NaN");       // sensor read error
    } else {
      Serial.print(h, 1);
      Serial.print(",");
      Serial.println(t, 1);
    }
  }
}