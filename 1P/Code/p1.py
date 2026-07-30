import serial
import csv
from datetime import datetime

PORT = "COM9"        #my port
BAUD = 9600 # Baud rate matching the Arduino sketch

# Open CSV files in write mode to store sensor data streams separately
pir_file = open("pir_log.csv", "w", newline="")
dht_file = open("dht22_log.csv", "w", newline="")
pir_writer = csv.writer(pir_file)
dht_writer = csv.writer(dht_file)

# Write the header rows for both CSV files
pir_writer.writerow(["pc_datetime", "arduino_ms", "state"])
dht_writer.writerow(["pc_datetime", "arduino_ms", "humidity", "temperature"])

# Establish connection with the Arduino
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Reading from {PORT}... Press  Ctrl+C to stop")

try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        print(line)

        parts = line.split(",")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Check if the incoming data is from the PIR sensor
        if parts[0] == "PIR" and len(parts) == 3 and parts[1].isdigit():
            pir_writer.writerow([now, parts[1], parts[2]])
            pir_file.flush()
        # Check if the incoming data is from the DHT22 sensor
        elif parts[0] == "DHT" and len(parts) == 4 and parts[1].isdigit():
            dht_writer.writerow([now, parts[1], parts[2], parts[3]])
            dht_file.flush()
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
    pir_file.close()
    dht_file.close()
