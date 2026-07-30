import serial
import csv
from datetime import datetime

PORT = "COM9"        #my port
BAUD = 9600

pir_file = open("pir_log.csv", "w", newline="")
dht_file = open("dht22_log.csv", "w", newline="")
pir_writer = csv.writer(pir_file)
dht_writer = csv.writer(dht_file)

pir_writer.writerow(["pc_datetime", "arduino_ms", "state"])
dht_writer.writerow(["pc_datetime", "arduino_ms", "humidity", "temperature"])

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

        if parts[0] == "PIR" and len(parts) == 3 and parts[1].isdigit():
            pir_writer.writerow([now, parts[1], parts[2]])
            pir_file.flush()
        elif parts[0] == "DHT" and len(parts) == 4 and parts[1].isdigit():
            dht_writer.writerow([now, parts[1], parts[2], parts[3]])
            dht_file.flush()
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
    pir_file.close()
    dht_file.close()
