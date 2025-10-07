"""
temp_humidity_publisher.py
Publishes JSON messages with temperature (50-52 F) and humidity (60-80 %)
every second to topic "weather/temphum".
"""
import json
import time
import random
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "weather/temphum"
CLIENT_ID = "temp_hum_publisher"

client = mqtt.Client(CLIENT_ID)
client.connect(BROKER, PORT, keepalive=60)

try:
    count = 0
    while True:
        temp = round(random.uniform(50.0, 52.0), 2)
        hum = round(random.uniform(60.0, 80.0), 2)
        payload = {
            "id": count,
            "temp_f": temp,
            "humidity_pct": hum,
            "ts": time.time()
        }
        client.publish(TOPIC, json.dumps(payload))
        print(f"Published: {payload}")
        count += 1
        time.sleep(1.0)
except KeyboardInterrupt:
    print("Stopping publisher.")
finally:
    client.disconnect()
