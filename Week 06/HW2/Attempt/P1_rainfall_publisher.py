"""
rainfall_publisher.py
Publishes rainfall measurements (0-2 inches) every 2 seconds to topic "weather/rain".
"""
import json
import time
import random
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "weather/rain"
CLIENT_ID = "rain_publisher"

client = mqtt.Client(CLIENT_ID)
client.connect(BROKER, PORT, keepalive=60)

try:
    count = 0
    while True:
        rain = round(random.uniform(0.0, 2.0), 3)
        payload = {
            "id": count,
            "rain_in": rain,
            "ts": time.time()
        }
        client.publish(TOPIC, json.dumps(payload))
        print(f"Published: {payload}")
        count += 1
        time.sleep(2.0)
except KeyboardInterrupt:
    print("Stopping publisher.")
finally:
    client.disconnect()
