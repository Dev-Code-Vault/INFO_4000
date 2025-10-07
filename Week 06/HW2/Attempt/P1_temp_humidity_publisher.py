# Week 06/HW2/Attempt/P1_temp_humidity_publisher.py

#import libraries
import paho.mqtt.client as mqtt
import json
import random
import time

#define variables
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "siona/weather/temp_humidity"
CLIENT_ID = "temp_humidity_publisher"

#initialize mqtt client
client = mqtt.Client(CLIENT_ID)
client.connect(BROKER, PORT, 60)

#start the loop
try:
    msg_id = 0
    while True: #infinite loop
        temp_f = round(random.uniform(50, 52), 2)
        humidity_pct = round(random.uniform(60, 80), 2)
        payload = {
            "id": msg_id,
            "temp_f": temp_f,
            "humidity_pct": humidity_pct,
            "ts": time.time()
        }
        client.publish(TOPIC, json.dumps(payload))
        print(f"Published: {payload}")
        msg_id += 1
        time.sleep(1)
except KeyboardInterrupt:
    print("Publisher stopped.")
    client.disconnect()
