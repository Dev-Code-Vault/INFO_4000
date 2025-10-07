# Week 06/HW2/Attempt/P1_rainfall_publisher.py
#publish rainfall measurements (0-2 inches) every 2 seconds to topic "weather/rain".

#import libraries
import json
import time
import random
import paho.mqtt.client as mqtt

#define variables
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "weather/rain"
CLIENT_ID = "rain_publisher"

#initialize mqtt client
client = mqtt.Client(CLIENT_ID)
client.connect(BROKER, PORT, keepalive=60)

#start the loop
try:
    count = 0
    while True:
        rain = round(random.uniform(0.0, 2.0), 3) 
        payload = {
            "id": count,
            "rain_in": rain,
            "ts": time.time()
        }
        client.publish(TOPIC, json.dumps(payload)) #publish
        print(f"Published: {payload}")
        count += 1
        time.sleep(2.0) #publish every 2 seconds
except KeyboardInterrupt: #stop
    print("Stopping publisher.")
finally:
    client.disconnect() #disconnect
