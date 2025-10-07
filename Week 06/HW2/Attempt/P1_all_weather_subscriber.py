"""
all_weather_subscriber.py
Subscribes to "weather/#", uses client.loop_start(), stops after 50 messages using loop_stop().
"""
import json
import time
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "weather/#"
CLIENT_ID = "all_weather_sub"

count = 0
MAX_MESSAGES = 50

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    global count
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        payload = msg.payload.decode()
    print(f"[{count+1}] Topic: {msg.topic}, Payload: {payload}")
    count += 1
    if count >= MAX_MESSAGES:
        print("Reached max messages. Stopping client loop.")
        client.loop_stop()  # stops the background loop thread

client = mqtt.Client(CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=60)
client.loop_start()  # non-blocking, runs in background

# Keep main thread alive until loop_stop is called
while client.is_connected() and count < MAX_MESSAGES:
    time.sleep(0.1)

client.disconnect()
print("Disconnected.")
