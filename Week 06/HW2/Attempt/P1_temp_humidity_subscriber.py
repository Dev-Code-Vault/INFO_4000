"""
temp_humidity_subscriber_forever.py
Subscribes to "weather/temphum" and uses client.loop_forever() (blocking).
"""
import json
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "weather/temphum"
CLIENT_ID = "temphum_sub_forever"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        payload = msg.payload.decode()
    print(f"Topic: {msg.topic}, Payload: {payload}")

client = mqtt.Client(CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=60)
print("Starting loop_forever (press Ctrl+C to exit)...")
client.loop_forever()  # blocking; will run indefinitely
