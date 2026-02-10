# Week 06/HW2/Attempt/P1_temp_humidity_subscriber.py

#import libraries
import json
import paho.mqtt.client as mqtt

#define variables
BROKER = "test.mosquitto.org" 
PORT = 1883 # default MQTT port
TOPIC = "weather/temphum"
CLIENT_ID = "temphum_sub_forever"

#define callback functions
def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg): 
    try:
        payload = json.loads(msg.payload.decode()) # if payload is JSON
    except Exception: # if payload is not JSON
        payload = msg.payload.decode()
    print(f"Topic: {msg.topic}, Payload: {payload}")

#initialize mqtt client
client = mqtt.Client(CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message

#connect to broker
client.connect(BROKER, PORT, keepalive=60)
print("Starting loop_forever (press Ctrl+C to exit)...")
client.loop_forever()  # blocking; will run indefinitely
