# Week 06/HW2/Attempt/P1_all_weather_subscriber.py
#import libraries
import paho.mqtt.client as mqtt
import json

# define variables
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "siona/weather/#"  # listens to all weather topics
CLIENT_ID = "all_weather_subscriber"

data_points = []

# define callback functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully!")
        client.subscribe(TOPIC)
    else:
        print("Connection failed. Code:", rc)

def on_message(client, userdata, msg):
    global data_points
    payload = json.loads(msg.payload.decode())
    data_points.append(payload)
    print(f"Received: {payload}")
    if len(data_points) >= 50:
        print("Received 50 data points. Stopping...")
        client.disconnect()  # stops the loop and disconnects

# initialize mqtt client
client = mqtt.Client(client_id=CLIENT_ID)
client.on_connect = on_connect
client.on_message = on_message

# connect to broker
client.connect(BROKER, PORT, 60)

# start the loop
print("Listening for all weather data...")
client.loop_forever()  # blocking, will run until disconnect
