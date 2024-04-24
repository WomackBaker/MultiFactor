from datetime import datetime, timedelta
import time
import random

def similar_fake_data(data, num):
    # Convert 'currentTime' from string to Unix timestamp in milliseconds
    initial_time = datetime.strptime(data["currentTime"], '%Y-%m-%d %H:%M:%S')
    fake_time = initial_time.replace(
        hour=random.randint(0, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )
    # Generate similar user
    user = data["user"]
    user = user + str("."+str(num+1))

    # Generate similar latitude and longitude
    latitude = float(data["latitude"]) + round(random.uniform(-0.5, 0.5), 6)
    longitude = float(data["longitude"]) + round(random.uniform(-0.5, 0.5), 6)

    # Generate a fake IP address (using the last two octets from the initial one)
    ip_octets = data["ipString"].split('.')
    new_ip_address = f"{ip_octets[0]}.{ip_octets[1]}.{ip_octets[2]}.{random.randint(0, 255)}"

    # Generate similar availableMemory, rssi, Processors, Battery, systemPerformance, accel, gyro, and magnet
    availableMemory = data["availableMemory"] + random.randint(-100000000, 100000000)
    rssi = data["rssi"] + random.randint(-10, 10)
    Processors = data["Processors"]
    Battery = min(100, max(0, data["Battery"] + random.randint(-10, 10)))
    systemPerformance = data["systemPerformance"] + random.randint(-1, 1)
    accel = data["accel"]
    gyro = data["gyro"]
    magnet = data["magnet"]

    return {
        "user": user,
        "latitude": latitude,
        "longitude": longitude,
        "ipString": new_ip_address,
        "currentTime": fake_time.strftime('%Y-%m-%d %H:%M:%S'),
        "availableMemory": availableMemory,
        "rssi": rssi,
        "timezone": data["timezone"],
        "Processors": Processors,
        "Battery": Battery,
        "Vendor": data["Vendor"],
        "Model": data["Model"],
        "systemPerformance": systemPerformance,
        "cpu": data["cpu"],
        "accel": accel,
        "gyro": gyro,
        "magnet": magnet
    }