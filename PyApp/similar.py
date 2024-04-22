import random
from datetime import datetime, timedelta

def similar_fake_data(data, num):
    # Generate similar user
    user = data["user"]
    user = user + str("."+str(num+1))

    # Generate similar latitude and longitude
    latitude = float(data["latitude"]) + round(random.uniform(-0.5, 0.5), 6)
    longitude = float(data["longitude"]) + round(random.uniform(-0.5, 0.5), 6)

    # Generate a fake IP address (using the last two octets from the initial one)
    ip_octets = data["ipString"].split('.')
    new_ip_address = f"{ip_octets[0]}.{ip_octets[1]}.{random.randint(0, 255)}.{random.randint(0, 255)}"

    # Generate a fake time close to the initial time
    initial_time = datetime.fromtimestamp(data["currentTime"] / 1000.0)
    fake_time = initial_time + timedelta(days=random.randint(-5, 5), hours=random.randint(-1, 1), minutes=random.randint(-30, 30))

    # Generate similar availableMemory, rssi, Processors, Battery, systemPerformance, accel, gyro, and magnet
    availableMemory = data["availableMemory"] + random.randint(-100000000, 100000000)
    rssi = data["rssi"] + random.randint(-10, 10)
    Processors = data["Processors"] + random.randint(-1, 1)
    Battery = min(100, max(0, data["Battery"] + random.randint(-10, 10)))
    systemPerformance = data["systemPerformance"] + random.randint(-1, 1)
    accel = data["accel"] + random.randint(-1, 1)
    gyro = data["gyro"] + random.randint(-1, 1)
    magnet = data["magnet"] + random.randint(-1, 1)

    return {
        "user": user,
        "latitude": latitude,
        "longitude": longitude,
        "ipString": new_ip_address,
        "currentTime": int(fake_time.timestamp() * 1000),
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