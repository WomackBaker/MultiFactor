import random
from datetime import datetime, timedelta

def similar_fake_data(data, num):
    # Generate similar latitude and longitude
    user = data["user"]
    user = user + str("."+str(num+1))
    latitude = float(data["latitude"]) + round(random.uniform(-0.5, 0.5), 6)
    longitude = float(data["longitude"]) + round(random.uniform(-0.5, 0.5), 6)
    
    # Generate a fake IP address (using the last two octets from the initial one)
    ip_octets = data["ip"].split('.')
    new_ip_address = f"{ip_octets[0]}.{ip_octets[1]}.{random.randint(0, 255)}.{random.randint(0, 255)}"
    
    # Generate a fake time close to the initial time
    initial_time = datetime.strptime(data["time"], "%Y-%m-%d %H:%M:%S")
    fake_time = initial_time + timedelta(days=random.randint(-5, 5), hours=random.randint(-1, 1), minutes=random.randint(-30, 30))
    
    memory_sizes = [16, 32, 64, 128, 256, 512]
    initial_memory = int(data["memory"].replace("GB", ""))
    initial_memory_index = memory_sizes.index(initial_memory)
    # Ensure we select a memory size close to the initial one, within bounds
    new_memory_index = max(0, min(len(memory_sizes)-1, initial_memory_index + random.choice([-1, 0, 1])))
    memory_size = f"{memory_sizes[new_memory_index]}GB"
    
    return {
        "user": user,
        "latitude": latitude,
        "longitude": longitude,
        "ip": new_ip_address,
        "time": fake_time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory": memory_size
    }