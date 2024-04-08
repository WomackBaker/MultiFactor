import random
import faker
from datetime import datetime, timedelta
import requests

# Initialize a Faker generator
fake = faker.Faker()

# Generate a fake latitude and longitude
latitude = fake.latitude()
longitude = fake.longitude()

# Generate a fake IP address
ip_address = fake.ipv4()

# Generate a fake current time, +/- 30 days from now
fake_time = datetime.now() + timedelta(days=random.randint(-30, 30), hours=random.randint(-23, 23), minutes=random.randint(-59, 59))

# Generate fake memory size for a phone, in GB, assuming a range between 16GB to 512GB
memory_size = f"{random.choice([16, 32, 64, 128, 256, 512])}GB"
data1 ={
    "user": "User",
    "latitude": latitude,
    "longitude": longitude,
    "ip": ip_address,
    "time": fake_time.strftime("%Y-%m-%d %H:%M:%S"),
    "memory": memory_size
}

import similar

def loging(data1):
    for i in range(10):
        data = similar.similar_fake_data(data1 , i)
        response = requests.post("http://127.0.0.1:8080/get-data", json=data)
        print(response.status_code)


for i in range(10):
    loging(data1)