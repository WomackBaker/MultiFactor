import random
import faker
from datetime import datetime, timedelta
import requests
import similar
import csv
import os

NumofUsers = 1
NumofVariations = 1

def User(Usernum, NumofVariations):
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
        "user": "User"+str(Usernum+1),
        "latitude": latitude,
        "longitude": longitude,
        "ip": ip_address,
        "time": fake_time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory": memory_size
    }
    for i in range(NumofVariations):
        data = similar.similar_fake_data(data1 , i)
        with open("Users.csv", 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writerow(data)
        #response = requests.post("http://127.0.0.1:30081/data", json=data)
        #print(response.status_code)

for i in range(NumofUsers):
    data = User(i, NumofVariations)


    