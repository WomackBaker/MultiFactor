import random
import pandas as pd

# Define the original UUIDs
original_uuids = [
    "4f92d22c-66d0-4d3f-ad38-ceeec8771da0",
    "8ea9a20a-a277-417b-b067-d77aef14c7f7",
    "89b5b1d2-c0d6-4284-a636-306be8ed5d27",
    # Repeat or add more UUIDs as needed to match your data
] * 50  # Adjust replication to reach 200 rows if needed

# Define the columns
columns = [
    "uuid", "latitude", "longitude", "ipString", "currentTime", "availableMemory",
    "rssi", "timezone", "Processors", "Battery", "Vendor", "Model", "systemPerformance",
    "cpu", "accel", "gyro", "magnet", "screenWidth", "screenLength", "screenDensity",
    "hasTouchScreen", "hasCamera", "hasFrontCamera", "hasMicrophone", "hasTemperatureSensor"
]

# Generate realistic data
def generate_realistic_data(uuids):
    data = []
    for uuid_val in uuids:
        row = [
            uuid_val,  # Keep the UUID the same
            random.choice([0, 1]),  # Latitude
            random.choice([0, 1]),  # Longitude
            random.choice([0, 1]),  # IP String
            random.choice([0, 1]),  # Current time
            random.choice([0, 1]),  # Available memory
            random.choice([0, 1]),  # RSSI
            random.choice([0, 1]),  # Timezone
            random.choice([0, 1]),  # Processors count
            random.choice([0, 1]),  # Battery percentage
            random.choice([0, 1]),  # Vendor
            random.choice([0, 1]),  # Model
            random.choice([0, 1]),  # System performance
            random.choice([0, 1]),  # CPU count
            random.choice([0, 1]),  # Accelerometer
            random.choice([0, 1]),  # Gyroscope
            random.choice([0, 1]),  # Magnetometer
            random.choice([0, 1]),  # Screen width
            random.choice([0, 1]),  # Screen length
            random.choice([0, 1]),  # Screen density
            random.choice([0, 1]),  # Has touchscreen
            random.choice([0, 1]),  # Has camera
            random.choice([0, 1]),  # Has front camera
            random.choice([0, 1]),  # Has microphone
            random.choice([0, 1])   # Has temperature sensor
        ]
        data.append(row)
    return data

# Generate the data
realistic_data = generate_realistic_data(original_uuids)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(realistic_data, columns=columns)
df.to_csv("realistic_data.csv", index=False)

print("CSV file 'realistic_data.csv' has been generated.")
