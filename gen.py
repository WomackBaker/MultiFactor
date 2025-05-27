import csv
import random

# Define the number of users and attackers
num_users = 1000
num_attackers = 200

# Define the base data for users and attackers
base_user_data = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
base_attacker_data = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]

# Function to introduce slight variations in user data
def generate_user_data(base_data):
    return [random.choice([0, 1]) if random.random() < 0.1 else val for val in base_data]

# Function to generate attacker data with more 0s
def generate_attacker_data(base_data):
    return [0 if random.random() < 0.3 else val for val in base_data]

# Open the CSV file for writing
with open('SVM.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write user rows
    for i in range(1, num_users + 1):
        user_data = generate_user_data(base_user_data)
        row = [f'User {i}'] + user_data + [1]
        writer.writerow(row)
    
    # Write attacker rows
    for i in range(1, num_attackers + 1):
        attacker_data = generate_attacker_data(base_attacker_data)
        row = [f'Attacker {i}'] + attacker_data
        writer.writerow(row)