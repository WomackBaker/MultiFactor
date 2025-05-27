import csv
import os
import random

userbaseline = []
users = []

# Read user data and baselines
for i in os.listdir('./data'):
    if i.endswith('.csv'):
        id = i.split('.')[0]
        if id not in users:
            users.append(id)
            with open(f'./data/{i}', 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                userbaseline.append(next(reader))
        for j in range(len(users)):
            if users[j] == id:
                baseline = userbaseline[j]
                with open(f'./data/{i}', 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    with open(f'./output.csv', 'a', newline='') as outputfile:  # Open in append mode
                        writer = csv.writer(outputfile)
                        next(reader)
                        line = next(reader)
                        output = []
                        output.append(id)
                        for k in range(len(line)):
                            if baseline[k] == line[k]:
                                output.append('1')
                            else:
                                output.append('0')
                        writer.writerow(output)

# Generate Attackers
attackers = 200
num_users = len(users)
attackers_per_user = attackers // num_users

for i in range(num_users):
    id = users[i]
    baseline = userbaseline[i]
    for j in range(attackers_per_user):
        with open(f'./output.csv', 'a', newline='') as outputfile:  # Open in append mode
            writer = csv.writer(outputfile)
            output = []
            output.append(id)
            for k in range(len(baseline)):
                if k in [0, 1, 2, 3, 4, 6, 7, 8, 11, 16, 17, 20]:
                    output.append(random.choice(['0', '1']))
                else:
                    output.append('1')
            output.append('1')
            writer.writerow(output)