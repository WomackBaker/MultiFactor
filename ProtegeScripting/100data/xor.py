import csv
import os

userbaseline = []
users = []

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
                            #print(baseline[k] +" vs " + line[k])
                            if baseline[k] == line[k]:
                                output.append('1')
                            else:
                                output.append('0')
                        
                        if id not in ('.31'):
                            output.append('ATTCK')
                        writer.writerow(output)