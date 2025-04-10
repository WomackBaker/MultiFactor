import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Read the CSV file line by line and adjust the number of columns
rows = []
with open('output.csv', 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        if len(fields) < 26:
            fields.extend([''] * (26 - len(fields)))
        rows.append(fields)

# Create a DataFrame from the adjusted rows
df = pd.DataFrame(rows)

# Step 2: Separate the data into features and labels
# Assuming the first column is the label (User/Attacker) and the rest are features
X = df.iloc[:, 1:]  # Features
y = df.iloc[:, 0]   # Labels

# Step 3: Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine the features and labels back into DataFrames
train_df = pd.concat([y_train, X_train], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)

# Step 4: Save the train and test sets into separate CSV files
train_df.to_csv('train.csv', index=False, header=False)
test_df.to_csv('test.csv', index=False, header=False)