import pandas as pd
import mysql.connector
from mysql.connector import Error
import os

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

def insert_into_table(connection, table_name, data_frame):
    cursor = connection.cursor()
    
    # Get columns from the dataframe
    columns = ', '.join(data_frame.columns)
    
    # Create a SQL insert statement
    insert_query = f"""
    INSERT INTO {table_name} (latitude, longitude, ipString, currentTime, availableMemory, rssi, timezone, Processors,
    Battery, Vendor, Model, systemPerformance, cpu, accel, gyro, magnet, screenWidth, screenLength, screenDensity, 
    hasTouchScreen, hasCamera, hasFrontCamera, hasMicrophone, hasTemperatureSensor)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    # Iterate over the dataframe rows
    for row in data_frame.itertuples(index=False):
        cursor.execute(insert_query, tuple(row))
    
    connection.commit()
    print(f"{cursor.rowcount} rows were inserted into the table {table_name}")

def load_csv_to_dataframe(csv_file_path):
    return pd.read_csv(csv_file_path)

# Function to process all CSV files in a folder
def process_folder(folder_path, connection, table_name):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            data_frame = load_csv_to_dataframe(file_path)
            insert_into_table(connection, table_name, data_frame)
            print(f"Finished processing file: {file_path}")

def main():
    
    folder_path = './data' 

    # MySQL database details
    host_name = 'CHANGME'
    user_name = 'CHANGEME'
    user_password = 'CHANGEME'
    db_name = 'CHANGEME'
    table_name = 'CHANGEME'  
    
    connection = create_connection(host_name, user_name, user_password, db_name)
    
    if connection:
        process_folder(folder_path, connection, table_name)

if __name__ == '__main__':
    main()
