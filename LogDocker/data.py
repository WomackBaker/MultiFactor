from flask import Flask, request, jsonify
import csv
import os
# Runs endpoint to receive log information and write to CSV file
app = Flask(__name__)
# Define the route to receive data
@app.route('/data', methods=['POST'])
def GetData():
    data = request.get_json()
    print(data)
    name = data.pop('user', None)
    if data:
        try:
            # Specify the CSV file path
            file_path = os.path.join('data', name + '.csv')
            # Check if the file already exists to decide whether to write headers
            file_exists = os.path.isfile(file_path)
            
            mode = 'a' if file_exists else 'w'
            with open(file_path, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                writer.writerow(data)
            return jsonify({"message": "Data processed and appended to CSV file successfully"}), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"message": "No data received"}), 400
# Hosts on port 30081
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30081, debug=True)
