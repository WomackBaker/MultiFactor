from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def GetData():
    data = request.get_json()
    name = data.pop('user', None)
    if data:
        try:
            file_path = os.path.join('data', name + '.csv')
            file_exists = os.path.isfile(file_path)
            mode = 'a' if file_exists else 'w'
            with open(file_path, mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()  # Write headers to the file
                writer.writerow(data)
            return jsonify({"message": "Data processed and appended to CSV file successfully"}), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"message": "No data received"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30081, debug=True)