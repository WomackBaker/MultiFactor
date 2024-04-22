import requests
import similar
from flask import Flask, request, jsonify

NumofVariations = 3
app = Flask(__name__)

@app.route('/data', methods=['POST'])

def MakeData():
    data = request.get_json()
    if data:
        try:
            # Generate similar data based on the received JSON
            similar_data = similar.similar_fake_data(data, NumofVariations)
            
            # Send the similar data to the server
            response = requests.post('http://127.0.0.1:30081/data', json=similar_data)
            
            if response.status_code == 200:
                return jsonify({"message": "Data processed and sent successfully"}), 200
            else:
                return jsonify({"message": "Failed to send data to the other server",
                                "error": response.text}), response.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"message": "No data received"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30082)