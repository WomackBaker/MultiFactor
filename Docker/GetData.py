from flask import Flask, request, jsonify
import requests
app = Flask(__name__)


@app.route('/get-data', methods=['POST'])
def GetData():
    
    data = request.get_json()
    if data:
        try:
            response = requests.post('http://127.0.0.1:8081/data', json=data)
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
    app.run(host='0.0.0.0', port=8080)
