import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/data', methods=['POST'])

def MakeData():
    data = request.get_json()
    print(data)
    response = requests.post('http://host.docker.internal:30082/data', json=data)
    if response.status_code == 200:
        return jsonify({"message": "Data processed and sent successfully"}), 200
    else:
        return jsonify({"message": "Failed to send data to the other server",
                        "error": response.text}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30083)