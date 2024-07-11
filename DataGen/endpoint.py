import requests
import similar
from flask import Flask, request, jsonify

NumofVariations = 5
app = Flask(__name__)

@app.route('/data', methods=['POST'])

def MakeData():
    data = request.get_json()
    responses = []  # Create a list to store responses
    if data:
        for i in range(NumofVariations):
            try:
                # Generate similar data based on the received JSON
                similar_data = similar.similar_fake_data(data, i)
                # Send the similar data to the server
                response = requests.post('http://host.docker.internal:30081/data', json=similar_data)
                
                if response.status_code == 200:
                    responses.append({"message": "Data processed and sent successfully"})
                else:
                    responses.append({"message": "Failed to send data to the other server",
                                      "error": response.text})
            except Exception as e:
                print(f"Error on iteration {i}: {str(e)}")
                responses.append({"error": str(e)})
    else:
        return jsonify({"message": "No data received"}), 400

    return jsonify(responses), 200 if all(r.get('error') is None for r in responses) else 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30082)