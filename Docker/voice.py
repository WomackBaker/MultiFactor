from flask import Flask, request, jsonify
import voice_auth
import os

app = Flask(__name__)

# Directory where uploaded files will be saved
UPLOAD_FOLDER = './data/wav'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/verify_voice', methods=['POST'])
def verify_voice():
    if 'voice' not in request.files:
        return jsonify({"error": "No voice file part"}), 400
    
    voice_file = request.files['voice']
    name = request.form.get('name')
    
    if voice_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if voice_file:
        try:
            result = voice_auth.recognize(voice_file)
            if result == name:
                return jsonify({'result': result}), 200
            else:
                return jsonify({'error': 'Voice not recognized'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
@app.route('/add_voice', methods=['POST'])
def add_voice():
    voice_file = request.files['voice']
    name = request.form.get('name')
    if voice_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if voice_file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], name + '.wav')
        
        if os.path.exists(filepath):
            return jsonify({"error": "Files for this name already exist"}), 400

        voice_file.save(filepath)
    try:
        print("name", name, "voice_file", filepath)
        result= voice_auth.enroll(name, filepath)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
