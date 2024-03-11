from flask import Flask, request, jsonify
import voice_auth
import os

app = Flask(__name__)

# Directory where uploaded files will be saved
UPLOAD_FOLDER = 'voiceuploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/verify-voice', methods=['POST'])
def verify_voice():
    if 'voice' not in request.files:
        return jsonify({"error": "No voice file part"}), 400
    
    voice_file = request.files['voice']
    
    if voice_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if voice_file:

        result = voice_auth.recognize(voice_file)
    try:
        result = voice_auth.recognize(voice_file)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-voice', methods=['POST'])
def verify_voice():
    voice_file = request.files['voice']
    name = request.form.get('name')
    if voice_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if voice_file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], voice_file.filename)
        voice_file.save(filepath)
    try:
        result= voice_auth.enroll(name, voice_file)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
