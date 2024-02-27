from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Directory where uploaded files will be saved
UPLOAD_FOLDER = 'uploads'
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
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], voice_file.filename)
        voice_file.save(filepath)

        #TODO: Implement the verification logic
        
        is_verified = True  # This should be replaced with the actual verification logic

        return jsonify({"is_verified": is_verified})

if __name__ == '__main__':
    app.run(debug=True)
