from flask import Flask, request, jsonify
from deepface import DeepFace
import voice_auth
import requests
import os

app = Flask(__name__)

# Directory where uploaded files will be saved
UPLOAD_FOLDER = './data/wav'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/verify-image', methods=['POST'])
def verify_image():
    # Check if the post request has the file part for img1
    if 'img1' not in request.files:
        return jsonify({'error': 'Missing image'}), 400
    img1 = request.files['img1']
    # Check for file name and allowed type
    if img1.filename == '' or not allowed_file(img1.filename):
        return jsonify({'error': 'No selected file or invalid file format'}), 400
    
    try:
        # Saving temporary image for the uploaded file
        temp_img1_path = "temp_img1.jpg"
        img1.save(temp_img1_path)
        img2_path = os.path.join(os.getcwd(), 'randomperson.jpg')
        # Using DeepFace to verify the images
        result = DeepFace.verify(img1_path=temp_img1_path, img2_path=img2_path)
        
        # Optionally, delete the temporary image file
        os.remove(temp_img1_path)
        
        # Return the result of the verification
        return jsonify({'result': result}), 200
    except Exception as e:
        # Optionally, ensure temporary files are deleted in case of error
        if os.path.exists(temp_img1_path):
            os.remove(temp_img1_path)
        return jsonify({'error': str(e)}), 500
    
@app.route('/verify-voice', methods=['POST'])
def verify_voice():
    if 'voice' not in request.files:
        return jsonify({"error": "No voice file part"}), 500
    
    voice_file = request.files['voice']
    name = request.form.get('name')
    
    if voice_file.filename == '':
        return jsonify({"error": "No selected file"}), 500
    
    if voice_file:
        try:
            result = voice_auth.recognize(voice_file)
            print(result)
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
        return jsonify({"error": "No selected file"}), 500
    
    if voice_file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], name + '.wav')
        
        if os.path.exists(filepath):
            return jsonify({"error": "Files for this name already exist"}), 500

        voice_file.save(filepath)
    try:
        print("name", name, "voice_file", filepath)
        result= voice_auth.enroll(name, filepath)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
