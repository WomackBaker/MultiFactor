from flask import Flask, request, jsonify
import os
from deepface import DeepFace

app = Flask(__name__)

# Invidual file to run voice recognition using flask

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
