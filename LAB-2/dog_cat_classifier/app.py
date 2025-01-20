
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load models
rf_model = joblib.load('random_forest_model.joblib')

def preprocess_image(image, model_type):
    if model_type == 'cnn':
        img = cv2.resize(image, (256, 256))
        img = img.astype('float32') / 255.0
        img = img.reshape((1, 256, 256, 3))
    else:  # random forest
        img = cv2.resize(image, (64, 64))
        img = img.astype('float32') / 255.0
        img_flat = img.flatten().reshape(1, -1)
        return img_flat
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        model_type = request.form['model']

        # Read and preprocess image
        nparr = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'result': 'Error', 'confidence': 0})

        processed_image = preprocess_image(image, model_type)

        if model_type == 'cnn':
            prediction = cnn_model.predict(processed_image)[0][0]
            result = 'Dog' if prediction > 0.5 else 'Cat'
            confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        else:  # random forest
            prediction = rf_model.predict(processed_image)[0]
            probability = rf_model.predict_proba(processed_image)[0]
            result = 'Dog' if prediction == 0 else 'Cat'  # Note the change here to match your RF model
            confidence = float(probability[1] if prediction == 0 else probability[0])

        return jsonify({
            'result': result,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'result': 'Error', 'confidence': 0})

if __name__ == '__main__':
    app.run(debug=True)