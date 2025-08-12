import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image  # Pillow library for image manipulation
import webbrowser
import threading

# Create Flask app
app = Flask(__name__)

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224

MODELS_CONFIG = {
    "covid": {
        "path": 'trained_medical_models/COVID19_XRay_Classifier.h5',
        "class_names": ['COVID', 'Normal', 'Pneumonia']
    },
    "brain_tumor": {
        "path": 'trained_medical_models/Brain_Tumor_MRI_Classifier.h5',
        "class_names": ['glioma', 'notumor', 'meningioma', 'pituitary']
    }
}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load both models once when the application starts ---
loaded_models = {}
for model_type, config in MODELS_CONFIG.items():
    model_path = config["path"]
    try:
        model = tf.keras.models.load_model(model_path)
        loaded_models[model_type] = model
        print(f"Model '{model_path}' ({model_type}) loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{model_path}' ({model_type}): {e}")
        loaded_models[model_type] = None

# --- Preprocessing function ---
def preprocess_image(img_path):
    """Prepare image for prediction."""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')
    if model_type not in loaded_models or loaded_models[model_type] is None:
        return jsonify({'error': f'Model for {model_type} not loaded.'}), 500

    model = loaded_models[model_type]
    class_names = MODELS_CONFIG[model_type]["class_names"]

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(filepath)
        processed_image = preprocess_image(filepath)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(predictions))

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f"{confidence:.2f}"
        })
    except ValueError as ve:
        return jsonify({'error': f'Image processing error: {ve}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# --- Open browser automatically ---
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

# --- Run App ---
if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=True, host='127.0.0.1', port=5000)
