from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('helmet_detection_model.h5')

# Model metrics (replace with your actual metrics)
model_metrics = {
    'accuracy': 95.2,
    'precision': 94.8,
    'recall': 93.7,
    'f1_score': 94.2,
    'training_samples': 5000,
    'validation_samples': 1000,
    'model_architecture': 'MobileNetV2',
    'epochs': 20
}

@app.route('/')
def index():
    return render_template('index.html', 
                          prediction_made=False, 
                          **model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Save the uploaded image
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Preprocess the image for prediction
    img = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0]) * 100
    wearing_helmet = bool(prediction[0][0] > 0.5)
    
    # Render template with prediction results
    return render_template('index.html',
                          prediction_made=True,
                          image_path=filename,
                          wearing_helmet=wearing_helmet,
                          confidence=round(confidence, 2) if wearing_helmet else round(100-confidence, 2),
                          **model_metrics)

if __name__ == '__main__':
    app.run(debug=True)