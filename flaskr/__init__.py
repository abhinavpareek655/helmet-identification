from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K

app = Flask(__name__)

# Load your trained Keras model 
model = load_model('model.h5')

# Function to get model summary as a string
def get_model_summary():
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    model.summary()
    model_summary = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return model_summary

# Route for model insights and metrics
@app.route('/model-insights', methods=['GET'])
def model_insights():
    # Get the model architecture summary
    model_summary = get_model_summary()
    
    # Get model parameters
    total_params = model.count_params()

    # Get trainable parameters
    trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
    
    # Create the insights dictionary
    insights = {
        'model_architecture': model_summary,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'accuracy': None, 
        'loss': None,  

    }
    
    return jsonify(insights)

# Route to predict with the model (just as an example for testing the model's functionality)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Load the image and preprocess it
        img = load_img(file, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        # Assuming you have a mapping of class indices to class names
        class_names = ['helmet', 'nohelmet'] 
        predicted_label = class_names[predicted_class[0]]
        return jsonify({'predicted_label': predicted_label}), 200
    return jsonify({'error': 'Invalid file type'}), 400 
      