from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "*"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'alzheimers_model.h5')

# Initialize model variable
model = None

def load_model_on_startup():
    """Load the model at startup or return a default response if it fails."""
    global model
    try:
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Function to map prediction to class name
def names(number):
    if number == 0:
        return 'Non Demented'
    elif number == 1:
        return 'Mild Dementia'
    elif number == 2:
        return 'Moderate Dementia'
    elif number == 3:
        return 'Very Mild Dementia'
    else:
        return 'Error in Prediction'

@app.route('/', methods=['GET'])
def root():
    """Root endpoint to confirm API is running."""
    return jsonify({
        'status': 'online',
        'message': 'Alzheimer\'s Prediction API is running'
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global model
    if model is None:
        model_loaded = load_model_on_startup()
        if not model_loaded:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Failed to load model'
            }), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'Model server is running'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    logger.info("Received prediction request")
    
    global model
    if model is None:
        model_loaded = load_model_on_startup()
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
    
    try:
        # Validate request
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({
                'error': 'No image provided',
                'detail': 'No image file in request'
            }), 400

        file = request.files['image']
        if file.filename == '':
            logger.warning("Empty filename received")
            return jsonify({
                'error': 'No image provided',
                'detail': 'Empty filename'
            }), 400

        logger.info(f"Processing file: {file.filename}")
        
        # Read and preprocess image exactly as in your test code
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        x = np.array(img.resize((128, 128)))
        x = x.reshape(1, 128, 128, 3)
        
        # Use predict_on_batch as in your test code
        prediction = model.predict_on_batch(x)
        logger.info(f"Raw prediction values: {prediction[0]}")
        
        # Find the class with the highest probability using np.where
        class_idx = np.where(prediction == np.amax(prediction))[1][0]
        confidence = float(prediction[0][class_idx] * 100)
        
        # Get prediction class name
        prediction_class = names(class_idx)
        
        # Prepare response
        result = {
            'prediction': prediction_class,
            'confidence': round(confidence, 2),
            'status': 'success',
            'class_index': int(class_idx),
            'raw_predictions': {
                'Non Demented': float(prediction[0][0] * 100),
                'Mild Dementia': float(prediction[0][1] * 100),
                'Moderate Dementia': float(prediction[0][2] * 100),
                'Very Mild Dementia': float(prediction[0][3] * 100)
            }
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Load model at startup
    load_model_on_startup()
    
    # Get port from environment variable with a default of 10000 (Render's default)
    port = int(os.getenv('PORT', 10000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )