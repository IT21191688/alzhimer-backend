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
        "origins": ["http://localhost:3000", "http://localhost:5173"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'alzheimers_model.h5')

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image_bytes):
    """Preprocess image for model prediction."""
    try:
        # Open and preprocess image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((128, 128))
        x = np.array(img)
        x = x.astype('float32') / 255.0  # Normalize pixel values
        x = np.expand_dims(x, axis=0)
        return x
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Model server is running'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    logger.info("Received prediction request")
    
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
        
        # Read and preprocess image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][class_idx] * 100)
        
        # Define classes
        classes = [
            'Non Demented',
            'Mild Dementia',
            'Moderate Dementia',
            'Very Mild Dementia'
        ]
        
        # Prepare response
        result = {
            'prediction': classes[class_idx],
            'confidence': round(confidence, 2),
            'status': 'success',
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
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
