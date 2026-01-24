import yaml
import logging
import logging.config
import joblib
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Dependency check
try:
    import yaml  # PyYAML is imported as yaml
    import flask
except ImportError as e:
    raise ImportError(f"Missing required package: {str(e)}. Install via requirements.txt.")

# Load configuration from config.yml
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yml')
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Load logging configuration from logging.yml
logging_path = os.path.join(base_dir, 'config', 'logging.yml')
with open(logging_path, 'r') as logging_file:
    log_config = yaml.safe_load(logging_file)
    # Update log file path to use absolute path
    log_config['handlers']['file']['filename'] = os.path.join(base_dir, 'logs', 'threatvision.log')
    logging.config.dictConfig(log_config)
logger = logging.getLogger('main')

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model, scaler, and PCA (adjust paths based on config)
model_path = os.path.join(base_dir, config['models']['path'])
scaler_path = os.path.join(base_dir, config['models']['scaler_path'])
pca_path = os.path.join(base_dir, config['models']['pca_path'])
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)

# Verify all models are loaded
if not all([model, scaler, pca]):
    raise FileNotFoundError("One or more model files (model, scaler, or PCA) could not be loaded.")

logger.info(f"Loaded model from {model_path}, scaler from {scaler_path}, and PCA from {pca_path}")

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    logger.info("Root endpoint accessed")
    return jsonify({
        'name': 'ThreatVision AI API',
        'version': '1.0',
        'description': 'Network threat detection API using machine learning',
        'endpoints': {
            '/': 'API information (this endpoint)',
            '/health': 'Health check endpoint',
            '/predict': 'Prediction endpoint (POST)'
        }
    }), 200

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'}), 200

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json['data']  # Expect a list of feature values
        logger.debug(f"Received data: {data}")

        # Input validation
        if not isinstance(data, list) or len(data) != pca.components_.shape[1]:
            raise ValueError(f"Expected a list of {pca.components_.shape[1]} features, got {len(data)}")

        # Convert input data to numpy array and preprocess
        data_array = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data_array)
        data_pca = pca.transform(data_scaled)

        # Make prediction with probabilities
        prediction = model.predict(data_pca)
        try:
            prediction_proba = model.predict_proba(data_pca)
            predicted_class = np.argmax(prediction_proba, axis=1)[0]
            return jsonify({
                'prediction': int(predicted_class),
                'probability': float(prediction_proba[0][predicted_class])
            })
        except:
            # Fallback if predict_proba is not available
            return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host=config['api']['host'], port=config['api']['port'], debug=False)