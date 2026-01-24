from flask import Blueprint, request, jsonify
from api.utils.helpers import load_model, preprocess_input

endpoints = Blueprint('endpoints', __name__)

# Load the trained model
model = load_model('models/threatvision_rf_model.pkl')

@endpoints.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
    
    # Preprocess the input data
    processed_data = preprocess_input(data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return jsonify({'prediction': prediction.tolist()})