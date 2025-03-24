from flask import Blueprint, request, jsonify
from flask_cors import cross_origin

data_pipeline = Blueprint('data_pipeline', __name__)

@data_pipeline.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    # FUTURE: use data to make predictions
    print("Received:", data)
    return jsonify({"predicted_enjoyment": 10})