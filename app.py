import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows React to access Flask API

# Load the trained model correctly
import joblib

model = joblib.load("Linear_regression_model.pkl")
print(model.n_features_in_)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([data["features"]])
        
        # Ensure 'model' is a trained ML model with a 'predict' method
        if not hasattr(model, "predict"):
            return jsonify({"error": "Loaded object is not a valid model!"})

        prediction = model.predict(features)
        price = round(prediction[0], 2)
        return jsonify({"prediction": price})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5004
            )
