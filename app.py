from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # for connecting with HTML frontend

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (needed for HTML + JS)

# Load the trained model
model = joblib.load("iris_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Extract inputs
        sepal_length = float(data["sepal_length"])
        sepal_width  = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width  = float(data["petal_width"])

        # Validate input ranges (Iris dataset known range)
        if not (4.0 <= sepal_length <= 8.0): raise ValueError("Invalid sepal length")
        if not (2.0 <= sepal_width  <= 4.5): raise ValueError("Invalid sepal width")
        if not (1.0 <= petal_length <= 7.0): raise ValueError("Invalid petal length")
        if not (0.1 <= petal_width  <= 2.5): raise ValueError("Invalid petal width")

        # Predict
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

        return jsonify({"species": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
