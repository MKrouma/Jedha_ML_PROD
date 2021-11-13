from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def index():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        print(f"request : {req}")
        print(req.keys())

        # Check mandatory key
        if "input" in req.keys():
            # Load model
            reg = joblib.load("model.joblib")

            # Predict
            prediction = reg.predict(req["input"])
            prediction = prediction.tolist()

            # return prediction
            return jsonify({"predict": prediction}), 200
    return jsonify({"msg": "Error: not a JSON or no email key in your request"})


if __name__ == "__main__":
    app.run(debug=True)