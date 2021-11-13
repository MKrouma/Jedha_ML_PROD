import os
import joblib
import numpy as np
from flask_restx import Resource, Api
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
api = Api(app, version='1.0', title='White Winne Prediction',
    description='Jedha ML production project.',
)

@api.route('/predict')
class Predict(Resource):
    def post(self):
        # Check if request has a JSON content
        if request.json:
            # Get the JSON as dictionnary
            req = request.get_json()
            print(f"request : {req}")
            print(req.keys())

            # Check mandatory key
            if "input" in req.keys():
                # Load model
                model_file = os.path.join("./model/model.joblib")
                reg = joblib.load(model_file)

                # Predict
                prediction = reg.predict(req["input"])
                prediction = prediction.tolist()

                # return prediction
                return jsonify({"predict": prediction}), 200
        return jsonify({"msg": "Error: not a JSON or no email key in your request"})


# # Predict endpoint
# @app.route("/predict", methods=["POST"])
# def endpoint():
#     # Check if request has a JSON content
#     if request.json:
#         # Get the JSON as dictionnary
#         req = request.get_json()
#         print(f"request : {req}")
#         print(req.keys())

#         # Check mandatory key
#         if "input" in req.keys():
#             # Load model
#             model_file = os.path.join("./model/model.joblib")
#             reg = joblib.load(model_file)

#             # Predict
#             prediction = reg.predict(req["input"])
#             prediction = prediction.tolist()

#             # return prediction
#             return jsonify({"predict": prediction}), 200
#     return jsonify({"msg": "Error: not a JSON or no email key in your request"})


if __name__ == "__main__":
    app.run(debug=True)