import os
import joblib
import numpy as np
import pandas as pd
from flask_restx import Resource, Api, reqparse
from flask import Flask, request, jsonify, make_response



app = Flask(__name__)
api = Api(app, version='1.0', title='White Winne Prediction',
    description='Jedha ML production project.',
)


@api.route('/fetchdata', methods=["POST"])
class FetchData(Resource):
    def post(self) :

        # Check if request has a JSON content
        if request.json:
            # Get the JSON as dictionnary
            req = request.get_json()
            print(f"request : {req}")
            print(req.keys())

            # Check mandatory key
            if "column" in req.keys():
                # col
                col = req["column"]

                # read data
                data = pd.read_csv("./data/winequality.csv")

                if col in data.columns : 
                    data = {col: data.loc[:, col].to_list()}

                    return make_response(jsonify(data), 200)
                else :
                    return jsonify({"msg": "Error : col to fetch not in dataset champs!"})
            else :
                return jsonify({"msg": "Error : column needed in json post files!"})
        return jsonify({"msg": "Error: not a JSON or no email key in your request"})


@api.route('/predict', methods=["POST"])
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
                # return jsonify({"predict": prediction}), 200
                data = {"predict": prediction}
                return make_response(jsonify(data), 200)
        return jsonify({"msg": "Error: not a JSON or no email key in your request"})


if __name__ == "__main__":
    app.run(debug=True)