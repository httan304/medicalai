from flask import Flask, url_for, request, Response, json

import pandas as pd
import healthcareai
import json
import numpy

app = Flask(__name__)

@app.route('/')
def index():
    return 'index'

@app.route('/predict', methods=['GET', 'POST'])
def makePrediction():
    print('make prediction')
    if request.method == 'POST':
        body = request.json

        # load model and make prediction
        trained_model = healthcareai.load_saved_model('2018-11-05T17-02-43_DecisionTreeClassifier.pkl', debug=True)
        print('trained_model')
        print(trained_model.features)
        # test model
        # prediction_dataframe = pd.DataFrame({'Pregnancies': [1] ,'Glucose': [85], 'BloodPressure': [66], 'SkinThickness': [29], 'Insulin': [0], 'BMI': [26.6], 'DiabetesPedigreeFunction': [0.351], 'Age': [31]})
        prediction_dataframe = pd.DataFrame({'Pregnancies': [body['Pregnancies']], 'Glucose': [body['Glucose']], 'BloodPressure': [body['BloodPressure']], 'SkinThickness': [body['SkinThickness']], 'Insulin': [body['Insulin']], 'BMI': [body['BMI']], 'DiabetesPedigreeFunction': [body['DiabetesPedigreeFunction']], 'Age': [body['Age']]})
        # make prediction
        predictions = trained_model.make_predictions(prediction_dataframe)
        print(predictions.get_values())
        result = predictions.to_json(orient='records')
        resp = app.response_class(result, status=200, mimetype='application/json')
        return resp
    else:
        return 'index'