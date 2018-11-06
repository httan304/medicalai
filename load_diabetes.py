"""Creates and compares classification models using sample clinical data.

Please use this example to learn about healthcareai before moving on to the next example.

If you have not installed healthcare.ai, refer to the instructions here:
  http://healthcareai-py.readthedocs.io

To run this example:
  python3 example_classification_1.py

This code uses the diabetes sample data in datasets/data/diabetes.csv.
"""
import pandas as pd
import healthcareai

def main():
    
    # load model and make prediction
    trained_model = healthcareai.load_saved_model('2018-11-05T17-02-43_DecisionTreeClassifier.pkl', debug=True)
    print('trained_model')
    print(trained_model.features)
    # test model
    prediction_dataframe = pd.DataFrame({'Pregnancies': [1] ,'Glucose': [85], 'BloodPressure': [66], 'SkinThickness': [29], 'Insulin': [0], 'BMI': [26.6], 'DiabetesPedigreeFunction': [0.351], 'Age': [31]})
    # make prediction
    predictions = trained_model.make_predictions(prediction_dataframe)
    print('\n\n-------------------[ Predictions ]----------------------------------------------------\n')
    print(predictions.head())

if __name__ == "__main__":
    main()
