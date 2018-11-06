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
import healthcareai.common.data_cleaner as clean
import healthcareai.common.modeling as train_model

def main():
    # diabetes
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Load the included diabetes sample data
    dataframe = healthcareai.load_diabetes1()
    # clean data
    data_mod = clean.clean_data(dataframe, feature_names)
    X = data_mod[feature_names]
    y = data_mod.Outcome
    # choose top 4 model by cross validator kfold
    models = train_model.modelSelectionByKfold(X, y)
    models = models.head(4)
    # selection model with feature importance
    result = train_model.featureSelectionByModel(X, y, feature_names, models)
    # Train model
    trained_supervised_medical_model = healthcareai.TrainedSupervisedMedicalModel(result.model[0], result.features, 'DecisionTreeClassifier')
    trained_supervised_medical_model.save()


if __name__ == "__main__":
    main()
