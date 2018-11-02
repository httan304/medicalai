import pandas as pd

def clean_data(dataframe, feature_names):
    """
        Cleaning data from source
        Ignore/remove these cases : This is not actually possible in most cases because that would mean losing valuable information.It might work for “BMI”, “glucose ”and “blood pressure” whenever just a few invalid data points.
        Put average/mean values : This might work for some data sets, but in our case putting a mean value to the blood pressure column would send a wrong signal to the model.
        Avoid using features : It is possible to not use the features with a lot of invalid values for the model. This may work for “skin thickness” but its hard to predict that.
    """
    # Drop rows which contain missing values.
    # Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
    # ‘any’ : If any NA values are present, drop that row or column.
    dataframe = dataframe.dropna(axis='index', how='any')
    result = dataframe[(dataframe.Pregnancies != 0) & (dataframe.Glucose != 0) 
        & (dataframe.BloodPressure != 0) & (dataframe.SkinThickness != 0) & (dataframe.BMI != 0) 
        & (dataframe.Insulin != 0) & (dataframe.DiabetesPedigreeFunction != 0) & (dataframe.Age) != 0]
    return result
