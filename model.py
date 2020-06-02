"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
import category_encoders as ce
from datetime import timedelta


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    clean_test_df = data.copy(deep=True)
    clean_test_df["Temperature"].fillna(clean_test_df["Temperature"].mean(), inplace=True)
    clean_test_df["Rainfall"] = clean_test_df["Precipitation in millimeters"]
    clean_test_df["Rainfall"]. loc[(clean_test_df["Rainfall"] >= 0)] = 1
    clean_test_df["Rainfall"].fillna(0, inplace=True)
    def seconds_after_midnight(input_df):
        df = input_df.copy(deep=True)
        placement_time = []
        confirmation_time = []
        arrival_at_pickup_time = []
        pickup_time = []
        arrival_at_destination_time = []
        for time in df["Placement - Time"]:
            i = 0
            if time[-2:] == "PM":
                i = 12
            split_time = time.split(":")
            placement_time.append(timedelta(hours=int(split_time[0].replace(" ", "")) + i, minutes=int(split_time[1]), seconds=int(split_time[2][:2])).total_seconds())
        for time in df["Confirmation - Time"]:
            i = 0
            if time[-2:] == "PM":
                i = 12
            split_time = time.split(":")
            confirmation_time.append(timedelta(hours=int(split_time[0].replace(" ", "")) + i, minutes=int(split_time[1]), seconds=int(split_time[2][:2])).total_seconds())
        for time in df["Arrival at Pickup - Time"]:
            i = 0
            if time[-2:] == "PM":
                i = 12
            split_time = time.split(":")
            arrival_at_pickup_time.append(timedelta(hours=int(split_time[0].replace(" ", "")) + i, minutes=int(split_time[1]), seconds=int(split_time[2][:2])).total_seconds())
        for time in df["Pickup - Time"]:
            i = 0
            if time[-2:] == "PM":
                i = 12
            split_time = time.split(":")
            pickup_time.append(timedelta(hours=int(split_time[0].replace(" ", "")) + i, minutes=int(split_time[1]), seconds=int(split_time[2][:2])).total_seconds())
        df["placement_time"] = placement_time
        df["confirmation_time"] = confirmation_time
        df["arrival_at_pickup_time"] = arrival_at_pickup_time
        df["pickup_time"] = pickup_time
        return df
    def data_preprocessing(df):
        duplicate_df = df.copy(deep=True)
        duplicate_df["Temperature"].fillna(round(duplicate_df["Temperature"].mean(), 0), inplace=True)
        columns_to_drop = ["Vehicle Type",
                       "Placement - Time", "Confirmation - Day of Month",
                       "Confirmation - Weekday (Mo = 1)", "Confirmation - Time",
                       "Arrival at Pickup - Day of Month", "Arrival at Pickup - Weekday (Mo = 1)",
                       "Arrival at Pickup - Time", "Pickup - Time",
                       "Pickup Lat", "Pickup Long", "Destination Lat", "Destination Long", "Rider Id",
                       "Order No", "User Id"]
        duplicate_df["Personal or Business"] = duplicate_df["Personal or Business"].replace(["Personal"], 0)
        duplicate_df["Personal or Business"] = duplicate_df["Personal or Business"].replace(["Business"], 1)
        duplicate_df.drop(columns_to_drop, axis=1, inplace=True)
        return duplicate_df
    clean_test_df['Personal or Business'].loc[(clean_test_df['Personal or Business'] == 'Business')] = 1
    clean_test_df['Personal or Business'].loc[(clean_test_df['Personal or Business'] == 'Personal')] = 0
    clean_test = data_preprocessing(seconds_after_midnight(clean_test_df))
    clean_test.drop(['Precipitation in millimeters', 'Rainfall', 'confirmation_time', 'arrival_at_pickup_time'], axis=1, inplace=True)
    ce_one_hot = ce.OneHotEncoder(cols=["Platform Type", "Personal or Business"])
    x_test_data = ce_one_hot.fit_transform(clean_test.iloc[:, :])
    # Convert the json string to a python dictionary object
    #feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    #feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Pickup Lat','Pickup Long',
    #                                    'Destination Lat','Destination Long']]
    # ------------------------------------------------------------------------

    return x_test_data

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
