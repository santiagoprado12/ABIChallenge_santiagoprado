"""module with classes for titanic user interface."""

import streamlit as st
import pandas as pd
import requests


# PredictionAPI class to handle communication with the API
class PredictionAPI:
    """This class handles the communication between the app and the prediction API.

    It sends requests to the API for both single and batch predictions and processes the responses.
    """

    def __init__(self, api_base_url: str):
        """Initializes the API URLs for single and batch predictions based on the base API URL.

        Parameters:
        api_base_url (str): The base URL of the API.
        """
        self.single_prediction_url = f"{api_base_url}/v1/prediction"
        self.batch_prediction_url = f"{api_base_url}/v1/batch_prediction"

    def get_single_prediction(self, data: dict):
        """Sends a single prediction request to the API.

        Parameters:
        data (dict): A dictionary containing passenger data.

        Returns:
        dict: The API response containing the prediction or None if an error occurs.
        """
        try:
            response = requests.post(self.single_prediction_url, json=data)
            if response.status_code == 200:
                return response.json()  # Expected to return a dict like {"Survived": 1}
            else:
                st.error("Error: Unable to get prediction from the API.")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
        return None

    def get_batch_predictions(self, data: list):
        """Sends a batch prediction request to the API.

        Parameters:
        data (list): A list of dictionaries where each dictionary contains a passenger's data.

        Returns:
        list: A list of predictions or None if an error occurs.
        """
        try:
            response = requests.post(
                self.batch_prediction_url, json={"batch_data": data}
            )
            if response.status_code == 200:
                return response.json()[
                    "Survived"
                ]  # Expected to return {"Survived": [1, 0, 1]}
            else:
                st.error("Error: Unable to get batch predictions from the API.")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
        return None


# InputHandler class to manage inputs from user and CSV
class InputHandler:
    """This class is responsible for managing user input, either manually entered or uploaded via CSV."""

    def get_manual_input(self):
        """Collects manual input from the user for a single prediction.

        Returns:
        dict: A dictionary containing the user-provided passenger data.
        """
        # Collecting passenger details from the user using Streamlit input widgets
        pclass = st.selectbox("Passenger Class (1st, 2nd, 3rd)", [1, 2, 3])
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        siblings_spouses = st.number_input(
            "Number of Siblings/Spouses Aboard", min_value=0, value=0
        )
        parents_children = st.number_input(
            "Number of Parents/Children Aboard", min_value=0, value=0
        )
        fare = st.number_input("Ticket Fare", min_value=0.0, value=30.0)
        name = st.text_input("Passenger Name", value="NN")
        cabin = st.text_input("Passenger Cabin", value="Any")
        Embarked = st.selectbox("Embarked", ["S", "C", "Q"])
        Ticket = st.text_input("Ticket ID", value="Any")

        # Creating a dictionary with all the passenger data to send to the API
        passenger_data = {
            "PassengerId": 0,
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": siblings_spouses,
            "Parch": parents_children,
            "Ticket": Ticket,
            "Fare": fare,
            "Cabin": cabin,
            "Embarked": Embarked,
        }
        return passenger_data

    def get_csv_input(self):
        """Handles CSV upload by the user and checks if the uploaded CSV contains the required columns.

        Returns:
        pd.DataFrame: The DataFrame containing the uploaded passenger data or None if the file is invalid.
        """
        # Allowing the user to upload a CSV file
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Checking if the CSV contains the necessary columns
            expected_columns = [
                "PassengerId",
                "Pclass",
                "Name",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked",
            ]
            if all(col in df.columns for col in expected_columns):
                return df  # Return the DataFrame if valid
            else:
                st.error(
                    f"CSV file is missing one or more required columns: {expected_columns}"
                )
        return None


# App class to orchestrate the interface
class App:
    """This class orchestrates the Streamlit interface, managing both manual input and batch prediction options."""

    def __init__(self, api_base_url: str):
        """Initializes the App class by setting up the API client and input handler.

        Parameters:
        api_base_url (str): The base URL of the prediction API.
        """
        self.api_client = PredictionAPI(api_base_url)
        self.input_handler = InputHandler()

    def manual_prediction(self):
        """Handles manual input prediction by collecting user data and sending it to the API."""
        st.subheader("Manual Prediction")
        # Get user input through the InputHandler
        data = self.input_handler.get_manual_input()

        if st.button("Get Prediction"):
            # Send the data to the API and display the result
            prediction = self.api_client.get_single_prediction(data)
            if prediction is not None:
                if prediction["Survived"] == 1:
                    st.success("Prediction: This passenger would survive!")
                else:
                    st.error("Prediction: This passenger would not survive.")

    def batch_prediction(self):
        """Handles batch prediction by processing the uploaded CSV and sending the data to the API."""
        st.subheader("Batch Prediction")
        # Get the uploaded CSV input through the InputHandler
        df = self.input_handler.get_csv_input()

        if df is not None and st.button("Get Batch Predictions"):
            # Handle missing values, convert data to the correct types, and prepare for API request
            data = df.fillna(0)
            data["Age"] = data["Age"].astype(int)
            data["Cabin"] = data["Cabin"].astype(str)

            # Convert the DataFrame to a list of dictionaries
            data = data.to_dict(orient="records")

            # Send the data to the API and display the predictions
            predictions = self.api_client.get_batch_predictions(data)
            if predictions is not None:
                df["prediction"] = predictions  # Add predictions to the DataFrame
                st.write("Predictions:")
                st.write(df)  # Display the DataFrame with predictions

    def run(self):
        """The main method that runs the Streamlit app, allowing the user to choose between manual and batch prediction."""
        st.title("Titanic Survival Prediction")
        st.write("Choose between manual input or batch prediction using a CSV file.")

        # Let the user choose the mode of prediction (manual or batch)
        option = st.selectbox(
            "Select prediction mode:", ["Manual Input", "Batch Prediction via CSV"]
        )

        # Call the appropriate method based on the user's choice
        if option == "Manual Input":
            self.manual_prediction()
        else:
            self.batch_prediction()


# Instantiate the app with your base API URL and run it
api_base_url = "https://9k6yjdcqae.us-east-1.awsapprunner.com/"
app = App(api_base_url)
app.run()
