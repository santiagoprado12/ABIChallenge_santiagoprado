import streamlit as st
import pandas as pd
import requests


# PredictionAPI class to handle communication with the API
class PredictionAPI:
    def __init__(self, api_base_url: str):
        self.single_prediction_url = f"{api_base_url}/v1/prediction"
        self.batch_prediction_url = f"{api_base_url}/v1/batch_prediction"

    def get_single_prediction(self, data: dict):
        """Send a single prediction request to the API."""
        try:
            response = requests.post(self.single_prediction_url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                st.error("Error: Unable to get prediction from the API.")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
        return None

    def get_batch_predictions(self, data: list):
        """Send a batch prediction request to the API."""
        try:
            response = requests.post(
                self.batch_prediction_url, json={"batch_data": data}
            )
            if response.status_code == 200:
                return response.json()["Survived"]
            else:
                st.error("Error: Unable to get batch predictions from the API.")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
        return None


# InputHandler class to manage inputs from user and CSV
class InputHandler:
    def get_manual_input(self):
        """Collect input from manual fields."""
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
        """Handle CSV upload and return the DataFrame if valid."""
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
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
                return df
            else:
                st.error(
                    f"CSV file is missing one or more required columns: {expected_columns}"
                )
        return None


# App class to orchestrate the interface
class App:
    def __init__(self, api_base_url: str):
        self.api_client = PredictionAPI(api_base_url)
        self.input_handler = InputHandler()

    def manual_prediction(self):
        """Handle manual input prediction."""
        st.subheader("Manual Prediction")
        data = self.input_handler.get_manual_input()

        if st.button("Get Prediction"):
            prediction = self.api_client.get_single_prediction(data)
            if prediction is not None:
                if prediction["Survived"] == 1:
                    st.success("Prediction: This passenger would survive!")
                else:
                    st.error("Prediction: This passenger would not survive.")

    def batch_prediction(self):
        """Handle batch prediction from CSV."""
        st.subheader("Batch Prediction")
        df = self.input_handler.get_csv_input()

        if df is not None and st.button("Get Batch Predictions"):

            data = df.fillna(0)
            data["Age"] = data["Age"].astype(int)
            data["Cabin"] = data["Cabin"].astype(str)

            data = data.to_dict(orient="records")

            predictions = self.api_client.get_batch_predictions(data)
            if predictions is not None:
                df["prediction"] = predictions
                st.write("Predictions:")
                st.write(df)

    def run(self):
        """Main method to run the Streamlit app."""
        st.title("Titanic Survival Prediction")
        st.write("Choose between manual input or batch prediction using a CSV file.")

        option = st.selectbox(
            "Select prediction mode:", ["Manual Input", "Batch Prediction via CSV"]
        )

        if option == "Manual Input":
            self.manual_prediction()
        else:
            self.batch_prediction()


# Instantiate the app with your base API URL and run it
api_base_url = "https://9k6yjdcqae.us-east-1.awsapprunner.com/"
app = App(api_base_url)
app.run()
