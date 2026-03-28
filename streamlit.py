import pandas as pd
import pickle
import streamlit as st

# Set the page title and description
st.title("Mall Customer Clustering App")
st.write("""
This app predicts which cluster a mall customer belongs to based on customer characteristics.""")

try:
    # Load the pre-trained model
    with open("models/Kmodel.pkl", "rb") as k_file:
        k_model = pickle.load(k_file)

    # Prepare the form to collect user inputs
    with st.form("user_inputs"):
        st.subheader("Customer Details")

        # Gender input
        Gender = st.selectbox("Gender", options=["Male", "Female"])

        # Age
        Age = st.number_input("Age", min_value=1, step=1, value=19)

        # Annual Income
        Annual_Income = st.number_input("Annual_Income", min_value=0, step=1, value=15)

        # Spending Score
        Spending_Score = st.number_input("Spending_Score", min_value=0, max_value=100, step=1, value=39)

        # Submit button
        submitted = st.form_submit_button("Predict Customer Cluster")

    # Handle prediction
    if submitted:
        try:
            prediction_input = pd.DataFrame([[
                Annual_Income,
                Spending_Score
            ]], columns=[
                "Annual_Income",
                "Spending_Score"
            ])

            # Make prediction
            new_prediction = k_model.predict(prediction_input)

            # Display result
            st.subheader("Prediction Result:")
            st.write(f"This customer belongs to Cluster {new_prediction[0]}")

        except Exception as e:
            st.error(f"Error while making prediction: {e}")

    st.write(
        """We used a machine learning (KMeans Clustering) model to group customers into clusters based on their characteristics."""
    )

    st.image("cluster_plot.png")

except Exception as e:
    st.error(f"Error while loading the application: {e}")