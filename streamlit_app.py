
import streamlit as st
import pickle
import numpy as np

# Load the model

model_path = "classifier.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)


# Predict the outcome
def predict_outcome(model, features):
    features = np.array([features])
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app
def main():
    st.title("NBA Player Success Prediction")

    # Input fields for player statistics
    games_played = st.number_input("Games Played", value=40, min_value=0, step=1)
    minutes = st.number_input("Minutes Played", value=25, min_value=0, step=1)
    field_goals_made = st.number_input("Field Goals Made", value=5.0, min_value=0.0, step=0.1)
    three_point_attempts = st.number_input("Three-Point Attempts", value=3.0, min_value=0.0, step=0.1)
    three_point_percentage = st.number_input("Three-Point Percentage (%)", value=35.0, min_value=0.0, max_value=100.0, step=0.1)
    free_throws_made = st.number_input("Free Throws Made", value=3.0, min_value=0.0, step=0.1)
    free_throw_percentage = st.number_input("Free Throw Percentage (%)", value=75.0, min_value=0.0, max_value=100.0, step=0.1)
    offensive_rebounds = st.number_input("Offensive Rebounds", value=1.0, min_value=0.0, step=0.1)
    defensive_rebounds = st.number_input("Defensive Rebounds", value=3.0, min_value=0.0, step=0.1)
    total_rebounds = st.number_input("Total Rebounds", value=4.0, min_value=0.0, step=0.1)
    blocks = st.number_input("Blocks", value=0.5, min_value=0.0, step=0.1)
    turnovers = st.number_input("Turnovers", value=2.0, min_value=0.0, step=0.1)


    # Button to make the prediction
    if st.button("Predict Player Success"):
        # Collect the input values
        player_features = [
            games_played,
            minutes,
            field_goals_made,
            three_point_attempts,
            three_point_percentage,
            free_throws_made,
            free_throw_percentage,
            offensive_rebounds,
            defensive_rebounds,
            total_rebounds,
            blocks,
            turnovers
        ]

        # Make the prediction
        prediction = predict_outcome(model, player_features)

        # Display the result
        if prediction:
            st.success("The player is predicted to be successful.")
        else:
            st.error("The player is predicted to be unsuccessful.")

if __name__ == '__main__':
    main()
