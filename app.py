import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64


first_innings_model = joblib.load("final_score_predictor.pkl")
second_innings_model = joblib.load("win_predictor.pkl")


team_encoding = {
    'Kolkata Knight Riders': 0,
    'Royal Challengers Bangalore': 1,
    'Chennai Super Kings': 2,
    'Mumbai Indians': 3,
    'Delhi Capitals': 4,
    'Sunrisers Hyderabad': 5,
    'Rajasthan Royals': 6,
    'Punjab Kings': 7,
    'Gujarat Titans': 8,
    'Lucknow Super Giants': 9
}

def get_base64_gif(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode()

st.set_page_config(page_title="IPL Predictor", layout="centered")

st.title("🏏 IPL Match Predictor")

st.markdown("### Enter Match Details")

innings_type = st.selectbox("Select Innings", ["First", "Second"])

batting_team = st.selectbox("Select Batting Team", list(team_encoding.keys()))
bowling_team = st.selectbox("Select Bowling Team", list(team_encoding.keys()))

current_score = st.number_input("Current Score", min_value=0, max_value=300, step=1)


wickets = st.selectbox("Wickets Fallen", list(range(0, 11)))
overs_completed = st.selectbox("Overs Completed", list(range(1, 21)))


target = None
if innings_type == "Second":
    target = st.number_input("Target Score", min_value=1, max_value=300, step=1)

if st.button("Predict"):
    if not batting_team or not bowling_team or current_score == 0 or (innings_type == "Second" and not target):
        st.warning("⚠️ Please enter all required match details before predicting.")
    else:
        try:
            bat_enc = team_encoding[batting_team]
            bowl_enc = team_encoding[bowling_team]

            if innings_type == "First":
                input_features = pd.DataFrame([{
                    'batting_team_enc': bat_enc,
                    'bowling_team_enc': bowl_enc,
                    'cumulative_runs': current_score,
                    'cumulative_wickets': wickets,
                    'overs_completed': overs_completed
                }])
                predicted_score = first_innings_model.predict(input_features)[0]
                st.success(f"🏏 Predicted Final Score: {int(predicted_score)}")

            else:
                run_diff = target - current_score
                input_features = pd.DataFrame([{
                    'batting_team_enc': bat_enc,
                    'bowling_team_enc': bowl_enc,
                    'cumulative_runs': current_score,
                    'cumulative_wickets': wickets,
                    'overs_completed': overs_completed,
                    'target': target,
                    'run_diff': run_diff
                }])

                win_prob = second_innings_model.predict_proba(input_features)[0]
                batting_prob = round(win_prob[1] * 100, 2)
                bowling_prob = round(win_prob[0] * 100, 2)

                st.success("🔮 Win Prediction:")
                win_gif = get_base64_gif("win gif.gif")
                lost_gif = get_base64_gif("lost gif.gif")

                if batting_prob > bowling_prob:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{batting_team}: {batting_prob}% chance to win**")
                    with col2:
                        st.markdown(f"<img src='data:image/gif;base64,{win_gif}' width='60'>", unsafe_allow_html=True)

                    col3, col4 = st.columns([4, 1])
                    with col3:
                        st.markdown(f"**{bowling_team}: {bowling_prob}% chance to win**")
                    with col4:
                        st.markdown(f"<img src='data:image/gif;base64,{lost_gif}' width='60'>", unsafe_allow_html=True)
                else:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{batting_team}: {batting_prob}% chance to win**")
                    with col2:
                        st.markdown(f"<img src='data:image/gif;base64,{lost_gif}' width='60'>", unsafe_allow_html=True)

                    col3, col4 = st.columns([4, 1])
                    with col3:
                        st.markdown(f"**{bowling_team}: {bowling_prob}% chance to win**")
                    with col4:
                        st.markdown(f"<img src='data:image/gif;base64,{win_gif}' width='60'>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Turning cricket insights into winning strategies. 🏆📊")
st.markdown("<small><i>Note: While this model is trained on high-quality IPL data, predictions may not always be accurate. Cricket is a game of glorious uncertainties!</i></small>", unsafe_allow_html=True)
