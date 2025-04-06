import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os


os.makedirs("data", exist_ok=True)


match_data = pd.read_csv("data/match_data.csv", low_memory=False)
match_info = pd.read_csv("data/match_info_data.csv")

# FIRST INNINGS
first_innings = match_data[match_data['innings'] == 1].copy()
first_innings['total_runs'] = first_innings['runs_off_bat'] + first_innings['extras']
first_innings['over'] = first_innings['ball'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)


overwise = (
    first_innings.groupby(['match_id', 'over', 'batting_team', 'bowling_team'])
    .agg(
        runs_till_now=('total_runs', 'sum'),
        balls=('ball', 'count'),
        wickets=('player_dismissed', lambda x: x.notna().sum())
    )
    .reset_index()
)

overwise['cumulative_runs'] = overwise.groupby('match_id')['runs_till_now'].cumsum()
overwise['cumulative_wickets'] = overwise.groupby('match_id')['wickets'].cumsum()
overwise['overs_completed'] = overwise['over'] + 1

final_scores = overwise.groupby('match_id')['cumulative_runs'].max().reset_index()
final_scores.columns = ['match_id', 'final_score']

first_model_data = overwise.merge(final_scores, on='match_id')


teams = pd.concat([first_model_data['batting_team'], first_model_data['bowling_team']]).unique()
team_encoding = {team: i for i, team in enumerate(teams)}
first_model_data['batting_team_enc'] = first_model_data['batting_team'].map(team_encoding)
first_model_data['bowling_team_enc'] = first_model_data['bowling_team'].map(team_encoding)

X1 = first_model_data[['batting_team_enc', 'bowling_team_enc', 'cumulative_runs', 'cumulative_wickets', 'overs_completed']]
y1 = first_model_data['final_score']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X1_train, y1_train)

print("Final Score Prediction MAE:", mean_absolute_error(y1_test, reg_model.predict(X1_test)))


joblib.dump(reg_model, "data/final_score_predictor.pkl")

# SECOND INNINGS
second_innings = match_data[match_data['innings'] == 2].copy()
second_innings['total_runs'] = second_innings['runs_off_bat'] + second_innings['extras']
second_innings['over'] = second_innings['ball'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)


match_targets = (
    match_data[match_data['innings'] == 1]
    .groupby('match_id')['runs_off_bat'].sum().reset_index()
)
match_targets.columns = ['match_id', 'target']

second_overwise = (
    second_innings.groupby(['match_id', 'over', 'batting_team', 'bowling_team'])
    .agg(
        runs_till_now=('total_runs', 'sum'),
        balls=('ball', 'count'),
        wickets=('player_dismissed', lambda x: x.notna().sum())
    )
    .reset_index()
)

second_overwise['cumulative_runs'] = second_overwise.groupby('match_id')['runs_till_now'].cumsum()
second_overwise['cumulative_wickets'] = second_overwise.groupby('match_id')['wickets'].cumsum()
second_overwise['overs_completed'] = second_overwise['over'] + 1

second_model_data = second_overwise.merge(match_targets, on='match_id')
second_model_data = second_model_data.merge(match_info[['id', 'winner']], left_on='match_id', right_on='id', how='left')

second_model_data['batting_team_enc'] = second_model_data['batting_team'].map(team_encoding)
second_model_data['bowling_team_enc'] = second_model_data['bowling_team'].map(team_encoding)
second_model_data['run_diff'] = second_model_data['target'] - second_model_data['cumulative_runs']
second_model_data['won'] = (second_model_data['batting_team'] == second_model_data['winner']).astype(int)

X2 = second_model_data[['batting_team_enc', 'bowling_team_enc', 'cumulative_runs', 'cumulative_wickets', 'overs_completed', 'target', 'run_diff']]
y2 = second_model_data['won']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X2_train, y2_train)

print(" Win Prediction Accuracy:", accuracy_score(y2_test, clf_model.predict(X2_test)))
joblib.dump(clf_model, "data/win_predictor.pkl")
