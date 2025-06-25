import fastf1
import fastf1.plotting
import time
import random
import numpy as np  
from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fast f1 is largely implem.ented ontop of pandas dataframes
#fastf1.Cache.enable_cache('./cache')

CURRENT_DRIVERS = {
        'VER', 'TSU', 'PER', 'LEC', 'SAI', 'GAS', 'RUS', 'PIA', 'NOR', 'ANT', 'HAM', 'ALB', 'OCO', 'BEA', 'HAD', 'LAW', 'STR', 'ALO','BOR', 'COL'}

def load_training_data():

    # Get the data for a session (Monza 2019 Quali here):
    years = list(range(2015,2024))
    sessions = []

    for year in years:
        try:
            quali = fastf1.get_session(year, 'Monza', 'Q')
            quali.load()
            if quali.laps.empty:
                raise ValueError("Lap data missing")
            sessions.append(quali)
            print(f"Loaded Monza qualifying results from the year {year}")
        except Exception as e:
            print(f"Failed to load {year} quali data: {e}")

    print("Finished loading training data successfully!")
    
    return sessions

def preprocess_training_data(sessions):
    data = []

    for session in sessions:
        year = session.event.year

        if session.weather_data.empty or session.laps.empty:
            continue

        try:
            temp_air = session.weather_data['AirTemp'].mean()
            temp_track = session.weather_data['TrackTemp'].mean()
        except:
            temp_air, temp_track = np.nan, np.nan
    
        laps = session.laps.pick_quicklaps()

        for driver in CURRENT_DRIVERS:
            drv_laps = laps[laps['Driver'] == driver]
            if drv_laps.empty:
                continue

            # Use *all* quick laps for the driver
            for _, lap in drv_laps.iterrows():
                if pd.isna(lap['LapTime']):
                    continue  # Skip invalid laps

                data.append({
                    'year': year,
                    'driver': driver,
                    'team': lap['Team'],
                    'fastest_lap_time': lap['LapTime'].total_seconds(),
                    'compound': lap['Compound'],
                    'air_temp': temp_air,
                    'track_temp': temp_track
                })

    df = pd.DataFrame(data)
    print("Finished pre-processing data successfully!")
    print("Preview of processed data:")
    print(df.head())

    return df        

def initialise_and_train_model(df):

    # Select features and target
    features = ['driver', 'team', 'compound', 'air_temp', 'track_temp']
    target = 'fastest_lap_time'

    X = df[features]
    y = df[target]

    # One hot encode the categorical features (driver, team and compound)
    cat_features = ['driver', 'team', 'compound']
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_cat = encoder.fit_transform(X[cat_features]).toarray()

    # numeric features (air and track temps)
    X_num = X[['air_temp', 'track_temp']].to_numpy()

    # Combine encoded catergories and numeric features
    X_processed = np.hstack([X_cat, X_num])
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size = 0.2, random_state=42)

    # Initialise and train model.
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate prediction
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Model trained! Test MSE: {mae} seconds, RMSE: {rmse} seconds")

    return model, encoder

def predict_qualifying_order(model, encoder, air_temp, track_temp):
    # Define assumed team and compound per driver (can refine this later)
    assumed_compound = 'SOFT'
    assumed_teams = {
        'VER': 'Red Bull Racing',
        'TSU': 'Red Bull Racing',
        'ANT': 'Mercedes',
        'RUS': 'Mercedes',
        'LEC': 'Ferrari',
        'HAM': 'Ferrari',
        'NOR': 'McLaren',
        'PIA': 'McLaren',
        'ALO': 'Aston Martin',
        'STR': 'Aston Martin',
        'GAS': 'Alpine',
        'COL': 'Alpine',
        'ALB': 'Williams',
        'SAI': 'Williams',
        'OCO': 'Haas',
        'BEA': 'Haas',
        'BOT': 'Stake',
        'ZHO': 'Stake',
        'LAW': 'RB',
        'HAD': 'RB'
    }

    data = []
    for driver in CURRENT_DRIVERS:
        team = assumed_teams.get(driver, 'Unknown')
        compound = assumed_compound

        row = pd.DataFrame([{
            'driver': driver,
            'team': team,
            'compound': compound,
            'air_temp': air_temp,
            'track_temp': track_temp
        }])

        # Encode
        cat_features = ['driver', 'team', 'compound']
        X_cat = encoder.transform(row[cat_features]).toarray()
        X_num = row[['air_temp', 'track_temp']].to_numpy()
        X_processed = np.hstack([X_cat, X_num])

        # Predict
        predicted_lap_time = model.predict(X_processed)[0]
        predicted_lap_time += np.random.normal(0, 0.005)
        data.append((driver, predicted_lap_time))

    # Sort by predicted lap time
    sorted_results = sorted(data, key=lambda x: x[1])
    print("\nPredicted Qualifying Order:")
    for i, (driver, time) in enumerate(sorted_results, 1):
        print(f"{i}. {driver} - {time:.3f} seconds")

    return sorted_results


if __name__ == "__main__":
   
    sessions = load_training_data()

    df = preprocess_training_data(sessions)

    model, encoder = initialise_and_train_model(df)

    # Monza 2024 example conditions
    monza_air_temp = 28.0  # in °C
    monza_track_temp = 42.0  # in °C

    predict_qualifying_order(model, encoder, monza_air_temp, monza_track_temp)
