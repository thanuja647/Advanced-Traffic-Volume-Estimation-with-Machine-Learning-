from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

# Load trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load encoders
with open("holiday_encoder.pkl", "rb") as file:
    holiday_encoder = pickle.load(file)

with open("weather_encoder.pkl", "rb") as file:
    weather_encoder = pickle.load(file)

# Define feature columns
feature_columns = ["temp", "rain", "snow", "year", "month", "day", "hour", "holiday", "weather"]

# Function to preprocess user input
def preprocess_input(date_input, time_input, temp, rain, snow, holiday, weather):
    date_obj = datetime.datetime.strptime(date_input, "%Y-%m-%d")
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    
    time_obj = datetime.datetime.strptime(time_input, "%H:%M")
    hour = time_obj.hour

    holiday_encoded = holiday_encoder.transform([holiday])[0]
    weather_encoded = weather_encoder.transform([weather])[0]

    input_features = pd.DataFrame([[temp, rain, snow, year, month, day, hour, holiday_encoded, weather_encoded]],
                                  columns=feature_columns)
    
    input_features_scaled = scaler.transform(input_features)
    return input_features_scaled

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        date_input = request.form["date"]
        time_input = request.form["time"]
        temp = float(request.form["temp"])
        rain = float(request.form["rain"])
        snow = float(request.form["snow"])
        holiday = request.form["holiday"]
        weather = request.form["weather"]

        input_data = preprocess_input(date_input, time_input, temp, rain, snow, holiday, weather)
        prediction = model.predict(input_data)[0]

        if prediction > 5000:
            return render_template("chance.html", prediction=int(prediction))
        else:
            return render_template("noChance.html", prediction=int(prediction))

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
