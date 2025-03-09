import os
import joblib
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "weight_change_dataset.csv"
df = pd.read_csv(file_path)

# Standardizing column names
df.columns = df.columns.str.strip()

# Checking dataset columns
expected_columns = ['Age', 'Current Weight (lbs)', 'Gender', 'Physical Activity Level', 'Daily Calories Consumed', 'Duration (weeks)', 'Sleep Quality', 'Stress Level']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}")

# Encode categorical variables
le_gender = LabelEncoder()
le_activity = LabelEncoder()

df['Gender'] = df['Gender'].str.lower().map({'m': 'male', 'f': 'female'})
le_gender.fit(['male', 'female'])  # Ensure all possible values are included

df['Gender'] = le_gender.transform(df['Gender'])
df['Physical Activity Level'] = le_activity.fit_transform(df['Physical Activity Level'].str.lower())

# Map sleep quality
sleep_quality_map = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
df['Sleep Quality'] = df['Sleep Quality'].map(sleep_quality_map)

# Ensure stress level is numeric
df['Stress Level'] = df['Stress Level'].astype(int)

# Define features and target
X = df[['Age', 'Current Weight (lbs)', 'Gender', 'Physical Activity Level', 'Duration (weeks)', 'Sleep Quality', 'Stress Level']]
y = df['Daily Calories Consumed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump((model, le_gender, le_activity), "calorie_predictor.pkl")

# Flask App
app = Flask(__name__)

# Synthwave HTML Page
html_template = """
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Calorie Predictor</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #00fff7;
            font-family: 'Press Start 2P', cursive;
            text-align: center;
            padding: 20px;
        }
        h1 {
            color: #ff00ff;
            text-shadow: 0 0 10px #ff00ff, 0 0 20px #ff1493;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px #ff00ff;
            display: inline-block;
            margin-top: 20px;
        }
        input, select, button {
            width: 80%;
            padding: 10px;
            margin: 10px;
            border: none;
            border-radius: 5px;
            background: black;
            color: #00fff7;
            border: 1px solid #ff00ff;
            font-size: 14px;
        }
        button {
            background: #ff00ff;
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            background: #ff1493;
            transform: scale(1.1);
        }
    </style>
</head>
<body>
    <h1>💜 Synthwave Calorie Predictor 💙</h1>
    <div class='container'>
        <form action='/' method='post'>
            <input type='number' name='age' placeholder='Age' required><br>
            <input type='number' step='0.1' name='weight' placeholder='Weight (lbs)' required><br>
            <select name='gender' required>
                <option value='male'>Male</option>
                <option value='female'>Female</option>
            </select><br>
            <select name='activity_level' required>
                <option value='sedentary'>Sedentary</option>
                <option value='lightly active'>Lightly Active</option>
                <option value='moderately active'>Moderately Active</option>
                <option value='very active'>Very Active</option>
            </select><br>
            <input type='number' name='duration' placeholder='Duration (weeks)' required><br>
            <select name='sleep_quality' required>
                <option value='poor'>Poor</option>
                <option value='fair'>Fair</option>
                <option value='good'>Good</option>
                <option value='excellent'>Excellent</option>
            </select><br>
            <input type='number' name='stress_level' placeholder='Stress Level (1-9)' required><br>
            <button type='submit'>Predict</button>
        </form>
        {% if prediction %}
            <h2>You need {{ prediction }} calories/day</h2>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        model, le_gender, le_activity = joblib.load("calorie_predictor.pkl")

        # Get form inputs
        age = int(request.form["age"])
        weight = float(request.form["weight"])
        gender = request.form["gender"].lower()
        activity_level = request.form["activity_level"].lower()
        duration = int(request.form["duration"])
        sleep_quality = request.form["sleep_quality"].lower()
        stress_level = int(request.form["stress_level"])

        # Ensure gender encoding is trained properly
        if gender not in le_gender.classes_:
            return "Error: Invalid gender input. Please use 'male' or 'female'."

        gender_encoded = le_gender.transform([gender])[0]
        activity_encoded = le_activity.transform([activity_level])[0]
        sleep_encoded = {"poor": 1, "fair": 2, "good": 3, "excellent": 4}[sleep_quality]

        # Predict
        input_data = pd.DataFrame([[age, weight, gender_encoded, activity_encoded, duration, sleep_encoded, stress_level]],
                                  columns=X.columns)
        predicted_calories = model.predict(input_data)[0]
        prediction = round(predicted_calories)

    return render_template_string(html_template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)



