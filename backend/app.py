from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
# Attrition prediction model
with open('employee_attrition_model.pkl', 'rb') as file:
    attrition_model = pickle.load(file)

# Retention policy model
retention_model = pickle.load(open("retention_policy_model.pkl", "rb"))
policy_encoder = pickle.load(open("policy_label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------- Helper Function for Attrition ----------------------
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])

    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    df = pd.get_dummies(df,
                        columns=[
                            'BusinessTravel', 'JobRole', 'MaritalStatus',
                            'EducationField', 'Department'
                        ],
                        drop_first=True)

    df = df.reindex(columns=attrition_model.feature_names_in_, fill_value=0)
    return df

# ---------------------- Routes ----------------------

@app.route('/')
def home():
    return "✅ Backend running: Employee Attrition & Retention Recommendation"

@app.route('/check', methods=['POST'])
def check():
    input_data = request.json
    print(input_data)
    return "<h1>Hello Deepak</h1>"

# Attrition Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    processed_input = preprocess_input(input_data)
    prediction = attrition_model.predict(processed_input)
    return jsonify({'prediction': int(prediction[0])})

# Retention Policy Prediction Route
@app.route('/predict-retention', methods=['POST'])
def predict_retention():
    try:
        data = request.json
        features = np.array([
            data["Age"], data["JobSatisfaction"], data["WorkLifeBalance"],
            data["EnvironmentSatisfaction"], data["YearsAtCompany"],
            data["YearsInCurrentRole"], data["YearsSinceLastPromotion"],
            data["TotalWorkingYears"], data["MonthlyIncome"],
            1 if data["OverTime"] == "Yes" else 0,
            {"Travel_Rarely": 0, "Travel_Frequently": 1, "Non-Travel": 2}[data["BusinessTravel"]]
        ]).reshape(1, -1)

        features[:, :7] = scaler.transform(features[:, :7])
        policy_prediction = retention_model.predict(features)[0]
        recommended_policy = policy_encoder.inverse_transform([policy_prediction])[0]

        return jsonify({
            "RecommendedRetentionPolicy": recommended_policy
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------- Run App ----------------------
if __name__ == '__main__':
    app.run(debug=True)
