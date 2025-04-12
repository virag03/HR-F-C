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


# Load the models and scaler
with open('retention1_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler1.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('kmeans1.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('feature1_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)




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

def map_policy(cluster):
    policies = {
        0: "Improve Work-Life Balance + Reduce Overtime",
        1: "Salary Hike + Recognition Programs",
        2: "Offer Career Development & Upskilling",
        3: "Job Role Rotation + Flexible Scheduling"
    }
    return policies.get(cluster, "Better Management + Mental Health Support")


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


@app.route('/retention', methods=['POST'])
def retention_form():
    policy = None

    # Safely get data
    if request.is_json:
        input_data = request.get_json()
    else:
        input_data = request.form.to_dict()

    if not input_data:
        return jsonify({'error': 'No input data received'}), 400

    # Ensure no nested dicts sneak in
    for key, value in input_data.items():
        if isinstance(value, dict):
            return jsonify({'error': f'Invalid value for "{key}": nested dictionaries are not allowed.'}), 400

    input_df = pd.DataFrame([input_data])

    try:
        input_encoded = pd.get_dummies(input_df)
    except Exception as e:
        return jsonify({'error': f'Failed to encode input: {str(e)}'}), 400

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    input_scaled = scaler.transform(input_encoded)
    cluster = kmeans.predict(input_scaled)[0]
    policy = map_policy(cluster)

    return jsonify({'prediction': policy})





# ---------------------- Run App ----------------------
if __name__ == '__main__':
    app.run(debug=True)



#   employee will stay karun dakhav