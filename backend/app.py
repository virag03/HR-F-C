from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 


with open('employee_attrition_model.pkl', 'rb') as file:
    model = pickle.load(file)


def preprocess_input(data):
    # Convert input data into a DataFrame
    df = pd.DataFrame(data, index=[0])

    
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

     
    df = pd.get_dummies(df,
                        columns=[
                            'BusinessTravel', 'JobRole', 'MaritalStatus',
                            'EducationField', 'Department'
                        ],
                        drop_first=True)

    
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    return df

@app.route('/check', methods=['POST'])
def hello():
    input_data = request.json
    print(input_data)
    return "<h1>hello deepak</h1>"


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    processed_input = preprocess_input(input_data)  # Call
    prediction = model.predict(processed_input)

    return jsonify({'prediction': int(prediction[0])})  # Return prediction as an integer

if __name__ == '__main__':
    app.run(debug=True)
