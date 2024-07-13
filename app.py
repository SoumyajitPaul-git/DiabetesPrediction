from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from JSON data
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = np.array([[data[feat] for feat in features]])

    # Scale the input data using the fitted scaler
    input_scaled = scaler.transform(input_data)

    # Predict using the loaded model
    result = model.predict(input_scaled)[0]

    # Convert the result to a native Python int before returning
    result = int(result)

    return jsonify({'Diabetic': result})

if __name__ == "__main__":
    app.run(debug=True)
