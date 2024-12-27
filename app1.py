from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/random_forest_model.pkl')

# HTML Template as a string
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f9;
        }
        h1 {
            color: #333;
        }
        form {
            max-width: 500px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
        }
        input {
            width: 100%;
            padding: 8px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 4px;
        }
        button:hover {
            background-color: #218838;
        }
        .result {
            max-width: 500px;
            margin: 20px auto;
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>Diabetes Prediction</h1>
    <form action="/predict" method="post">
        <label>Pregnancies:</label>
        <input type="number" name="Pregnancies" required>
        <label>Glucose:</label>
        <input type="number" name="Glucose" required>
        <label>Blood Pressure:</label>
        <input type="number" name="BloodPressure" required>
        <label>Skin Thickness:</label>
        <input type="number" name="SkinThickness" required>
        <label>Insulin:</label>
        <input type="number" name="Insulin" required>
        <label>BMI:</label>
        <input type="number" name="BMI" step="0.1" required>
        <label>Diabetes Pedigree Function:</label>
        <input type="number" name="DiabetesPedigreeFunction" step="0.01" required>
        <label>Age:</label>
        <input type="number" name="Age" required>
        <button type="submit">Predict</button>
    </form>

    {% if prediction_text %}
    <div class="result">
        <h2>{{ prediction_text }}</h2>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.form
        input_features = np.array([[float(data[field]) for field in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]])
        
        # Make a prediction
        prediction = model.predict(input_features)
        output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
        
        return render_template_string(html_template, prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template_string(html_template, prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
