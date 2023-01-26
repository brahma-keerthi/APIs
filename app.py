from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('ML Android/Employee Salary Predictor Android App/model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    exp = request.form.get('exp')

    input_query = np.array([[float(exp)]])
    result = model.predict(input_query)[0]

    return jsonify( {'salary' : str(result)} )

if __name__ == '__main__':
    app.run(debug=True)