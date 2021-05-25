import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  #loading pickle file

@app.route('/')
def home():
    return render_template('index.html')    #home page

@app.route('/predict',methods=['POST'])   #post method, here we will provide some features to model.pkl
def predict():
    int_features = [int(x) for x in request.form.values()]  #request input from all 3 text fields
    final_features = [np.array(int_features)]    #converting values of 3 text field to array
    prediction = model.predict(final_features)   #giving the array for prediction

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)