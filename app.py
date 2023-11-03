import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, request,jsonify, render_template




# create flask app
flask_app = Flask(__name__)
model = pickle.load(open("mobile_price_model.pkl", "rb"))
scaler = pickle.load(open('scaler.pkl','rb'))

@flask_app.route("/")
def index():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    
    input_features =[[int(x) for x in request.form.values()]]
    features_values = scaler.transform(input_features)
    prediction = model.predict(features_values)
    
    
    
    return render_template("index.html", prediction=prediction[0])

if __name__== "__main__":
    flask_app.run(debug=True)