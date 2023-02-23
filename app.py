import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler


# Create flask app
app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction==0:
        final = "You are healthy"
    elif prediction==1:
        final = "You are unhealthy"
    return render_template('index.html', prediction_text = final)

app.run(debug=True)