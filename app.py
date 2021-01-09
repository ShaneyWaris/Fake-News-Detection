from flask import Flask, render_template, request, url_for, redirect
import os
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import joblib

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
mlp = joblib.load('saved_model.pkl')
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    b = False
    if request.method == 'POST':
        body = request.form.get('body')
        if len(body)!= 0:
            pred_value = mlp.predict(embed([body]))
            if pred_value == 0.0:
                l = 'Fake'
            if pred_value == 1.0:
                l = 'Real'
            b = True
        if b == False:
            return render_template('index.html', b=b)
        else:
            return render_template('index.html', b=b, l=l)

    return render_template('index.html', b=b)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/members')
def members():
    return render_template('members.html')

@app.route('/readme')
def readme():
    return render_template('ReadMe.html')

port = int(os.getenv('PORT', 8000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
