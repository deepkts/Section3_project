from flask import Flask, render_template, request
import pickle
import numpy as np


def create_app():
    app = Flask(__name__)

    model = pickle.load(open('model.pkl', 'rb'))

    @app.route('/')
    def index():
        return render_template('index.html')


    @app.route('/predict', methods=['POST'])
    def sub():
        data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        data4 = request.form['d']
        data5 = request.form['e']
        arr = np.array([[data1, data2, data3, data4, data5]])
        pred = model.predict(arr)
        return render_template('sub.html', data=pred)

    return app

