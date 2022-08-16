from unittest import result
from flask import Flask, render_template, redirect, url_for, request

import pandas as pd
import numpy as np
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

df = pd.read_csv('bitcoin_data.csv', index_col=False)
dfs = df[:-7]

to_row = int(len(dfs)*0.8)
def train():
    training_data = list(df[:to_row]['Adj Close'])
    return training_data

def train2():
    training_data = list(dfs[:]['Adj Close'])
    return training_data

def test():
    test_data = list(df[-7:]['Adj Close'])
    return test_data

@app.route("/")
def main():
    return render_template('index.html', menu='home', data=dfs, len=len, list=list, round=round)

@app.route("/calculation")
def calculation():
    test = []
    j=0
    while j < len(dfs[to_row:]):
        test.append(dfs['Adj Close'][to_row+j])
        j+=1
    
    testing_data = list(test)
    training_data = train()
    
    model_prediction = []
    n_test_obser = len(testing_data)
    
    date_range = list(dfs[to_row:]['Date'])

    for i in range(n_test_obser):
        model = ARIMA(training_data, order = (4,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        #first prediction
        yhat = list(output[0])[0]
        model_prediction.append(yhat)
        actual_test_value = testing_data[i]
        training_data.append(actual_test_value)
    
    model = model_fit.summary()
    mape = np.mean(np.abs(np.array(model_prediction) - np.array(testing_data)) / np.abs(testing_data))*100

    return render_template('calculate.html', menu='calculation', data=dfs, training=training_data, testing=testing_data, train2=train(), model=model, mape=mape, prediction=model_prediction, date=date_range, len=len, list=list, round=round)

@app.route("/predictions")
def predictions():
    model_prediction = []
    n_test_obser = 7
    date = list(df[-7:]['Date'])
    training_data = train()
    test_data = test()

    for i in range(n_test_obser):
        model = ARIMA(training_data, order = (3,1,3))
        model_fit = model.fit()
        output = model_fit.forecast()
        #first prediction
        yhat = list(output[0])[0]
        model_prediction.append(yhat)
        actual_test_value = test_data[i]
        training_data.append(actual_test_value)
    
    test_data = list(df[-7:]['Adj Close'])
    mape = np.mean(np.abs(np.array(model_prediction) - np.array(test_data)) / np.abs(test_data))*100
    return render_template('predics.html', menu='predictions', data=dfs, predic=model_prediction, list=list, len=len, date=date, round=round, mape=mape)

if __name__ == "__main__":
    app.run(debug=True)