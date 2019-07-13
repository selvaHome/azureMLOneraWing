import numpy as np
import requests
from sklearn.metrics import r2_score
import json
import os, pickle

# project name
project_name = 'oneraWing'

# service url
scoring_uri = "give-the-scoring-url-here"

# get test data from project_data_folder
dataTest= np.loadtxt(open('./data_'+project_name+'/dataTest.csv',"rb"), delimiter=",")

# handle if the test dataset has only one sample 
if len(dataTest.shape) > 1: 
    x_actual = dataTest[:,:-1]
    y_actual = dataTest[:,[-1]]
else:
    x_actual = dataTest[:-1].reshape(-1, len(dataTest[:-1]))
    y_actual = dataTest[[-1]]

# make HTTP requests 
y_hat = np.zeros(shape=(x_actual.shape[0]))
for i in range(x_actual.shape[0]):    
    input_data = '{"data":['+str(list(x_actual[i,:]))+']}'
    headers = {'Content-Type':'application/json'}
    resp = requests.post(scoring_uri, input_data, headers=headers)
    y_hat[i] = float(resp.text.lstrip('[').rstrip(']'))

# score model (r2 error)
score = r2_score(y_actual, y_hat)
score_json = {"score": score}
print("score_json", score_json)