import os
import argparse
import numpy as np
from azureml.core import Run
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor

# project name
project_name = 'oneraWing'

# get cloud data folder as argument
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()
data_folder = args.data_folder

# get training data from cloud storage
dataTrain = np.loadtxt(open(os.path.join(data_folder, 'dataTrain.csv'),"rb"), delimiter=",")

# get hold of the current run
run = Run.get_context()

# train the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(dataTrain[:,:-1], dataTrain[:,-1])

# get a score on the training dataset
score = model.score(dataTrain[:,:-1], dataTrain[:,[-1]])

# write log report
run.log('data_folder :', data_folder)
run.log('model score :', score)

# note file saved in the outputs folder is automatically uploaded into experiment record
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/'+project_name+'_model.pkl')