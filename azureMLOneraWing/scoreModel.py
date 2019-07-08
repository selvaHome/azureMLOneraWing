import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from azureml.core import Workspace
from azureml.core.model import Model
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from mpl_toolkits import mplot3d


# project name
project_name = 'oneraWing'

# local model directory
local_model_dir = os.path.join(os.getcwd(), "outputs")

# get test data from project_data_folder
dataTest = np.loadtxt(open('./data_'+project_name+'/dataTest.csv',"rb"), delimiter=",")

# download the model from cloud
ws = Workspace.from_config()
model=Model(ws, project_name+'_model')
model.download(target_dir=local_model_dir, exist_ok=True)

# verify downloaded model file
file_path = os.path.join(local_model_dir, project_name+"_model.pkl")
os.stat(file_path)

# load cloud model locally
black_box = joblib.load(os.path.join(local_model_dir, project_name+"_model.pkl"))

# score model locally (r2 error)
if len(dataTest.shape) > 1: # to handle if the test dataset has only one sample
    x_actual = dataTest[:,:-1]
    y_actual = dataTest[:,[-1]]
else:
    x_actual = dataTest[:-1].reshape(-1, len(dataTest[:-1]))
    y_actual = dataTest[[-1]]

y_hat = black_box.predict(x_actual)
score = r2_score(y_actual, y_hat)
score_json = {"r2score": score}

# save the score as JSON
with open('./outputs/'+project_name+'_model_score.json', 'w') as fp:
    json.dump(score_json, fp)


# stack both train and test data
dataTrain = np.loadtxt(open('./data_'+project_name+'/dataTrain.csv',"rb"), delimiter=",")
dataTest = np.loadtxt(open('./data_'+project_name+'/dataTest.csv',"rb"), delimiter=",")
data_actual = dataTrain #np.vstack((dataTrain,dataTest))
data_bounds = np.vstack((np.amin(data_actual[:,], axis=0), np.amax(data_actual[:,], axis=0)))

# create necessary mesh data fro 3D plot
mesh_size = 300
angle_attack = np.linspace(data_bounds[0,0], data_bounds[1,0], mesh_size,).reshape(-1,1)
mach = np.linspace(data_bounds[0,1], data_bounds[1,1], mesh_size).reshape(-1,1)
angle_attack_grid, mach_grid = np.meshgrid(angle_attack, mach)
cl_cd_model = np.zeros((mesh_size,mesh_size))
for i in range(mesh_size):
    for j in range(mesh_size):
        cl_cd_model[i,j] = black_box.predict([[angle_attack_grid[i,j], mach_grid[i,j]]])


# 3D plot (x=angle of attach, y=Mach number, z=CL/CD from model)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(angle_attack_grid, mach_grid, cl_cd_model, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Angle of attack')
ax.set_ylabel('Mach number')
ax.set_zlabel('CL / CD')
ax.set_title('CL / CD surface')
plt.savefig('./outputs/'+project_name+'_3D.png')
plt.clf()
plt.cla()
plt.close()

# plot 'angle of attack' vs 'aerodynamic efficiency'
data_actual_sorted = data_actual[data_actual[:,0].argsort()] # based on angle of attack
cl_cd_model = black_box.predict(np.hstack((angle_attack,mach)))
plt.plot(data_actual_sorted[:,[0]],data_actual_sorted[:,[-1]], 'o', label='Actual')
plt.plot(angle_attack, cl_cd_model, label='ML model')
plt.legend(loc='upper right')
plt.xlabel('angle of attack')
plt.ylabel('CL / CD')
plt.savefig('./outputs/'+project_name+'_AOA_vs.png')
plt.clf()
plt.cla()
plt.close()

# plot 'Mach number' vs 'aerodynamic efficiency'
data_actual_sorted = data_actual[data_actual[:,1].argsort()] # based on mach
plt.plot(data_actual_sorted[:,[1]],data_actual_sorted[:,[-1]], 'o', label='Actual')
plt.plot(mach, cl_cd_model, label='ML model')
plt.legend(loc='upper right')
plt.xlabel('Mach number')
plt.ylabel('CL / CD')
plt.savefig('./outputs/'+project_name+'_Mach_vs.png')