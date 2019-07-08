import os
import numpy as np

# project name
project_name = "oneraWing"

# source data folder
src_data_folder = "./data"

# target data folder
target_data_folder = "./data_"+project_name

# get data from source folder 
data = np.loadtxt(open(src_data_folder+"/data_to_model.csv", "rb"), delimiter=",", skiprows=1)

# create target folder and write (input => first 2 columns & output => last one column)
os.makedirs(target_data_folder, exist_ok=True)
np.savetxt(target_data_folder+"/data.csv", data[:, [0,1,4]], delimiter=",")