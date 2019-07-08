import numpy as np
from azureml.core import Workspace
from sklearn.model_selection import train_test_split

# project name
project_name = "oneraWing"

# cloud or local run
create_ws = True # ignored if cloudRun = False

# project data folder (local and cloud if valid)
project_data_folder = "./data_"+project_name

# get data from project_data_folder (last column is the one to be modelled)
data = np.loadtxt(open(project_data_folder+"/data.csv", "rb"), delimiter=",", skiprows=0)

# split & write processed data to project_data_folder
xTrain, xTest, yTrain, yTest = train_test_split(data[:,:-1], data[:,[-1]], test_size = 1/3, random_state = 0)
np.savetxt(project_data_folder+"/dataTrain.csv", np.column_stack((xTrain,yTrain)), delimiter=",")
np.savetxt(project_data_folder+"/dataTest.csv", np.column_stack((xTest,yTest)), delimiter=",")

# create or open workspace in case of cloud run
if create_ws == True:
    print("creating new workspace")
    ws = Workspace.create(name='give-a-name-for-ur-workspace',
                      subscription_id='give-ur-subscription-id',
                      resource_group='give-a-name-for-resourcegroup',
                      create_resource_group=True,
                      location='westeurope' 
                     )
    ws.get_details()
    ws.write_config()
elif create_ws == False:
    # load workspace configuration from the config.json file in the current folder.
    print("opening existing workspace")
    ws = Workspace.from_config()

# upload processed data to default cloud storage folder
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')
ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)
ds.upload(src_dir=project_data_folder, target_path='data_'+project_name, overwrite=True, show_progress=True)