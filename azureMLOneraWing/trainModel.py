import os
import pickle
import numpy as np
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from sklearn.externals import joblib
from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.core.compute import AmlCompute
from azureml.train.estimator import Estimator
from azureml.core.compute import ComputeTarget
from sklearn.ensemble import GradientBoostingRegressor

# project name
project_name = 'oneraWing'

# load workspace
print("opening existing workspace")
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

# create an experiment
exp = Experiment(workspace=ws, name=project_name)

# choose a name for your cluster
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# CPU VM is used. To use GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

# either create or use an existing cluster
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found existing compute target, thus using it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                            min_nodes = compute_min_nodes,
                                                            max_nodes = compute_max_nodes)

    # create the cluster of CPUs
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# For a more detailed view of current AmlCompute status, use get_status()
print(compute_target.get_status().serialize())

# get default storage associated with workspace & pass it to be used in 'train.py'
ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)
script_params = {
    '--data-folder': ds.path('data_'+project_name).as_mount()
}

# create an object of SKLearn estimator
est = SKLearn(source_directory="./"+project_name,
            script_params=script_params,
            compute_target=compute_target,
            entry_script='train.py')

# Run the experiment
run = exp.submit(config=est)
run

# Monitor the remote run 
RunDetails(run).show()

# show log results upon completion
run.wait_for_completion(show_output=True) # specify True for a verbose log

# display run results
print(run.get_metrics())

# list files in the 'outputs' dir
print(run.get_file_names())

# register model
model = run.register_model(model_name=project_name+'_model', model_path='outputs/'+project_name+'_model.pkl')
print(model.name, model.id, model.version, sep = '\t')    