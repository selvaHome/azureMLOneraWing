from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core import Workspace
from azureml.core.model import Model


# project name
project_name = 'oneraWing'

# create environment file
myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")
with open(project_name+".yml","w") as f:
    f.write(myenv.serialize_to_string())

# create configuration
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "cl/cd",  "method" : "sklearn_linear"}, 
                                               description='Predict cl/cd with sklearn')

# configure the image
ws = Workspace.from_config()
model=Model(ws, project_name+'_model')
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file=project_name+".yml")

# deploy model
service = Webservice.deploy_from_model(workspace=ws,
                                       name='onera-wing-model',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)
print(service.scoring_uri)



