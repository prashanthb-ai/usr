import argparse
import config
import sagemaker
import json
from sagemaker.model import Model
from sagemaker.predictor import RealTimePredictor

parser = argparse.ArgumentParser()
parser.add_argument("--deploy", type=bool, default=False)
args = parser.parse_args()

role = config.configmap["role"]
image = config.configmap["image"]
model_data = config.configmap["model"]
sm_model_name = config.configmap["model_name"]
endpoint_name = config.configmap["endpoint_name"]
file_name = "0.png"

if args.deploy:
    torchserve_model = Model(
            model_data=model_data,
            image_uri=image,
            role=role,
            predictor_cls=RealTimePredictor,
            name=sm_model_name)

    torchserve_model.deploy(
        instance_type='ml.m4.xlarge',
        initial_instance_count=1,
        endpoint_name=endpoint_name)

with open(file_name, 'rb') as f:
 payload = f.read()
 payload = payload

predictor = RealTimePredictor(endpoint_name)
response = predictor.predict(data=payload)
print("Model prediction: {}".format(json.loads(response)))

