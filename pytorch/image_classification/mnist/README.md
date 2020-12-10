# Quickstart

Saving/training the model:
```
$ python3 main.py --save-model
$ mv mnist_cnn.pt ~/rtmp/ml/models/model.pt
```
Create a `mar` with the model and the handler
```
$ torch-model-archiver --model-name mnist --version 1.0 --model-file model.py --serialized-file ~/rtmp/ml/models/model.pt --handler handler.py
```
Run the server
```
$ torchserve --start --model-store model_store --models mnist=mnist.mar
$ curl http://127.0.0.1:8080/models
$ curl http://127.0.0.1:8080/predictions/mnist -T 0.png
0
```

## Deployment on SM

Deploy the model on AWS fully managed instances.

### Prerequisites

* [AWS Cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-prereq)
* AWS SDKs - [for python (boto)](https://aws.amazon.com/sdk-for-python/), [for sagemaker](https://github.com/aws/sagemaker-python-sdk#installing-the-sagemaker-python-sdk)


Sign into your AWS account and get your access key from the [IAM page](https://console.aws.amazon.com/iam/home) (search for your own name under `Users`). The next command will create a file like
```
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
region=us-east-1
```

Under `~/.aws/credentials`, so make sure you retrieve this information before running
```
$ aws configure
```
After this you should be able to write a basic boto script
```
import boto3
for bucket in boto3.resource('s3').buckets.all():
  print(bucket.name)
```

### Upload the mode

tar and upload to an S3 buckt
```
$ tar cvfz model.tar.gz mnist.mar
$ aws s3 cp model.tar.gz s3://<bucket>/torchserve/model
```


## TODO

* Figure out `model.pt` bug. Torchserve logs have
```
[handler] about to torch load, model_def_path /tmp/models/26d0de933c30470f81e2e8e73372b09c/model.py model_pt_path /tmp/models/26d0de933c30470f81e2e8e73372b09c/model.pt
```
But that path only contains `mnist_cnn.pt` if the file produced by `main.py` isn't renamed.
    - This is because the name `model.py` is hardcoded in the initializing
      routine of the `handler.py` file.

## Appendix

* [Source](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)
