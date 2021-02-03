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
$ torchserve --start --model-store ~/rtmp/ml/models/archive/ --models mnist=mnist.mar
```

you should see the following logs
```
$ grep -ir mnist ./logs/

config/20210105172147102-startup.cfg:model_snapshot={\n  "name"\: "20210105172147102-startup.cfg",\n  "modelCount"\: 1,\n  "created"\: 1609847507107,\n  "models"\: {\n    "mnist"\: {\n      "1.0"\: {\n        "defaultVersion"\: true,\n        "marName"\: "mnist.mar",\n        "minWorkers"\: 8,\n        "maxWorkers"\: 8,\n        "batchSize"\: 1,\n        "maxBatchDelay"\: 100,\n        "responseTimeout"\: 120\n      }\n    }\n  }\n}
config/20210105172147102-startup.cfg:load_models=mnist\=mnist.mar
access_log.log:2021-01-05 17:22:48,640 - /127.0.0.1:50396 "PUT /predictions/mnist HTTP/1.1" 200 40
```

and the following test should work
```
$ curl http://127.0.0.1:8081/models
$ curl http://127.0.0.1:8080/predictions/mnist -T 0.png
0
```
You can also run the same thing through docker, as follows
```
$ cd docker
$ docker build -t bprashanth/torchserve-mnist:0.1 .
$ docker run -d -v /home/beeps/rtmp/ml/models/archives/:/opt/ml/model -p 8080:8080 -p 8081:8081 bprashanth/torchserve-mnist:0.1
```

## Deployment on SM

Deploy the model on AWS fully managed instances.

### Prerequisites

* [AWS Cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-prereq)
* AWS SDKs - [for python (boto)](https://aws.amazon.com/sdk-for-python/), [for sagemaker](https://github.com/aws/sagemaker-python-sdk#installing-the-sagemaker-python-sdk)
* AWS [ecr credential helper](https://github.com/awslabs/amazon-ecr-credential-helper)
* A file called `config.env` in this directory, with the following parameters
```
image=<docker image name:version>
role=<arn of sagemaker role>
model=<uri of the s3 bucket the tar.gz of the .mar is uploaded to>
endpoint_name=<name of endpoint to deploy to sagemaker>
model_name=<name of the model, this should match the model uploaded to s3>
```

#### AWS Configure

Sign into your AWS account and get your access key from the [IAM page](https://console.aws.amazon.com/iam/home) (search for your own name under `Users`). The next command will create a file like
```
[default]
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
region=us-east-1
```

Under `~/.aws/credentials`, so make sure you retrieve the access key from the console before running it. Also make sure the User has the required AWS permissions sets (eg: s3).
```
$ aws configure
```

##### IAM roles

You should also craate an IAM `role` with SageMaker full and S3 read access, for later use. The `ARN` of this role is read and used in a later stage to host the model.

Optional: To grant yourself permissions to switch into this role (yourself = the user configured via `aws configure`, i.e the default user in the console), you need to modify its [trust relationships](https://aws.amazon.com/blogs/security/how-to-use-trust-policies-with-iam-roles/). Once you do this you should also be able to run
```
$ aws sts assume-role --role-arn <arn> --role-session-name sagemaker
```
You should now also be able to switch to this role in the console. The `account` field in the console requires your numeric user-id, and the role field requires the name of the role. Neither field needs the arn.

#### Boto (optional)

After this you should be able to write a basic boto script
```
import boto3
for bucket in boto3.resource('s3').buckets.all():
  print(bucket.name)
```
It is not clear whether boto actually makes anything easier, so you can skip this step.

#### Docker/ECR credentials

Install the credential helper
```
$ sudo apt-get update
$ sudo apt-get install amazon-ecr-credential-helper
$ docker-credential-helper-ecr
...
```
Export the following environment variables
```
export AWS_SECRET_ACCESS_KEY=[secure]
export AWS_ACCESS_KEY_ID=[secure]
export AWS_DEFAULT_REGION=[secure]
```
You can get those values from `~/.aws/credentials`
Add the `credsStore` to your `~/.docker/config.json`
```
{
    "credsStore":"ecr-login"
}
```
This should allow you to push/pull from private registries created in the next step.

### Upload the model to S3

tar and upload to an S3 buckt (this is the same mnist.mar from the [quickstart](quickstart) step)
```
$ cd model_store/
$ tar cvfz model.tar.gz mnist.mar
$ aws s3 cp model.tar.gz s3://<bucket>/torchserve/model
```
Create an ECS registry
```
$ aws ecr create-repository --repository-name {repository_name}
  registryId: ''
  repositoryArn: arn:aws:ecr:{aws_region}:{aws_account}:repository/{repository_name}
  repositoryName: {repository_name}
  repositoryUri:
{aws_region}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}
```

### Upload a torchserve image to ECR

The first part of the url is the aws account id, which you can also find via
```
$ aws sts get-caller-identity
```
Login to the registry
```
$ aws_region=ap-south-1
$ aws_account=$(aws sts get-caller-identity)
$ repository_name=<repository-name>
```
Push a docker image with the model up to the repository
```
$ cd ./docker
$ version='0.1'
$ aws_account=<aws account from sts get-caller-identity, also the registryId from ecr create-repository>
$ image=<repositoryUri from create-repository, also {aws_account}.dkr.ecr.{aws_region}.amazonaws.com/{repository-name}>
$ region='ap-south-1'
$ docker build -t ${image}:${version} .
$ docker push {image}:{tag}
# eg: docker push 1231231231.dkr.ecr.ap-south-1.amazonaws.com/foo-tmp/torchserve:0.1
```
The last step requires AWS ECR credentials helper. If you get permissions errors, look at the [prerequisites](prerequisites).

### Hosting the inference endpoint

Populate `config.env` with variables
```
image=1231231.dkr.ecr.region.amazonaws.com/foo/torchserve:0.1
role=arn:aws:iam::1231231:role/SomethingSM
model=s3://foo/torchserve/model
```
Then run `predict.py --deploy True`, this will call sagemaker's [model](https://sagemaker.readthedocs.io/en/stable/api/inference/model.html) library with the arguments:
* `model_data`: s3 location of the model tar.gz file
* `image_uri`: docker image name
* `role`: the arn string of the IAM role created previously (not this is
  different from the arn string of the user)

And calls: [CreateModel](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html) followed by [CreateEndpoint](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpoint.html). When it finishes, you should find the model and endpoint in the AWS console, as well as in the output of the aws cli
```
$ aws sagemaker list-models
$ aws sagemaker list-endpoints
```
Infer against running resources by running `predict.py` without the `--deploy` flag.
Delete resources
```
$ aws sagemaker delete-endpoint --endpoint-name <endpoint name>
$ aws sagemaker delete-model --model-name <model name>
```

## TODO

* Figure out `model.pt` bug. Torchserve logs have
```
[handler] about to torch load, model_def_path /tmp/models/26d0de933c30470f81e2e8e73372b09c/model.py model_pt_path /tmp/models/26d0de933c30470f81e2e8e73372b09c/model.pt
```
But that path only contains `mnist_cnn.pt` if the file produced by `main.py` isn't renamed.
    - This is because the name `model.py` is hardcoded in the initializing
      routine of the `handler.py` file.

* Real time inference: only some instance types are supported (all large?)
```
botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the CreateEndpointConfig operation: 1 validation error detected: Value 'ml.t3.medium' at 'productionVariants.1.member.instanceType' failed to satisfy constraint: Member must satisfy enum value set: [ml.r5d.12xlarge, ml.r5.12xlarge, ml.p2.xlarge, ml.m5.4xlarge, ml.m4.16xlarge, ml.r5d.24xlarge, ml.r5.24xlarge, ml.p3.16xlarge, ml.m5d.xlarge, ml.m5.large, ml.t2.xlarge, ml.p2.16xlarge, ml.m5d.12xlarge, ml.inf1.2xlarge, ml.m5d.24xlarge, ml.c4.2xlarge, ml.c5.2xlarge, ml.c4.4xlarge, ml.inf1.6xlarge, ml.c5d.2xlarge, ml.c5.4xlarge, ml.g4dn.xlarge, ml.g4dn.12xlarge, ml.c5d.4xlarge, ml.g4dn.2xlarge, ml.c4.8xlarge, ml.c4.large, ml.c5d.xlarge, ml.c5.large, ml.g4dn.4xlarge, ml.c5.9xlarge, ml.g4dn.16xlarge, ml.c5d.large, ml.c5.xlarge, ml.c5d.9xlarge, ml.c4.xlarge, ml.inf1.xlarge, ml.g4dn.8xlarge, ml.inf1.24xlarge, ml.m5d.2xlarge, ml.t2.2xlarge, ml.c5d.18xlarge, ml.m5d.4xlarge, ml.t2.medium, ml.c5.18xlarge, ml.r5d.2xlarge, ml.r5.2xlarge, ml.p3.2xlarge, ml.m5d.large, ml.m5.xlarge, ml.m4.10xlarge, ml.t2.large, ml.r5d.4xlarge, ml.r5.4xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.m5.24xlarge, ml.m4.2xlarge, ml.p2.8xlarge, ml.m5.2xlarge, ml.r5d.xlarge, ml.r5d.large, ml.r5.xlarge, ml.r5.large, ml.p3.8xlarge, ml.m4.4xlarge]
```
* Bad request, means you probably tarred up the model and it's root directory.
  Make sure you cd into the model directory and only tar the `model.mar` file
  into s3.

```
    raise error_class(parsed_response, operation_name)
botocore.errorfactory.ModelError: An error occurred (ModelError) when calling the InvokeEndpoint operation: Received client error (400) from model with message "{
  "code": 400,
  "type": "BadRequestException",
  "message": "Parameter model_name is required."
}
". See https://ap-south-1.console.aws.amazon.com/cloudwatch/home?region=ap-south-1#logEventViewer:group=/aws/sagemaker/Endpoints/mnistv1 in account 12312313 for more information.
```

## Appendix

* [Source](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)
* [How SM runs models](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)
* [Preditions against running endpoints](https://sagemaker.readthedocs.io/en/stable/overview.html#how-do-i-make-predictions-against-an-existing-endpoint)
