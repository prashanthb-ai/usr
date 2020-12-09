# Model Serving

Serving a pytorch model dump via Torchserve

## Prerequisites

* Install [Conda](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)
* Install python 3.8 via conda (torchserve needs python 3.8)
```
$ conda create -n python38 python=3.8
$ conda activate python38
```
* Install torch-model-archiver
```
$ pip install torch-model-archiver
$ torch-model-archiver --help
```
* Install torchserve
```
$ pip install torchserve
```

## Classifiers

The default classifier used below is coded [here](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py). It's specified via the `--handler` flag, which can point at either one of the [predefined handlers](https://github.com/pytorch/serve/issues/374#issuecomment-632015182), or a custom .py file.


## Archival/hosting the model

Generating the `.mar` file
```
$ torch-model-archiver --model-name dnn \
--version 1.0 --model-file dnn.py \
--serialized-file /home/beeps/rtmp/ml/models/model.pth \
--handler image_classifier
```

## Hosting the model

Directly on machine
```
$ torchserve --start --model-store /path/to/model_store --models dnn=dnn.mar
$ curl "http://localhost:8081/models"
{
  "models": [
    {
      "modelName": "dnn",
      "modelUrl": "dnn.mar"
    }
  ]
}
```

In docker
```

```

## Make inferences

See logs stored in `./logs` if there's an issue.
```
$ curl -X POST http://127.0.0.1:8080/predictions/dnn -T kitten.jpg
$ torchserve --stop
```

## TODO

* Write torchserve [handlers](https://towardsdatascience.com/how-to-deploy-your-pytorch-models-with-torchserve-2452163871d3)
* Install standard dnn model
* Reconfigure logging


## Appendix

* [Torchserve](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/)
