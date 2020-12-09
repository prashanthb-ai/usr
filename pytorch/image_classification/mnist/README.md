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

## TODO

* Figure out `model.pt` bug. Torchserve logs have
```
[handler] about to torch load, model_def_path /tmp/models/26d0de933c30470f81e2e8e73372b09c/model.py model_pt_path /tmp/models/26d0de933c30470f81e2e8e73372b09c/model.pt
```
But that path only contains `mnist_cnn.pt` if the file produced by `main.py` isn't renamed.

## Appendix

* [Source](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)
