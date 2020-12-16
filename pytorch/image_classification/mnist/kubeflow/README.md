Kubeflow deployment
---


* Track all the artifacts needed to version control your models
    - registry
    - changes
    - data
* Kubeflow is a collection of various components running on Kubernetes

## Challenges to building ML Pipeline

* How to track all the models you're experimenting with?
    - Accuracy
    - All users
* Constant stream of new models post experimentation stage
    - Drift to show that the accuracy drops
    - Take steps if it drops below a threshold
* Training / serving skew
    - accuracy drops during serving
    - code used by apps to do predicions has missing data/nulls/bad transforms
    - one time training process and runtime process needs to have exact same
      data pipelines
* Freshness requirements - retraining models, versioning
* Data Science = building a model
    - ML serving is a lot more
    - Validation? monitoring? logging?
    - Make it easier to develop, deploy and manage the models
* Reproducibility of ML systems
```
    Model          |
    Ux             |
    Tooling        |
    Framework      |
    Storage        |
    Runtime - vm - |    dev , test , prod
    Drivers
    OS
    Accelerator
    HW
```

## Capabilities

* IDE: jupyter notebooks
* Training operators for lots of frameworks
* Datamanagement versioning
* Hyperparameter turning (learning rate, batch size - distribute load over Kube
  to find optimal parameters)
* Metadata management?
* Serving: REST endpoint to use the model


DIY vs Hosted
* Eventually migrate vs Economics good for 5m

* Composable
    - Data ingestion, transformation, validation
    - Training
    - Building model, validation
    - Roll out
    - Serving
    - Monitoring, Logging
* Scalable
* Portable


## Minikube setup

* Iterate locally on minikube
* Spawner option to launch a notebook
* Ksonette for model configuration, set local training
    - Packging into pods
    - User spec -> TF/pytorch configuration via pod spec
* Serving
    - default/canary endpoints (99/1%)
    - predict / explain

## Text summarization pipeline

TODO: Run on minikube

## Appendix

* [kubeflow aws](https://aws.amazon.com/blogs/opensource/kubeflow-amazon-eks/)
* [Minikube](https://www.kubeflow.org/docs/started/workstation/minikube-linux/)
