Kubeflow deployment
---


## TLDR

See deployment [guide](https://www.kubeflow.org/docs/other-guides/kustomize/)
```
# Private alias, look at kubeflow startup instructions for relevant cmdline
$ minikube start --cpus 6 --memory 12288 --disk-size=120g --extra-config=apiserver.service-account-issuer=api --extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/apiserver.key --extra-config=apiserver.service-account-api-audiences=api
```
Check kubernetes version
```
$ k version
Client Version: version.Info{Major:"1", Minor:"18", GitVersion:"v1.18.3", GitCommit:"2e7996e3e2712684bc73f0dec0200d64eec7fe40", GitTreeState:"clean", BuildDate:"2020-05-20T12:52:00Z", GoVersion:"go1.13.9", Compiler:"gc", Platform:"linux/amd64"}
Server Version: version.Info{Major:"1", Minor:"19", GitVersion:"v1.19.4", GitCommit:"d360454c9bcd1634cf4cc52d1867af5491dc9c5f", GitTreeState:"clean", BuildDate:"2020-11-11T13:09:17Z", GoVersion:"go1.15.2", Compiler:"gc", Platform:"linux/amd64"}
```
Check kubeflow compatibility [matrix](https://www.kubeflow.org/docs/started/k8s/overview/#minimum-system-requirements), and apply the appropriately versioned yaml. Note if you have problems deploying kubeflow 1.1, see this [issue](https://github.com/kubeflow/website/issues/2206).
```
$ kfctl apply -V -f https://github.com/kubeflow/manifests/blob/v1.1-branch/kfdef/kfctl_k8s_istio.v1.1.0.yaml
```

Kfctl uses [kf.yaml](https://raw.githubusercontent.com/kubeflow/manifests/v1.2-branch/kfdef/kfctl_k8s_istio.v1.2.0.yaml) as the index to lookup resources from the appropriately versioned [manifests](https://github.com/kubeflow/manifests/archive/v1.2-branch.tar.gz) on GH, and then runs `kustomize` against them to create the following resources on the kubernetes cluster. Look at `.cache/manifests` for where it actually stores the manifests.

```
$ watch 'kubectl get po -n kubeflow'
```

Delete the cluster
```
$ kfctl delete -V -f kf.yaml
```
Note that to clear all state you will also need to rm the `./kustomize` and `./.cache` directories, otherwise eg: trying to install a different kubeflow version will fail.

## Notes


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


```
$ minikube start
```



## Text summarization pipeline

TODO: Run on minikube

## Appendix

* [kubeflow aws](https://aws.amazon.com/blogs/opensource/kubeflow-amazon-eks/)
* [Minikube](https://www.kubeflow.org/docs/started/workstation/minikube-linux/)
* [Fix for hairpin](https://github.com/kubernetes/minikube/issues/8949#issuecomment-734106051)
