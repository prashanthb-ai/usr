# Deploying and managing pytorch models

* Torch.NN
* Torch.jit
    - Torchscript
* Torch.optim
* Torch.Autograd
* Torch.Vision
* Torch.Data

## Workflow

* Approach
* Prepare data
* Build and train

Sample training loop:
```
for epoch in range(1, 11):
    for data, target in data_loader:
        optimizer.zero_grad()
        prediction = net.forward(data)
        loss = F.nll_loss(prediction, target)
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        torch.save(net, "net.pt")
```
* Publishing model to prod
* Deploy and scale

```
Eager mode -> Script mode (@torch.jit.script)
```
* Add decision gates to Linear layers
* Tape based autodifferentiation
* Torchscript: Serialized torch code that we can load into a non-python server (c++)

## Airflow

```
ETL -> train -> test -> push -> deploy
```
* Airflow + Amundsen (lift)

## CI/CD

* kubeflow, Azure ml + GH + CI/CD platform (jenkins)
    - tune hyperparams
* Tensorboard -> GH -> SRE not happy
    - HW/SW stack, driver version, hard coded heuristics, latency, x datacenter
      latency, doesn't predict outliers
    - Who goes on call for this?
* Tensorboard -> GH -> explain model/look for bias, package for rollout, deploy
    - Canary a deployment: 1,10% ...
    - Auditability: what data the ml researcher trained on
    - Explainability: lots of tools out there, can't even begin to use them
        * Need some basic SHA tracking all the way back to dataset used
* Models can go stale quickly
    - Distributions shift
    - Automated systems that understand this is critical
    - Tooling for data scientists to pick and choose a model for canary vs prod
        * ml flows is doing this?
        * kubeflow and kale: notebook -> python

## Mlops

[Source](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
[Hidden technical debt in ml systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

* Assemply line analogy for ml delivery (continuous, automatic)

```
         -------------local data
         |              |                              Data Science                   Dev
         |           manual steps                       training       S3 storage   Serving
         v           ------------                                           |
Exploratory data -> data prep -> model train -> model eval -> Trained model | Model deploy
                                    ^             |                         |
                                    |_____________|

```
* What's wrong with this model?
    - Tuning is more of a dark art: Manual steps, inflexible, not reusable, error prone
    - Training / serving skew
    - Hard to debug / explain

```
        Train                                       Serve
                                       |
Experimentation -> Continuous training | -> model ci/cd -> Continuous monitoring
                  rebuild training on  |    auditable       High quality improvement
                  new data             |    reversible       signals
```

* Provenance: Who created the mode and what data was it trained on?

## Early stages?

```
Collect phone data -> generate features -> credit model
```
* Collect: text messages, installed apps, contact lists, inapp events, what
  other apps do you have?
* Extract bank balance, #contacts, installed fb app etc
* Bighead feature service
    - zipline: 5
    - ml automator w/ airfolw: 5
    - redspot : 10
    - zipline k/v data store: 5 DS
    - Deepthought: 10
    - Net PMs: 2
* Be strategic on your ml infra
    - feature service
* Do we need to make feature discovery easier?
* Manual loan review
* Linear regressions running on excel
* Iterate till improvements plateau
* Differences between dev, train, prod
* All features were custom code - bugs?

```
Feature service <- get features for user 2143134 @ timestamp, loc
                -> {avg_bank_bal, #referrals, read_faq...}
```
* Don't care about what the feature vector returned is
* API
    - GET `feature/<fname>/<fversion>?pid=<userid>?date=<timestamp>`
    - GET `feature/bank_balance/v0_1?pid=1234`
    - Version of feature that's rolled when a big is fixed
* Unwise for small companies to build custom stuff
    - Gather data: hive, kafka, postgres
    - Feature service
    - Model: pytorch, tf
    - Serve: aws, kubernetes
* feature service: server infra, cache, python framework
* Data source dependencies
    - How many data sources do you really have?
```
Raw data source A                       inference
                    - Feature service - training
Raw data source B                       development
```
* Data science just connects to the feature service
* Code reuse across similar models for similar features
* Architecture: Flask on Beanstalk talking to DynamoDB (no sql)

### Generation of features

* Framework: extractor -> transformers
    - extractor connects to the raw data source
    - transformers: filter, map, reduce (avg, select bank, remove values)
    - output: `average_bank_balance: 23432`
        * 2 features related to bank balance, share the same extractor and
          initial transformers.
    - Base classes: Feature, Transform etc
        * pipeline defined as a list of child classes.


## Issues

* Do we need to invert features?
* Is there a high level of feature reuse?
    - Abstracted data sources
    - Shared features
    - Consistent features
* How often do you add a new feature?
* Modularity between model and feature computation, important?
    - Train old model on buggy feature
    - Fix feature
    - Need to redeploy model
* A feature store is also great for analytics and monitoring, do you need this?
    - Graph between different features
* Speeds up model training - can start from a checkpoint and aggregate. Is model
  training speed an issue?
* Do you have a common source for all features?

### Stated issues

* Is feature quality a problem?
* Are you data sources complex/varied?
    - How frequently do you add a new source
* Do you want to support multiple models?
    - Development, exploration or new models quickly using existing features
    - Make new models without building feature set
* Are you features compute intensive?
    - Caching features
* Clear pids? time, user etc
* What is the timescale of your feature changing? Store updates etc

### Q&A

* What's in model vs store?
* Batch jobs for feature generation?
* How far back to store data?
    - When do you version?

## Sagemaker

* 4:1 service: Notebook server, training, hosting, feature. Fully managed.
    - pytorch, skikitlearn, tf: estimators?
```
from sagemaker import pytorch
pytorch.Estimator
```
* Estimator
* Multiple inference steps
* Container that just contains hyperparams and model code
    - Experimentation with reusable preprocessing code
    - Built in sagemaker algorithms
* Text -> sex of the speaker via transforms

* TorchServe
    - local: CLI
    - SageMaker
    - EKS
    - default handlers?



## Appendix

* [Ref](https://www.youtube.com/watch?v=EkELQw9tdWE&t=702s)
* [Features](https://www.youtube.com/watch?v=GcimUEwbydo&list=PLJHNhcCAHd7inG1RU53j4pNPkP8Ws8jMo&index=2&t=1004s)
* [Sagemaker: AWS](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/)
