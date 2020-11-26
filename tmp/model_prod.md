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








## Appendix

* [Ref](https://www.youtube.com/watch?v=EkELQw9tdWE&t=702s)
