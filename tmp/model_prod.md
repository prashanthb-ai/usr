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



## Appendix

* [Ref](https://www.youtube.com/watch?v=EkELQw9tdWE&t=702s)
