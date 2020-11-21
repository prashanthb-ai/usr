# NNs

* Node/Neuron: an implementation of `Activation(i/p * W + bias)`
* Activation: A function (typically non-linear) applied to the linear input
value (`ip * W + bias`) by the Neuron. Without Activation, all NNs turn into
linear regressions (i.e predictions of such a network would lie on/around a
plane, hyper-plane or line). Real world phenomenon usually model non-linear
functions, which require non-linear Activations.
    - Relu: hidden layers. 0 for x < 0, x for x >= 0
    - Sigmoid: o/p in the range of 0 and 1. Probabilities.
    - Softmax: typically the o/p layer of classification networks. N probability classes. Sum of all classes = 1.
* Bias: A shift applied to the activation function graph. Bias and Activation
  allow the network to search through the solution space.

## Appendix

* [Primer](https://nestedsoftware.com/2019/05/05/neural-networks-primer-374i.105712.html)
* [HW](https://nestedsoftware.com/2019/08/15/pytorch-hello-world-37mo.156165.html)
