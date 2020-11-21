# NNs

* Node/Neuron: an implementation of `Activation(i/p * W + bias)`
* Weights: are specific to an edge. Meaning every edge has different weights.
* Activation: A function (typically non-linear) applied to the linear input
value (`ip * W + bias`) by the Neuron. Without Activation, all NNs turn into
linear regressions (i.e predictions of such a network would lie on/around a
plane, hyper-plane or line). Real world phenomenon usually model non-linear
functions, which require non-linear Activations.
    - Activations are specific to the source of an edge between 2 nodes. Meaning
      the same activation value is applied for all edges coming from a node `j`.
regardless of the node this output is going into. Common activations:
        * `Relu`: hidden layers. 0 for x < 0, x for x >= 0
        * `Sigmoid`: o/p in the range of 0 and 1. Probabilities.
        * `Softmax`: typically the o/p layer of classification networks. N probability classes. Sum of all classes = 1.
* Bias: A shift applied to the activation function graph. Bias and Activation
  allow the network to search through the solution space.
    - The bias is specific to the destination node of an edge between 2 nodes.
      Meaning regardless of where the input is coming from, the bias applied is
always the same for a node `j`.
* Gradient descent: training method. Keep subtracting steps from the weights
  as long as the slope of the loss function is negative. It gives us the minima
on the weight x loss graph.
* Cost function: quantifies loss, i.e predicted vs true. Only [certain types](https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications) of
  cost functions can be used in back propogation.
* Stochastic Gradient Descent: avg gradient descent over a batch of inputs.
  Jerky descent down the gradient slope, but more efficient.
* Backpropogation: Draw a DAG of all the inputs that depend on the cost
  function. Compute the gradient of each input with _its_ input. Apply the chain
rule to minimize the cost function with respect to the entire graph. Typically
this results in minimizing the cost wrt the: weights, bias, previous layers activation.
    - The output of back propogation is a set of differentials used to update
      the weights and biases `updated_w = w - d(cost)/d(weights) * step`
    - `d(c)/d(a)` is the root of this loss computation, where `a` is the
      activation that finallys produces the prediction used to compute `c`. This
activation was produced by some `z` (pre-activation function raw `input*w+bias` value). This `z` can be used to compute `d(z)/d(a-1)` where `a-1` is the activation from the previous layer that produced `z`. This value serves as a proxy for `d(c)/d(a)` in minimizing `w-1` and `b-1`.
* Step size: hyperparameter (manual tuned param, as opposed to weights and bias,
  which are tuned by the network as it learns) for tuning gradient descent.
* Dot product: `[ i x j ] * [ j x k ] = [ i x k ]`

## Appendix

* [Primer](https://nestedsoftware.com/2019/05/05/neural-networks-primer-374i.105712.html)
* [HW](https://nestedsoftware.com/2019/08/15/pytorch-hello-world-37mo.156165.html)
