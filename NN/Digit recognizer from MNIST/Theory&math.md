Neural Networks performs better on nonlinear functions. Threre is forward functions which eventually reach into loss function. Moreover we want to know the effect of every params on that loss function. It's really a daunting task. But using computational graph and local gradients it becomes a simple task.

To calculate gradients on backpropagation, the task is to multiply upstream gradient with the local gradients.

In this particular problem set, we take MNIST dataset which contains digits. Our task is to build a model which can recognize digits. For this task we will build a neural network. Here we will take one hidden layer having 10 neurons and output layer having 10 neurons. Because the number of class label is 10.

### Dataset
Mnist datset. Contains grayscale image of 28X28. Therefore we can stretch each image 28X28=784 into columns.

###  Forward Pass
As we discussed about our problem and neural network architecture, we can visualize the neural network and the math behind the task. First of all we calculate **Forward Pass**. We will stretch each image into columns. Therefore our input matrix will be of shape [num_features X num_examples]. For the first hidden layer, the weight matrix W1 will be of shape [num_neurons X num_features]. And the weights matrix W2 will be of shape [num_labels X num_features]. b1 will be of shape [num_neurons X 1] and b2 will be of shape [num_labels X 1]. Therefore forward pass will be like,

- $Z^{[1]} = W^{[1]} X + b^{[1]}$
- $A^{[1]} = g_{\text{ReLU}}(Z^{[1]}))$
- $Z[2]=W[2]A[1]+b[2]$
- $A[2]=gsoftmax(Z[2])$


### Backward Pass:

after completing forward pass we will eventually evalutate loss. The loss at the output layer is simply the differences between the output labes and the ground truth labels. We want to calculate the impact of each weights and biases on that loss functions. We do this using chain rule. We also discussed simpliest formula, gradient at any node is simply the 
**upstream gradient X local gradients** .

- $dZ[2]=A[2]−Y$
- $dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$
- $dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}$
- $dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]})$
- $dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$
- $dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}$


### Parameters update
- $W^{[2]} := W^{[2]} - \alpha dW^{[2]}$
- $b[2]:=b[2]−αdb[2]$
- $W[1]:=W[1]−αdW[1]$
- $b^{[1]} := b^{[1]} - \alpha db^{[1]}$