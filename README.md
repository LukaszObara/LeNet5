# LeNet5
We describe the components and how to built a convolution neural network (CNN) similar to that described in <u>Gradient Based Learning Applied to Document Recognition</u> by LeCunn et al.. We built all the necessary layers/components  using `Theano` and test various gradient descent methods on the MNIST dataset.

The network will consist of a series of convolution layers, pooling layers and a fully connected layer and can be visualized by the following diagram 

![Alt text](https://github.com/LukaszObara/LeNet5/blob/master/Notebook/architec2.png "LeNet5").

We show how to implement various forms of gradient descent and show how to define the `Exponential Linear Unit (ELU)` as described in <u>Fast and accurate deep network learning by exponetial lienar units</u> by Clevert et al. 

The full describtion can be found in the ipython notebook located in the `Notebook` directory.

# References
Bengio Yoshua, Glorat Xavier, Understanding the difficulty of training deep feedforward neural networks, #AISTATS#, pages 249–256, 2010
Clevert Djork-Arne, Unterthiner Thomas, Hochreiter Sepp, Fast And Accurate Deep Network Learning by Exponential Linear Units (ELU), *ICLR* 2016, https://arxiv.org/abs/1511.07289
He Kaiming et al. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, #ICCV# 2015, pp. 1026-1034
LeCun Yann et al., Gradient-Based Learning Applied to Document Recognition, *PROC. OF THE IEEE.*, Nov 1998
