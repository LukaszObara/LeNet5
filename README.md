# LeNet5
We describe the components and how to built a convolution neural network (CNN) similar to that described in <em>Gradient Based Learning Applied to Document Recognition</em> by LeCunn et al.. All the necessary layers/components  are built using `Theano`. In addition, a number gradient descent methods are tested on the MNIST dataset.

The network will consist of a series of convolution layers, pooling layers and a fully connected layer and can be visualized by the following diagram 

![Alt text](https://github.com/LukaszObara/LeNet5/blob/master/images/architec2.png "LeNet5").

We show how to implement various forms of gradient descent and show how to define the `Exponential Linear Unit (ELU)` as described in <em>Fast and accurate deep network learning by exponetial lienar units</em> by Clevert et al. 

A full describtion can be found in the ipython notebook located in the ipython notebook `LeNet5.ipynb`.

# References
<ol>
<li>Bengio Yoshua, Glorat Xavier, <em>Understanding the difficulty of training deep feedforward neural networks</em>, AISTATS, pages 249–256, 2010</li>
<li>Clevert Djork-Arne, Unterthiner Thomas, Hochreiter Sepp, <em>Fast And Accurate Deep Network Learning by Exponential Linear Units (ELU)</em>, ICLR 2016, https://arxiv.org/abs/1511.07289</li>
<li>Goodfellow Ian, Bengio Yoshua, Courville Aaron, <em>Deep Learning</em>, MIT Press, 2016, http://www.deeplearningbook.org</li>
<li>He Kaiming et al., <em>Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification</em>, ICCV, 2015, pp. 1026-1034</li>
<li>LeCun Yann et al., <em>Gradient-Based Learning Applied to Document Recognition</em>, PROC. OF THE IEEE., Nov 1998</li>
</ol>


### TODO:
Rewrite LeNet5 code in a more modular fashion
