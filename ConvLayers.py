# ConvLayers.py
#### Libraries
# Third Party Libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

def elu(x, alpha=1.0):
	return T.switch(x > 0, x, T.exp(x)-1)

def l2_reg(x, lmbd=0.05):
	"""
	L_2 regularization 

	Parameters
	----------

	returns
	----------
	"""

	l2 = 0
	for elements in x:
		l2 += T.sum(elements[0]**2)

	return lmbd / 2 * l2 

class ConvLayer(object):
	"""
	Performs a convolution on an input layer.

	Parameters
	----------
	:type input: theano.tensor.dtensor4
	:param input: symbolic image tensor, of shape image_shape

	:type filter_shape: tuple or list of length 4
	:param filter_shape: (number of filters, num input feature maps,
						  filter height, filter width)

	:type image_shape: tuple or list of length 4
	:param image_shape: (batch size, num input feature maps, 
						 image height, image width)

	:type padding: tuple or list of length 2
	:param padding: 

	:type stride: tuple or list of length 2
	:param stride: 

	:type activation_fn: theano.Op or function
	:param activation_fn: Non linearity to be applied in the hidden 
						  layer (sigmoid, tanh, relu, elu)

	References
	----------
    .. [1] http://deeplearning.net/tutorial/lenet.html
    .. [2] http://neuralnetworksanddeeplearning.com/chap6.html
    .. [3] http://cs231n.github.io/convolutional-networks/
	"""

	def __init__(self, input, filter_shape, image_shape, padding=(0, 0), 
				 stride=(1, 1), activation_fn=None, seed=3235):

		assert image_shape[1] == filter_shape[1]

		# rng = np.random.RandomState(seed)

		self.input = input
		self.filter_shape = filter_shape
		self.image_shape = image_shape
		self.activation_fn = activation_fn

		fan_in = np.prod(filter_shape[1:])
		fan_out = filter_shape[0]*np.prod(filter_shape[2:]) // 2
		W_bound = np.sqrt(6/(fan_in+fan_out))
		w = np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape)
		b_vals = np.random.uniform(size=filter_shape[0])

		# Initiliaze weights with random variables
		self.W = theano.shared(name='weights',
							   value=w.astype(theano.config.floatX),
							   borrow=True)
		self.b = theano.shared(name='bias',
							   value=b_vals.astype(theano.config.floatX), 
							   borrow=True)

		conv_out = conv2d(input=input, filters=self.W, border_mode=padding,
						  subsample=stride, filter_shape=filter_shape, 
						  input_shape=image_shape)

		l_output = conv_out + self.b.dimshuffle(('x', 0, 'x', 'x'))
		self.output = (l_output if activation_fn is None 
					   else activation_fn(l_output))

		# Parameters of the model
		self.params = [self.W, self.b]


class BatchNormLayer(object):
	"""
	This layer implements batch normalization of its inputs. This is 
	performed by taking the mean and standard deviation across axis 0.

    .. math::
	    y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

	A remedy to internal covariate shift, the solution is to normalize 
	each batch by both mean and variance by insertin a BatchNorm layer 
	immediately after fully connected layers (or convolutional layers, 
	and before non-linearities.

	Parameters
	----------
	:type input: theano.tensor.dtensor4
	:param input: symbolic image tensor, of shape image_shape

	:type gamma: theano.tensor.dtensor4
	:param gamma: symbolic image tensor, of shape image_shape

	:type beta: theano.tensor.dtensor4
	:param beta: symbolic image tensor, of shape image_shape

	:type activation_fn: theano.Op or function
	:param activation_fn: Non linearity to be applied in the hidden 
						  layer (sigmoid, tanh, relu, elu)

	References
	----------
    .. [1] https://arxiv.org/abs/1502.03167
    .. [2] https://arxiv.org/abs/1502.03167
	"""

	def __init__(self, input, shape, gamma=None, beta=None, epsilon=1e-6,
				 activation_fn=None):
		self.input = input
		self.shape = shape

		rng = np.random.RandomState(45)

		if gamma is None:
			gamma_values = rng.uniform(low=-1.0, high=1.0, size=shape)\
							.astype(theano.config.floatX)
			gamma = theano.shared(name='gamma', value=gamma_values, 
								  borrow=True)

		if beta is None:
			beta_values = np.zeros(shape=shape, dtype=theano.config.floatX)\
							.astype(theano.config.floatX)
			beta = theano.shared(name='beta', value=beta_values, borrow=True)

		self.gamma = gamma
		self.beta = beta

		self.mean = T.mean(input, axis=0)
		self.std = T.std(input + epsilon, axis=0) 

		l_output = T.nnet.bn.batch_normalization(input, self.gamma, self.beta, 
												 self.mean, self.std)

		self.output = (l_output if activation_fn is None 
					   else activation_fn(l_output))

		self.params = [self.gamma, self.beta]


class PoolingLayer(object):
	"""
	Performs a pooling operation, a form of non-linear down-sampling. 
	The pooling operation partitions the input image into a set of 
	non-overlapping rectangles and, for each such sub-region outputs  
	the corresponding value.

	Parameters
	----------
	:type input: theano.tensor.dtensor4
	:param input: symbolic image tensor, of shape image_shape

	:type pool_shape: tuple or list of length 2
	:param pool_shape: the downsampling (pooling) factor (#rows, #cols)

	:type ignore_border: bool (1-Default)
	:param ignore_border: inlcude border in computation of pooling
	
	References
	----------
    .. [1] http://deeplearning.net/tutorial/lenet.html
    .. [2] http://neuralnetworksanddeeplearning.com/chap6.html
    .. [3] http://cs231n.github.io/convolutional-networks/
	"""

	def __init__(self, input, pool_shape=(2, 2), ignore_border=True,
				 activation_fn=None):
		self.input = input
		self.pool_shape = pool_shape
		self.ignore_border = ignore_border

		l_output = pool.pool_2d(input=input, ds=pool_shape, 
								ignore_border=ignore_border)

		self.output = (l_output if activation_fn is None 
			   		   else activation_fn(l_output))


class FC(object):
	"""
	Typical hidden layer of a MLP: units are fully-connected and have
	sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
	and the bias vector b is of shape (n_out,).

	NOTE : The nonlinearity used here is tanh

	Hidden unit activation is given by: tanh(dot(W, input.T) + b)

	Parameters
	----------
	:type rng: numpy.random.RandomState
	:param rng: a random number generator used to initialize weights

	:type input: theano.tensor.dmatrix
	:param input: a symbolic tensor of shape (n_examples, n_in)

	:type n_in: int
	:param n_in: dimensionality of input

	:type n_out: int
	:param n_out: number of hidden units

	:type activation: theano.Op or function
	:param activation: Non linearity to be applied in the hidden
	                   layer  (sigmoid, tanh, relu, elu)

	References
	----------
    .. [1] http://deeplearning.net/tutorial/mlp.html
    .. [2] http://neuralnetworksanddeeplearning.com/chap6.html
    .. [3] http://cs231n.github.io/convolutional-networks/
	"""

	def __init__(self, input, n_in, n_out, W=None, b=None, seed=35,
	         	 activation_fn=None):

		# rng = np.random.RandomState(seed)

		self.input = input

		if W is None:
			W_values = np.random.uniform(low=-np.sqrt(6./(n_in+n_out)),
								   high=np.sqrt(6./(n_in+n_out)),
								   size=(n_out, n_in)).astype(theano.config.floatX)

			if activation_fn == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(name='Weights', value=W_values, borrow=True)


		if b is None:
			b_values = np.zeros(n_out, dtype=theano.config.floatX)
			b = theano.shared(name='bias', value=b_values, borrow=True)

		self.W = W
		self.b = b

		l_output = (T.dot(self.W, input.T)).T + self.b
		self.output = (l_output if activation_fn is None 
					   else activation_fn(l_output))

		# Parameters of the fully connected layer
		self.params = [self.W, self.b]

def dropout_from_layer(layer, p, seed=35):
    """
    p is the probablity of dropping a unit

    Parameters
	----------
	:type rng: numpy.random.RandomState
	:param rng: a random number generator used to initialize weights

	:type layer: 
	:param layer: Input layer to perform dropout

	:type p: float32
	:param p: Probablity of dropping a unit

	Returns
	-------

	References
	----------
	.. [1] https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    """

    srng = T.shared_randomstreams.RandomStreams(seed=seed)
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)

    return output

class Dropout(FC):
	def __init__(self, input, n_in, n_out, W=None, b=None, seed=35,
				 activation_fn=None, dropout_rate=0.5): 

		super().__init__(input=input, n_in=n_in, n_out=n_out, W=W, b=b, 
						 seed=seed, activation_fn=activation_fn)

		self.output = dropout_from_layer(self.output, p=dropout_rate, seed=seed)