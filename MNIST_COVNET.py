#### Libraries
# Standard Libraries
## To change to `device=gpu0`
# import os 
# os.environ['THEANO_FLAGS'] = "device=gpu0" 

# Third Party Libraries
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

def scalar_to_vec(labels_dense, num_classes):
	"""
	Converts the dense label into a sparse vector. 

	Parameters
	----------
	:type labels_dense: int8 
	:param labels_dense: dense label

	:type num_classes: int8
	:param num_classes: number of classes

	Returns
	-------
	<class 'numpy.ndarray'>
		A numpy array of length `num_classes` of zeros except for a 1 in 
		the position of `labels_dense`. 

	Examples
	--------
	>>> scalar_to_vec(4, 10)
	[ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]

	"""
	assert type(labels_dense) == int
	assert type(num_classes) == int

	vec = np.zeros(num_classes)
	vec[labels_dense] = 1

	return vec

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
		
########################################################################

# Model
print('---Building Model---')

X = T.tensor4(name='X', dtype=theano.config.floatX) 
Y = T.imatrix(name='Y')
y = T.ivector(name='y')
lr = T.scalar(name='learning_rate', dtype=theano.config.floatX)

nkerns = [8, 32]
batch_size = 256
act_f = elu

conv_layer1 = ConvLayer(input=X, 
						filter_shape=(nkerns[0], 1, 3, 3), 
						image_shape=(batch_size, 1, 28, 28),
						activation_fn=None)
# bn_layer1 = BatchNormLayer(input=conv_layer1.output,
# 						   shape=(batch_size, nkerns[0], 26, 26),
# 						   activation_fn=None)
pool_layer1 = PoolingLayer(input=conv_layer1.output,
						   activation_fn=act_f)
conv_layer2 = ConvLayer(input=pool_layer1.output,
						filter_shape=(nkerns[1], nkerns[0], 5, 5), 
						image_shape=(batch_size, nkerns[0], 13, 13),
						activation_fn=None)
# bn_layer2 = BatchNormLayer(input=conv_layer2.output,
# 						   shape=(batch_size, nkerns[1], 11, 11),
# 						   activation_fn=None)
pool_layer2 = PoolingLayer(input=conv_layer2.output,
						   activation_fn=act_f)

# outputs from convolution network need to be flattend before being 
# passed through to the the fully-connected layer
fc_layer_input = pool_layer2.output.flatten(2) 

fc_layer1 = FC(input=fc_layer_input,
				n_in=nkerns[1] * 4 * 4,
				n_out=512,
				activation_fn=act_f)
# bn_layer3 = BatchNormLayer(input=fc_layer1.output,
# 						   shape=(batch_size, 64),
# 						   activation_fn=act_f)
fc_layer2 = FC(input=fc_layer1.output,
				n_in=512,
				n_out=10,
				activation_fn=act_f)
# fc_layer3 = FC(input=fc_layer2.output,
# 			   n_in=256,
# 			   n_out=10,
# 			   activation_fn=act_f)


# with full batch normalization
# params = fc_layer2.params + bn_layer3.params + fc_layer1.params\
# 	   + bn_layer2.params + conv_layer2.params + bn_layer1.params\
# 	   + conv_layer1.params

# without batch normalization
params = fc_layer2.params + fc_layer1.params\
	   + conv_layer2.params + conv_layer1.params

cost_input = T.nnet.nnet.softmax(fc_layer2.output)
cost = T.mean(T.nnet.nnet.categorical_crossentropy(cost_input, Y)) \
	 + l2_reg(params)

grads = T.grad(cost, params)

# def sgd(l_rate, parameters, grads): 
# 	""" 
# 	Stochastic Gradient Descent.

# 	Parameters
# 	----------
# 	:type lr: theano.tensor.scalar
# 	:param lr: Initial learning rate
	
# 	:type parameters: theano.shared
# 	:params parameters: Model parameters to update
	
# 	:type grads: Theano variable
# 	:params grads: Gradients of cost w.r.t to parameters
# 	"""

# 	updates = []
# 	for param, grad in zip(parameters, grads):
# 		updates.append((param, param - l_rate * grad))

# 	return updates

# def momentum(l_rate, parameters, grads, momentum=0.9):
# 	"""
# 	Momentum update

# 	Parameters
# 	----------
# 	:type lr: theano.tensor.scalar
# 	:param lr: Initial learning rate
	
# 	:type parameters: theano.shared
# 	:params parameters: Model parameters to update

# 	:type grads: Theano variable
# 	:params grads: Gradients of cost w.r.t to parameters

# 	:type momentum: float32
# 	:params momentum: 
# 	"""

# 	def update_rule(param, velocity, df):
# 		v_next = momentum * velocity - l_rate * df
# 		updates = (param, param+v_next), (velocity, v_next)

# 		return updates

# 	assert momentum <=1 and momentum >= 0
	
# 	velocities = [theano.shared(name='v_%s' % param,
# 								value=param.get_value() * 0., 
# 								broadcastable=param.broadcastable) 
# 				  for param in parameters]

# 	updates = []
# 	for p, v, g in zip(parameters, velocities, grads):
# 		param_updates, vel_updates = update_rule(p, v, g)
# 		updates.append(param_updates)
# 		updates.append(vel_updates)

# 	return updates

# def nag(l_rate, momentum=0.9, anneal_rate=1e-2, parameters=None, grads=None):
# 	"""
# 	Momentum update

# 	Parameters
# 	----------
# 	:type l_rate: theano.tensor.scalar
# 	:param l_rate: Initial learning rate

# 	:type momentum: float32
# 	:params momentum: 

# 	:type anneal_rate: float32
# 	:param anneal_rate: 
	
# 	:type parameters: theano.shared
# 	:params parameters: Model parameters to update

# 	:type grads: Theano variable
# 	:params grads: Gradients of cost w.r.t to parameters

# 	:type noise: bool
# 	:params noise: 
# 	"""
# 	t = theano.shared(name='time_step', value=np.int16(0))

# 	def update_rule(param, velocity, df):
# 		v_prev = velocity
# 		v = momentum * velocity - l_rate * T.exp(-anneal_rate*t) * df
# 		x = momentum * v_prev + (1-momentum) * v
# 		updates = (param, param+x), (velocity, v)

# 		return updates
	
# 	velocities = [theano.shared(name='v_%s' % param,
# 								value=param.get_value() * 0., 
# 								broadcastable=param.broadcastable) 
# 				  for param in parameters]

# 	updates = []
# 	for p, v, g in zip(parameters, velocities, grads):
# 		param_updates, vel_updates = update_rule(p, v, g)
# 		updates.append(param_updates)
# 		updates.append(vel_updates)
# 	updates.append((t, t+1))

# 	return updates

def rmsprop(l_rate, d_rate=0.9, epsilon=1e-6, parameters=None, grads=None):
	"""
	Momentum update

	Parameters
	----------
	:type lr: theano.tensor.scalar
	:param lr: Initial learning rate
	
	:type parameters: theano.shared
	:params parameters: Model parameters to update

	:type grads: Theano variable
	:params grads: Gradients of cost w.r.t to parameters

	:type momentum: float32
	:params momentum: 
	"""

	one = T.constant(1.0)

	def update_rule(param, cache, df):
		cache_val = d_rate * cache + (one-d_rate) * df**2
		x = l_rate * df / (T.sqrt(cache_val) + epsilon)
		updates = (param, param-x), (cache, cache_val)

		return updates
	
	caches = [theano.shared(name='c_%s' % param,
								value=param.get_value() * 0., 
								broadcastable=param.broadcastable) 
			  for param in parameters]

	updates = []
	for p, c, g in zip(parameters, caches, grads):
		param_updates, cache_updates = update_rule(p, c, g)
		updates.append(param_updates)
		updates.append(cache_updates)

	return updates

# def adam(l_rate, beta1=0.9, beta2=0.999, epsilon=1e-6, parameters=None, 
# 		 grads=None):

# 	one = T.constant(1.0)
# 	t = theano.shared(name='iteration', value=np.float32(1.0))

# 	def update_rule(param, moment, velocity, df):
# 		m_t = beta1 * moment + (one-beta1) * df
# 		v_t = beta2 * velocity + (one-beta2) * df**2
# 		m_hat = m_t/(one-beta1**(t))
# 		v_hat = v_t/(one-beta2**(t))
# 		x = (l_rate * m_hat / (T.sqrt(v_hat) + epsilon))
# 		updates = (param, param-x), (moment, m_t), (velocity, v_t)

# 		return updates
	
# 	moments = [theano.shared(name='m_%s' % param,
# 							 value=param.get_value() * 0., 
# 							 broadcastable=param.broadcastable) 
# 			   for param in parameters]

# 	velocities = [theano.shared(name='v_%s' % param,
# 								value=param.get_value() * 0., 
# 								broadcastable=param.broadcastable) 
# 				  for param in parameters]

# 	updates = []
# 	for p, m, v, g in zip(params, moments, velocities, grads):
# 		p_update, m_update, v_update = update_rule(p, m, v, g)
# 		updates.append(p_update)
# 		updates.append(m_update)
# 		updates.append(v_update)
# 	updates.append((t, t+1))

# 	return updates

# theano functions for training and validation 
train = theano.function(inputs=[X, Y, lr], outputs=cost, 
						updates=rmsprop(l_rate=lr, parameters=params, 
									 grads=grads),
						allow_input_downcast=True)

# Validation results
pred_result = cost_input.argmax(axis=1)
accu = theano.function(inputs=[X, y], outputs=T.sum(T.eq(pred_result, y)), 
					   allow_input_downcast=True)

pred = theano.function(inputs=[X], outputs=pred_result, 
					   allow_input_downcast=True)

print('Finished Building LeNet5')

########################################################################

def train_model(training_data, validation_data, test_data=None,
				learning_rate=1e-2, epochs=300):

	print('---Training Model---')
	predicted_results = [] 

	total_values, total_val_values = len(training_data), len(validation_data)

	for epoch in range(epochs):
		print('Currently on epoch {}'.format(epoch+1))
		np.random.shuffle(training_data)

		mini_batches = [training_data[k: k+batch_size]
						for k in range(0, total_values, batch_size)] 

		training_cost, accuracy = 0, 0
		training_cost_list, accuracy_list = [], []

		for mini_batch in mini_batches:
			labels = mini_batch[:, 0]
			label_matrix = np.zeros(shape=(256, 10), dtype=theano.config.floatX)
			
			for i, label in enumerate(labels):
				vec = scalar_to_vec(int(label), 10)
				label_matrix[i] = vec

			digits = mini_batch[:, 1:]/255
			digits = digits.reshape(-1, 1, 28, 28)
			cost_ij = train(digits, label_matrix, learning_rate)
			training_cost += cost_ij
			accuracy += accu(digits, labels)

		training_cost_list.append(training_cost/len(mini_batch))
		accuracy_list.append(accuracy/total_values)
		
		print('The accuracy is: {}'.format(accuracy/total_values))
		print('The loss is: {}'.format(training_cost/len(mini_batch)))
		print('--------------------------')

	if np.any(test_data):
		print('===================================')
		print('Using test data to predict results')
		total_values = len(test_data)

		mini_batches = [test_data[k: k+batch_size]
						for k in range(0, total_values, batch_size)] 
		
		for mini_batch in mini_batches:
			digits = mini_batch[:, :]/255
			digits = digits.reshape(-1, 1, 28, 28)
			result = pred(digits)
			predicted_results = np.append(predicted_results, result)

		print('Done')

	return training_cost_list, accuracy_list, predicted_results


if __name__ == '__main__':

	location = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning'+\
				'\\Kaggle\\MNIST\\augmented_train.npy'
	location_test = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning'+\
					'\\Kaggle\\MNIST\\test_padded.npy'
	# loc_save_train = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning'+\
	# 				 '\\Kaggle\\MNIST\\Results\\train_results_relu_1632.npy'
	# loc_save_accu = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning'+\
	# 				 '\\Kaggle\\MNIST\\Results\\accu_results_relu_1632.npy'
	loc_submission = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning'+\
					 '\\Kaggle\\MNIST\\Results\\submission.csv'
	data = np.load(location)
	data_train, data_val = train_test_split(data, test_size=12800, 
											random_state=23)	
	test_val = np.load(location_test)

	t_data, a_data, label = train_model(data_train, data_val, test_val,
										learning_rate=1e-4, epochs=80)

	image_id = np.arange(1, 28001)
	label = label[:28000].astype(np.int8)
	results = {'ImageId': image_id, 'label': label}
	df = pd.DataFrame(results)
	df.to_csv(loc_submission, index=False)