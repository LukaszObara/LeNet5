#### Libraries
# Standard Libraries
# To change to `device=gpu0`
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu0"  

# Third Party Libraries
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

rng = np.random.RandomState(35)

inputs = T.tensor4(name='inputs', dtype=theano.config.floatX)
y = T.vector(name='y', dtype=theano.config.floatX)

nkerns = [4, 5]
w_shp = (nkerns[0], 1, 3, 3)
w_bound = np.sqrt(np.prod(w_shp[1:]))
w_bound_fc = np.sqrt(6./(45+20))

def elu(x, alpha=1.0):
	return T.switch(x > 0, x, T.exp(x)-1)

# W = theano.shared(name='Weights',
# 				  value=rng.randint(low=0, high=9, size=w_shp)
# 				  			.astype(theano.config.floatX))

w0_values = rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=w_shp)
w1_values = rng.uniform(low=-1.0/2, high=1.0/2, 
						size=(nkerns[1], nkerns[0], 3, 3))
w2_values = rng.uniform(low=-w_bound_fc, high=w_bound_fc, size=(20, 45))
w3_values = rng.uniform(low=-np.sqrt(6/30), high=np.sqrt(6/30), size=(10, 20))
b0_values = rng.uniform(low=-0.5, high=0.5, size=nkerns[0])
b1_values = rng.uniform(low=-0.5, high=0.5, size=nkerns[1])
b2_values = rng.uniform(low=-0.5, high=0.5, size=1)
b3_values = rng.uniform(low=-0.5, high=0.5, size=1)

# Weights and bias for layer 0
W0 = theano.shared(name='Weights0',
				   value=w0_values.astype(theano.config.floatX),
				   borrow=True)
b0 = theano.shared(name='bias0', 
		           value=b0_values.astype(theano.config.floatX),
		           borrow=True)
# Weights and bias for layer 1
W1 = theano.shared(name='Weights1',
				   value=w1_values.astype(theano.config.floatX),
				   borrow=True)
b1 = theano.shared(name='bias1',
				   value=b1_values.astype(theano.config.floatX),
				   borrow=True)
# Weights and bias for layer 2 - fully connected layer
W2 = theano.shared(name='Weights2',
				   value=w2_values.astype(theano.config.floatX),
				   borrow=True)
b2 = theano.shared(name='bias2',
				   value=b2_values.astype(theano.config.floatX),
				   borrow=True)
W3 = theano.shared(name='Weights3',
				   value=w3_values.astype(theano.config.floatX),
				   borrow=True)
b3 = theano.shared(name='bias3',
				   value=b3_values.astype(theano.config.floatX),
				   borrow=True)

params = [W3, b3, W2, b2, W1, b1, W0, b0]

conv_out0 = conv2d(inputs, W0)
conv_out1 = conv2d(inputs, W1)

output0 = elu(conv_out0 + b0.dimshuffle('x', 0, 'x', 'x'))
output1 = elu(conv_out1 + b1.dimshuffle('x', 0, 'x', 'x'))

# Max Pooling
maxpool_shape = (2, 2)
pool_out = pool.pool_2d(inputs, maxpool_shape, ignore_border=True)

f0 = theano.function(inputs=[inputs], outputs=conv_out0,
					 allow_input_downcast=True)

f1 = theano.function(inputs=[inputs], outputs=conv_out1,
					 allow_input_downcast=True)

f_pool = theano.function(inputs=[inputs], outputs=pool_out,
						 allow_input_downcast=True)

########################################################################
## Plotting                                                           ##
########################################################################

# img = mpimg.imread('3wolfmoon.jpg')
# img = img/256

# img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
# filtered_img = f(img_)

# fig = plt.figure(figsize=(30, 10))

# ax1 = fig.add_subplot(131)
# ax1.set_title('Original')
# plt.imshow(img)

# plt.gray()
# ax2 = fig.add_subplot(132)
# ax2.set_title('Filter 0')
# plt.imshow(filtered_img[0, 0, :, :])

# ax3 = fig.add_subplot(133)
# ax3.set_title('Filter 1')
# plt.imshow(filtered_img[0, 1, :, :])

# plt.show()

########################################################################
########################################################################

test = rng.random_integers(low=0, high=9, size=(12, 12))
# print(test)
test = test.reshape(1, 1, 12, 12)
conv0 = f0(test)
# print(conv)
pooled0 = f_pool(conv0)
conv1 = f1(pooled0)
# print('--------------------------')
conv1 = conv1.flatten(2)
fc1 = elu(T.dot(W2, conv1.T) + b2)
fc2 = elu(T.dot(W3, fc1.T) + b3)
cost = T.nnet.nnet.categorical_crossentropy()

# test2 = rng.random_integers(low=1, high=8, size=(3, 3))
# group = np.array([test, test2])
# group = group.reshape(1, 2, 3, 3)

# print(np.mean(group, axis=(2, 3)))

# A = T.tensor4(name='A', dtype=theano.config.floatX)
# g = theano.shared(name='gamma',
# 				  value=rng.uniform(low=-1, high=1, size=(2)))
# b = theano.shared(name='bias',
# 				  value=np.zeros(2))
# m = theano.shared(name='mean',
# 				  value=np.mean(group, axis=(2, 3)))
# s = theano.shared(name='st_deviation',
# 				  value=np.std(group, axis=(2, 3)))

# print(m.get_value().shape)

# g = g.dimshuffle('x', 0, 'x', 'x')
# b = b.dimshuffle('x', 0, 'x', 'x')
# m = m.dimshuffle(0, 1, 'x', 'x')
# s = s.dimshuffle(0, 1, 'x', 'x')

# batch_params = [g, b]

# f2 = theano.function(inputs=[A], 
# 					 outputs=T.nnet.bn.batch_normalization(A, g, b, m, s),
# 					 allow_input_downcast=True)

# print(f2(group))

########################################################################

# location = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning\\Kaggle'+\
# 		   '\\MNIST\\train.csv'

# data_all = pd.read_csv(location)
# target = data_all[[0]].values.ravel()
# digits = data_all.iloc[:, 1:].astype(np.uint8)
# digits = np.array(digits).reshape((-1, 1, 28, 28)).astype(np.uint8)
# test = data_all.iloc[0, 1:].astype(np.uint8)

# test = test.values.reshape(1, 1, 28, 28)
