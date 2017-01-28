# GradientDescent.py

# Libraries 
# Third-Party Libraries
import numpy as np
import theano
import theano.tensor as T


def sgd(l_rate, parameters, grads): 
	""" 
	Stochastic Gradient Descent.

	Parameters
	----------
	:type lr: theano.tensor.scalar
	:param lr: Initial learning rate
	
	:type parameters: theano.shared
	:params parameters: Model parameters to update
	
	:type grads: Theano variable
	:params grads: Gradients of cost w.r.t to parameters
	"""

	updates = []
	for param, grad in zip(parameters, grads):
		updates.append((param, param - l_rate * grad))

	return updates

def momentum(l_rate, parameters, grads, momentum=0.9):
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

	def update_rule(param, velocity, df):
		v_next = momentum * velocity - l_rate * df
		updates = (param, param+v_next), (velocity, v_next)

		return updates

	assert momentum <=1 and momentum >= 0
	
	velocities = [theano.shared(name='v_{}'.format(param),
								value=param.get_value() * 0., 
								broadcastable=param.broadcastable) 
				  for param in parameters]

	updates = []
	for p, v, g in zip(parameters, velocities, grads):
		param_updates, vel_updates = update_rule(p, v, g)
		updates.append(param_updates)
		updates.append(vel_updates)

	return updates

def nag(l_rate, momentum=0.9, anneal_rate=1e-2, parameters=None, grads=None):
	"""
	Momentum update

	Parameters
	----------
	:type l_rate: theano.tensor.scalar
	:param l_rate: Initial learning rate

	:type momentum: float32
	:params momentum: 

	:type anneal_rate: float32
	:param anneal_rate: 
	
	:type parameters: theano.shared
	:params parameters: Model parameters to update

	:type grads: Theano variable
	:params grads: Gradients of cost w.r.t to parameters

	:type noise: bool
	:params noise: 
	"""
	t = theano.shared(name='time_step', value=np.int16(0))

	def update_rule(param, velocity, df):
		v_prev = velocity
		v = momentum * velocity - l_rate * T.exp(-anneal_rate*t) * df
		x = momentum * v_prev + (1-momentum) * v
		updates = (param, param+x), (velocity, v)

		return updates
	
	velocities = [theano.shared(name='v_{}'.format(param),
								value=param.get_value() * 0., 
								broadcastable=param.broadcastable) 
				  for param in parameters]

	updates = []
	for p, v, g in zip(parameters, velocities, grads):
		param_updates, vel_updates = update_rule(p, v, g)
		updates.append(param_updates)
		updates.append(vel_updates)
	updates.append((t, t+1))

	return updates

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
	
	caches = [theano.shared(name='c_{}'.format(param),
							value=param.get_value() * 0., 
							broadcastable=param.broadcastable) 
			  for param in parameters]

	updates = []
	for p, c, g in zip(parameters, caches, grads):
		param_updates, cache_updates = update_rule(p, c, g)
		updates.append(param_updates)
		updates.append(cache_updates)

	return updates

def adagrad(l_rate, epsilon=1e-6, parameters=None, grads=None):
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

	def update_rule(param, cache, df):
		cache_val = df**2
		x = l_rate * df / (T.sqrt(cache_val) + eps)
		updates = (param, param-x), (cache, cache+cache_val)

		return updates
	
	caches = [theano.shared(name='c_{}'.format(param),
							value=param.get_value() * 0., 
							broadcastable=param.broadcastable) 
			  for param in parameters]

	updates = []
	for p, c, g in zip(parameters, caches, grads):
		param_updates, cache_updates = update_rule(p, c, g)
		updates.append(param_updates)
		updates.append(cache_updates)

	return updates


def adam(l_rate, beta1=0.9, beta2=0.999, epsilon=1e-6, parameters=None, 
		 grads=None):

	one = T.constant(1.0)
	t = theano.shared(name='iteration', value=np.float32(1.0))

	def update_rule(param, moment, velocity, df):
		m_t = beta1 * moment + (one-beta1) * df
		v_t = beta2 * velocity + (one-beta2) * df**2
		m_hat = m_t/(one-beta1**(t))
		v_hat = v_t/(one-beta2**(t))
		x = (l_rate * m_hat / (T.sqrt(v_hat) + epsilon))
		updates = (param, param-x), (moment, m_t), (velocity, v_t)

		return updates
	
	moments = [theano.shared(name='m_{}'.format(param),
							 value=param.get_value() * 0., 
							 broadcastable=param.broadcastable) 
			   for param in parameters]

	velocities = [theano.shared(name='v_{}'.format(param),
								value=param.get_value() * 0., 
								broadcastable=param.broadcastable) 
				  for param in parameters]

	updates = []
	for p, m, v, g in zip(params, moments, velocities, grads):
		p_update, m_update, v_update = update_rule(p, m, v, g)
		updates.append(p_update)
		updates.append(m_update)
		updates.append(v_update)
	updates.append((t, t+1))

	return updates

def adamax(l_rate, beta1=0.9, beta2=0.999, epsilon=1e-6, parameters=None, 
		   grads=None):

	one = T.constant(1.0)
	t = theano.shared(name='iteration', value=np.float32(1.0))

	def update_rule(param, moment, u, df):
		m_t = beta1 * moment + (one-beta1) * df
		u_t = T.maximum(beta2*u, T.abs_(df))
		x = (lr/(1-beta1**t)) * (m_t/u_t) 
		updates = (param, param-x), (moment, m_t), (u, u_t)
		return updates
	
	moments = [theano.shared(name='m_{}'.format(param),
							 value=param.get_value() * 0., 
							 broadcastable=param.broadcastable) 
			   for param in parameters]

	upd = [theano.shared(name='u_{}'.format(param),
						 value=param.get_value() * 0., 
						 broadcastable=param.broadcastable) 
				for param in parameters]

	updates = []
	for p, m, u, g in zip(params, moments, upd, grads):
		p_update, m_update, u_update = update_rule(p, m, u, g)
		updates.append(p_update)
		updates.append(m_update)
		updates.append(u_update)
	updates.append((t, t+1))

	return updates
