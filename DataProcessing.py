# DataAugment.py

# Libraries
# Third-Party Libraries
import pandas as pd
import numpy as np

def augment(data: np.array, size: int) -> np.array:
	import scipy.ndimage

	augmented_data = data[np.random.randint(data.shape[0], size=size), :]

	for i, aug in enumerate(augmented_data):
		random_values = np.random.randint(low=0, high=2, size=3)

		while sum(random_values) == 0:
			random_values = np.random.randint(low=0, high=2, size=3)

		aug_label = np.asarray(aug[0])
		aug_values = aug[1:].reshape(28, 28)	

		# Rotate
		if random_values[0] == 1:
			random_rotation = np.random.randint(low=-15.0, high=15.0)
			# Guarantees that the number will rotate
			while random_rotation == 0:
				random_rotation = np.random.randint(low=-15, high=15)

			aug_values = scipy.ndimage.rotate(aug_values, random_rotation, 
											  reshape=False)	

		# Shift
		# if random_values[1] == 1:
			random_shift = np.random.randint(low=-3, high=3, size=2)
			# Guarantees that there will be at least one shift
			while sum(random_shift) == 0:
				random_shift = np.random.randint(low=-3, high=3, size=2)

			aug_values = scipy.ndimage.shift(aug_values, random_shift)

		# Zoom
		if random_values[2] == 1:
			zoom_values = {0:1, 1:1.3, 2:1.5, 3:1.8, 4:2.0}
			rezoom_values = {0:0, 1:4, 2:7, 3:11, 4:14}

			random_zoom = np.random.randint(low=0, high=5, size=2)
			# Guarantees that there will be at least one zoom
			while sum(random_zoom) == 0:
				random_zoom = np.random.randint(low=0, high=5, size=2)

			zoom = (zoom_values[random_zoom[0]], zoom_values[random_zoom[1]])
			aug_values = scipy.ndimage.interpolation.zoom(aug_values, zoom)

			ax0, ax1 = aug_values.shape
			ax0_i, ax1_i = (ax0-28)//2, (ax1-28)//2
			ax0_f = -ax0_i if ax0_i != 0 else 28
			ax1_f = -ax1_i if ax1_i != 0 else 28
			aug_values = aug_values[ax0_i: ax0_f, ax1_i: ax1_f]

		aug_values.reshape(1, 784)
		aug = np.append(aug_label, aug_values)
		augmented_data[i] = aug

	return augmented_data

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
