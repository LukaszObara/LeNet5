import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage

location = 'C:\\...\\MNIST\\train.csv'

data_all = pd.read_csv(location)
# digits = data_all.iloc[:, 1:].astype(np.uint8)
# digits = np.array(digits).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = data_all.iloc[:, :].astype(np.uint8)
test = test.as_matrix()

augmented_data = test[np.random.randint(test.shape[0], size=13040), :]

for i, aug in enumerate(augmented_data):
	random_values = np.random.randint(low=0, high=2, size=3)
	
	# Guarantees that there will be at least one transformation
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
	if random_values[1] == 1:
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


test = np.append(augmented_data, test, axis=0)
np.random.shuffle(test)

location1 = 'C:\\...\\MNIST\\augmented_train.npy'

np.save(location1, test)
