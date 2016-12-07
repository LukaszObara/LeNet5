import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage

location = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning\\Kaggle'+\
		   '\\MNIST\\train.csv'

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

location1 = 'C:\\Users\\lukas\\OneDrive\\Documents\\Machine Learning\\Kaggle'+\
			'\\MNIST\\augmented_train.npy'

np.save(location1, test)

# label_temp = np.asarray(test1[0])
# number_temp = test1[1:]
# number_temp = number_temp.reshape(28, 28)
# print(test.shape)
# test_rot = scipy.ndimage.rotate(number_temp, -15, reshape=False)
# test_rot = test_rot.reshape(784,)
# data_temp = np.append(label_temp, test_rot)
# data_temp = data_temp.reshape(1, -1)
# print(data_temp.shape)
# test = np.append(test, data_temp, axis=0)
# print(test_rot.shape)
# test_shift = scipy.ndimage.shift(test, (0, 5)) # Shifts (y, x) positive is down in y direction
# print(test_shift.shape)
# test_stretch = scipy.ndimage.interpolation.zoom(test, (1, 1.3)) # needs to be a number between 1 and 2, >2 will not fit properly
# print(test_stretch.shape)
# test_stretch = test_stretch[:, 4: -4]
# print(test_stretch.shape)

# # plt.figure(figsize=(10,10))
# # plt.imshow(test, cmap=cm.binary)
# # plt.show()

# fig, axarr = plt.subplots(nrows=1, ncols=2) 
# axarr[0].imshow(number_temp, cmap=cm.binary)
# axarr[0].set_title('Original')
# axarr[1].imshow(test_rot, cmap=cm.binary)
# axarr[1].set_title('Rotation')
# axarr[2].imshow(test_shift, cmap=cm.binary)
# axarr[2].set_title('Shift')
# axarr[3].imshow(test_stretch, cmap=cm.binary)
# axarr[3].set_title('Strech')
# plt.show()
# for i in range(10):
# 	axarr[i].imshow(digits[i][0], cmap=cm.binary)
# 	axarr[i].axis('off')

# for i in range(10):
# 	plt.imshow(digits[i][0], cmap=cm.binary)
# plt.show()