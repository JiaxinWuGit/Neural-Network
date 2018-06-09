# Load MNIST data

import numpy as np 
import gzip
import pickle
import random

def load_mnist():
	"""
	Load the MNIST data.
	Returns:
		train_set -- data for training neural network. It has 50,000 images and corresponding numbers. 
		valid_set -- data for experimenting hyper parameters. It has 10,000 images and corresponding numbers.
		test_set -- data for testing neural network. It has 10,000 images and corresponding numbers.
	Every image is 28*28 = 784 pixels.
	Each pixel is a value in (0, 255), 0 -> black, 255 -> white.
	All data are in tuple.

	train_set=(array[50,000, 784], array[50,000]),similar to valid_set and test_set

	Data source: http://www.deeplearning.net/tutorial/gettingstarted.html

	"""

	ti = gzip.open("C:\\Python\\Numbers recognition\\MNIST data\\mnist.pkl.gz","rb")
	train_set, valid_set, test_set = pickle.load(ti, encoding = "latin1")
	ti.close()
	return (train_set, valid_set, test_set)

train_set, valid_set, test_set = load_mnist()



def construct_output(data_output):
	"""
	Reconstruct the expected output: from numbers 0 - 9 to an array Y with each number in the array indicating the probability
	of a certain number.
	i.e. [2] -> np.array[0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]
	CAUTION: data_output should not be a big array, since this function contain a for loop.
	Argument:
		data_output -- the expected output from the original data set.
	Returns:
		Y -- the reconstructed output in an array Y. Y.shape = (number of output, 10)
	"""

	size = data_output.shape[0] 
	# Size of the data_output.

	Y = np.zeros((size, 10))

	for l in range(0, size):
		Y[l, data_output[l]] = 1.0

	Y = Y.T

	assert (Y.shape == (10, size))

	return (Y)



def mini_batch(data, mini_batch_size):
	"""
	Randomly choose n examples from data. n = mini_batch_size.
	Argument:
		data -- data for NN.
		mini_batch_size -- number of examples in the mini batch.
	Returns:
		mini_A0 -- a set of input data in the mini batch.
		mini_Y -- a set of output data in the mini batch.
	"""
	data_size = data[1].shape[0]
	random_label = random.sample(range(data_size), mini_batch_size)
	#The labels for the random picked examples

	mini_A0 = data[0][random_label].T
	mini_Y = data[1][random_label]

	assert (mini_A0.shape == (784, mini_batch_size))
	assert (mini_Y.shape == (mini_batch_size, ))

	return (mini_A0, mini_Y)




#test 

#print(train_set[0].shape)
#print(train_set[1].shape[0])
print(valid_set[0].shape)
print(valid_set[1].shape)
#print(test_set[0].shape)
#print(test_set[1].shape)

#print(random.sample(range(10), 3))

mini_A0_t, mini_Y_t = mini_batch(data = test_set, mini_batch_size = 2)
#print(mini_A0_t)
#print(mini_A0_t[1])
#print(mini_A0_t.shape)
#print(mini_Y_t.shape)
#print(mini_Y_t)

#Y_new_t = construct_output(data_output = mini_Y_t)
#print(Y_new_t.shape)


#print(test_set[1][4])

#ll = np.array([[1], [2], [3], [4], [5]])
#ran = random.sample(range(5), 3)
#print(ran)
#print(ll[ran])

#for i in range(5, 0, -1):
#	print(i)