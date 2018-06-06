# Load MNIST data

import numpy as np 
import gzip
import pickle

def load_mnist():
	"""
	Return 
		train_set: data for training neural network. It has 50,000 images and corresponding numbers. 
		valid_set: data for experimenting hyper parameters. It has 10,000 images and corresponding numbers.
		test_set: data for testing neural network. It has 10,000 images and corresponding numbers.
	Every image is 28*28 = 784 pixels.
	Each pixel is a value in (0,255), 0 -> black, 255 -> white.
	All data are in tuple.

	train_set=(array[50,000,784],array[50,000]),similar to valid_set and test_set

	Data source: http://www.deeplearning.net/tutorial/gettingstarted.html

	"""

	ti = gzip.open("C:\\Python\\Numbers recognition\\MNIST data\\mnist.pkl.gz","rb")
	train_set, valid_set, test_set = pickle.load(ti, encoding = "latin1")
	ti.close()
	return (train_set, valid_set, test_set)

train_set, valid_set, test_set = load_mnist()


print(train_set[0].shape)
print(train_set[1].shape)
print(valid_set[0].shape)
print(valid_set[1].shape)
print(test_set[0].shape)
print(test_set[1].shape)

