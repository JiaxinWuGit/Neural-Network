import numpy as np 
import gzip
import pickle
import random
import matplotlib.pyplot as plt

# Input parameters for the neural network.
mini_batch_size = 350
layer_dims = [784, 70, 60, 10]   # Dimension of every layer, including the input layer.
L = len(layer_dims)
steps = 3600   # Number of times the NN update the parameters
activation = np.repeat("tanh", L - 2)  # Activation functions' names.
activation = np.append(activation, "sigmoid")
cost_type = "cross_entropy" # Type of the cost function. It could be "cross_entropy" or "quadratic".
learning_rate = 0.7

regularization = "None" # Method for regularization which shrink Wk's. It can be "None" or "L1" or "L2".
regular_parameter = 0.05 # Parameter for regularization.

del_step = 100 # Every del_step number of steps, calculate and keep the cost.



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



data_set = train_set # The set of data which will be used in training the NN.
data_set_test = test_set  # The set of date which will be used in calculating accuracy.




def initialize_parameters(layer_dims):
	"""
	Initialize parameters for neural network.

	Argument:
		layer_dims -- an array that contains dimension of the neural network. 
		e.g. layer_dims = [3,4] would create a NN with 3 perceptrons in the input layer, 4 in the 1st layer.

	Returns:
		parameters -- dictionary of "W1", "b1", ..., "WL", "bL"
				Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    			bl -- bias vector of shape (layer_dims[l], 1)
	
	"""

	#np.random.seed(3)
	#set a particular seed if desired.

	parameters = {}
	L = len(layer_dims)            # number of layers in the network, including the input layer

	for l in range(1, L):

		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		# *0.01 to make sure that W's are small and easy to update.

		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
	return parameters


parameters = initialize_parameters(layer_dims = layer_dims)


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




def forward(A_prev, W, b):
	"""
	feed forward z = w * a + b
	Argument:
		A_prev -- input from last layer
		W, b -- parameters
	Returns:
		Z = W * A_prev + b
	"""

	Z = np.dot(W, A_prev) + b
    
	assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
	return (Z)


def sigmoid(Z):
	"""
	sigmoid function
	Argument:
		Z -- a vector in array form
	Returns:
		A_sig = sigmoid(Z)
	"""

	A_sig = 1/(1+np.exp(-Z))
	return (A_sig)


def relu(Z):
	"""
	relu function
	Argument:
		Z -- a vector in array form
	Returns:
		A_relu = max(0, Z)
	"""

	A_relu = np.maximum(0, Z)
	return (A_relu)



def activation_forward(A_prev, W, b, f):
	"""
	feed forward A = f(W * A_prev + b)
	Argument:
		A_prev -- A from last layer
		f -- activation function f
			f = "sigmoid" or "relu" or "tanh"
	Returns:
		A -- A for the current layer
		df_dZ -- df/dZ
	"""

	Z = forward(A_prev, W, b)

	if f == "sigmoid":
		A = sigmoid(Z)
		df_dZ = np.multiply(sigmoid(Z), 1 - sigmoid(Z))

	elif f == "relu":
		A = relu(Z)
		df_dZ = A
		df_dZ[df_dZ > 0] = 1
		
	elif f == "tanh":
		A = np.tanh(Z)
		df_dZ = 4/(np.exp(Z) + np.exp(-Z))**2

	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	
	return (A, df_dZ)


def back_propagation(A_set, Y, cost_type, df, parameters, layer_dims, learning_rate, regularization, regular_parameter):
	"""
	Update parameters["W1", "b1", ..., "WL", "bL"] according to back propagation.

	Argument:
		A_set -- a dictionary of array A. Ak.shape = (number of perceptron in layer k, number of examples).
		Y -- the expected value of the output in array. Y.shape = (number of output, number of examples).
		cost_type -- the type of the cost function.
		df -- a dictionary for "df_dZ1", ..., "df_dZL"
		parameters -- dictionary of "W1", "b1", ..., "WL", "bL":
			Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    		bl -- bias vector of shape (layer_dims[l], 1)
			layer_dims -- an array that contains dimension of the neural network. 
				e.g. layer_dims = [3,4] would create a NN with 3 perceptrons in the input layer, 4 in the 1st layer.
			learning_rate -- the step size for parameters' update.
		regularization -- regularizations to keep Wk's small. It can be "None" or "L1" or "L2"
			See "L1" or "L2" regularization in http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization.
		regular_parameter -- control the regularization strength.

	Returns:
		parameters_new -- updated parameters according to back propagation.
	"""

	m = Y.shape[1]
	L = len(layer_dims)            # number of layers in the network, including the input layer

	AL = A_set["A" + str(L - 1)]

	if cost_type == "quadratic":
		dAL = np.subtract(AL, Y)/m 
		# dAL = dC/dAL, dAL.shape = (1, number of example).
	elif cost_type == "cross_entropy":
		dAL = - np.divide(np.subtract(Y, AL), np.multiply(1 - AL, AL))/m

	

	delta = np.multiply(dAL, df["df_dZ" + str(L - 1)])
	assert(delta.shape == (layer_dims[L - 1], m))

	parameters_new = {}	   #new parameters

	parameters_new["b" + str(L - 1)] = parameters["b" + str(L - 1)] - learning_rate * np.sum(delta, axis = 1, keepdims = True)

	if regularization == "None":
		parameters_new["W" + str(L -1)] = parameters["W" + str(L - 1)] - learning_rate * np.dot(delta, A_set["A" + str(L - 2)].T)
			
	elif regularization == "L1": 
		parameters_new["W" + str(L - 1)] = parameters["W" + str(L - 1)] - learning_rate * regular_parameter * np.sign(parameters["W" + str(L - 1)])/m - learning_rate * np.dot(delta, A_set["A" + str(L - 2)].T)
		
	elif regularization == "L2":
		parameters_new["W" + str(L - 1)] = parameters["W" + str(L - 1)] * (1 - learning_rate * regular_parameter / m) - learning_rate * np.dot(delta, A_set["A" + str(L - 2)].T)

	assert(parameters_new["W" + str(L - 1)].shape == (layer_dims[L - 1], layer_dims[L - 2]))
	assert(parameters_new["b" + str(L - 1)].shape == (layer_dims[L - 1], 1))

	for l in range (L - 2, 0, -1):
		delta = np.multiply(np.dot(np.transpose(parameters["W" + str (l + 1)]), delta), df["df_dZ" + str(l)])
		assert(delta.shape == (layer_dims[l], m))

		parameters_new["b" + str(l)] = parameters["b" + str(l)] - learning_rate * np.sum(delta, axis = 1, keepdims = True)

		if regularization == "None":
			parameters_new["W" + str(l)] = parameters["W" + str(l)] - learning_rate * np.dot(delta, A_set["A" + str(l - 1)].T)
			
		elif regularization == "L1": 
			parameters_new["W" + str(l)] = parameters["W" + str(l)] - learning_rate * regular_parameter * np.sign(parameters["W" + str(l)])/m - learning_rate * np.dot(delta, A_set["A" + str(l - 1)].T)
		
		elif regularization == "L2":
			parameters_new["W" + str(l)] = parameters["W" + str(l)] * (1 - learning_rate * regular_parameter / m) - learning_rate * np.dot(delta, A_set["A" + str(l - 1)].T)

		assert(parameters_new["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
		assert(parameters_new["b" + str(l)].shape == (layer_dims[l], 1))

	return (parameters_new)




def cost(AL, Y, cost_type):
	"""
	Compute the cost function C.
	Argument:
		AL -- array A from the very last layer, or say the output of the NN. AL.shape = (number of output, number of examples)
		Y -- the expected value of the output in array. Y.shape = (number of output, number of examples)
		cost_type -- the type of the cost function.
			It could be "quadratic": C = sum(|AL - Y|^2)/(2m)
			or "cross_entropy": C = -sum(Y log AL + (1 - Y) log (1-AL))/m
			m -- the total number of examples input in calculating C.
	Returns:
		C -- cost depending on the cost_type
	"""

	m = Y.shape[1]

	if cost_type == "quadratic":
		C = np.sum(np.power(np.subtract(AL, Y), 2), axis = None, keepdims = True)
		C = C / (2 * m)

	elif cost_type == "cross_entropy":
		C = - np.sum(np.multiply(Y, np.log(AL + 10**-7)) + np.multiply(1 - Y, np.log(1 - AL + 10**-7)), axis = None, keepdims = True)
		C = C / m

	C = np.squeeze(C)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(C.shape == ())

	return (C)




A_set = {}  # A dictionary which stores Ak's.
df = {} # A dictionary which stores df/dZk.
C_cost = np.zeros((2, steps // del_step))


for i in range(0, steps):
	mini_A0, mini_Y = mini_batch(data = data_set, mini_batch_size = mini_batch_size)
	mini_Y = construct_output(mini_Y)
	A_set["A0"] = mini_A0

	for j in range (1, L):
		A_set["A" + str(j)], df["df_dZ" + str(j)] = activation_forward(A_prev = A_set["A" + str(j - 1)], W = parameters["W" + str(j)] , b = parameters["b" + str(j)] , f = activation[j - 1] )
	
	if i % del_step == 0:
		nn = i // del_step 
		C_cost[0, nn] = (nn + 1) * del_step 
		C_cost[1, nn] = cost(AL = A_set["A" + str(L - 1)], Y = mini_Y, cost_type = cost_type)
	parameters = back_propagation(A_set = A_set, Y = mini_Y, cost_type = cost_type, df = df, parameters = parameters, layer_dims = layer_dims, learning_rate = learning_rate, regularization = regularization, regular_parameter = regular_parameter)




 
A_final = data_set_test[0].T # Store the final run of A

for k in range(1, L):
	A_final, df_final = activation_forward(A_prev = A_final, W = parameters["W" + str(k)] , b = parameters["b" + str(k)] , f = activation[k - 1] )




def accuracy(AL, Y):
	"""
	Calculate the accuracy of the trained NN.

	Argument:
		AL -- the outcome from the NN.
		Y -- the expected outcome, from the MNIST data (no need to put in construct_output).

	Returns:
		correct_rate -- percentage of correct results.
	"""

	size = AL.shape[1]

	outcome = np.argmax(AL, axis = 0)  # rectructure to know the outcome number based on the probability
	compare = outcome - Y
	wrong_example_number = np.count_nonzero(compare)
	correct_rate = 1.0 - wrong_example_number / size

	return (correct_rate)

data_size = data_set_test[1].shape[0]
Y_final = np.reshape(data_set_test[1], (1, data_size))
correct_rate = accuracy(AL = A_final, Y = Y_final)

print("accuracy = ", correct_rate * 100, "%")



g, ax = plt.subplots(1)
ax.plot(C_cost[0], C_cost[1])
ax.set_ylim(ymin = 0)
plt.ylabel('cost', fontsize = 18)
plt.xlabel('number of updates', fontsize = 18)
plt.show(g)




def export_parameters(parameters, name_of_file):
	"""
	Export parameters["W1","b1", ..., "WL", "bL"] in the NN.
	
	Argument:
		parameters -- dictionary of "W1", "b1", ..., "WL", "bL"
				Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    			bl -- bias vector of shape (layer_dims[l], 1)
    	name_of_file -- name of the file to be created and saved in.
    		e.g. name_of_file = "save_parameters"
	"""
	f1 = open("name_of_file" + ".pkl", "w+")
	f1.close()

	with open(name_of_file + ".pkl", "wb") as f:
		pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)


export_parameters(parameters = parameters, name_of_file = "parameters_nmist_2")
