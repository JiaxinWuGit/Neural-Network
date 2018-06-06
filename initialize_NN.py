import numpy as np 

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
	L = len(layer_dims)            # number of layers in the network

	for l in range(1, L):

		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		# *0.01 to make sure that W's are small and easy to update.

		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
	return parameters


layer_dims = [3,2,1]
parameters = initialize_parameters(layer_dims)
print(parameters)
print(len(layer_dims))
