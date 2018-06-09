import numpy as np 
import pickle

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



def load_parameters(name_of_file):
	"""
	Load parameters from file.

	Argument:
		name_of_file -- name of the file the parameters are saved in.

	Returns:
		parameters -- dictionary of "W1", "b1", ..., "WL", "bL"
				Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    			bl -- bias vector of shape (layer_dims[l], 1)
	"""

	with open(name_of_file + ".pkl", "rb") as f:
		return pickle.load(f)





#test

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


layer_dims = [3,2]
parameters = initialize_parameters(layer_dims = layer_dims)
#print(parameters)
#export_parameters(parameters = parameters, name_of_file = "C:\\Python\\Numbers recognition\\MNIST data\\trial_export")

loaded_para = load_parameters(name_of_file = "C:\\Python\\Numbers recognition\\MNIST data\\trial_export")
print(loaded_para)