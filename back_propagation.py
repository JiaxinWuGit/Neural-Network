import numpy as np 

def back_propagation(AL, Y, cost_type, df, parameters, layer_dims):
	"""
	Compute delta["del1", "del2", ..., "delL"].
	Argument:
		AL -- array A from the very last layer, or say the output of the NN. AL.shape = (number of output, number of examples)
		Y -- the expected value of the output in array. Y.shape = (number of output, number of examples)
		cost_type -- the type of the cost function.
		df -- a dictionary for "df_dZ1", ..., "df_dZL"
		parameters -- dictionary of "W1", "b1", ..., "WL", "bL"
			Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    		bl -- bias vector of shape (layer_dims[l], 1)
		layer_dims -- an array that contains dimension of the neural network. 
		e.g. layer_dims = [3,4] would create a NN with 3 perceptrons in the input layer, 4 in the 1st layer.
	Returns:
		delta["del1", "del2", ..., "delL"] -- dictionary of del's in arrays of input x. delk.shape = (layer_dims[l], number of examples)
			delL = np.multiple(dC/dA, df(ZL)/dZL)
			delk = np.dot(Wk, np.multiple(delk+1, df(Zk)/dZk))
	"""

	m = Y.shape[1]

	if cost_type == "quadratic":
		dAL = np.subtract(AL, Y)/m 
		# dAL = dC/dAL, dAL.shape = (1, number of example).
	elif cost_type == "cross_entropy":
		dAL = - np.divide(np.subtract(Y, AL), np.multiply(1 - AL, AL))/m

	delta = {}
	# delta is the dictionary to store del1, ..., delL.

	L = len(layer_dims)            # number of layers in the network

	delta["del" + str(L)] = np.multiply(dAL, df["df_dZ" + str(L)])
	assert(delta["del" + str(L)].shape == (layer_dims[L-1], m))

	for l in range (1,L):
		delta["del" + str(l)] = np.dot(parameters["W" + str (l + l)].T, np.multiply(delta["del" + str(l + 1)], df["df_dZ" + str(l)]))
		assert(delta["del" + str(l)].shape == (layer_dims[l-1], m))

	return (delta)



#test
#AL, Y, cost_type, df, parameters, layer_dims

AL_test = np.array([[0.7806], [0.5118]])
Y_test = np.array([[1], [0]])

#cost_type_t = "quadratic"
cost_type_t = "cross_entropy"

df_t = {}
df_t["df_dZ1"] = np.array([[0.0451], [0.1966]])
df_t["df_dZ2"] = np.array([[0.1712], [0.2499]])

parameters_test = {}
parameters_test["W1"] = np.array([[0, 1], [1, 0]])
parameters_test["b1"] = np.array([[1], [-2]])
parameters_test["W2"] = np.array([[0, 1], [-1, 0]])
parameters_test["b2"] = np.array([[1], [1]])


layer_dims_test = [2,2]

delta_test = back_propagation(AL = AL_test, Y = Y_test, cost_type = cost_type_t, df = df_t, parameters = parameters_test, layer_dims = layer_dims_test)
print(delta_test)