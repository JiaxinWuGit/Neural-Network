import numpy as np 


def forward(A, W, b):
	"""
	feed forward z = w * a + b
	Argument:
		A -- input from last layer
		W, b -- parameters
	Returns:
		Z = W * A + b
	"""

	Z = np.dot(W, A) + b
    
	assert(Z.shape == (W.shape[0], A.shape[1]))
	
	return (Z)


def sigmoid(Z):
	"""
	sigmoid function
	Argument:
		Z -- a vector in array form
	Returns:
		A = sigmoid(Z)
	"""

	A = 1/(1+np.exp(-Z))
	return (A)


def relu(Z):
	"""
	relu function
	Argument:
		Z -- a vector in array form
	Returns:
		A = max(0, Z)
	"""

	A = np.maximum(0, Z)
	return (A)



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







#test
parameters_test = {}
parameters_test["W1"] = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.2]])
parameters_test["b1"] = np.array([[1.0], [-2.0], [1.0]])
parameters_test["W2"] = np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]])
parameters_test["b2"] = np.array([[1.0],[2.0]])

A_set_test = {}
A_set_test["A0"] = np.array([[1.0, 2.0], [2.0, 3.0]])

layer_dims_test = [2, 3, 2]



A1_test, df_dZ1 = activation_forward(A_prev = A_set_test["A0"], W = parameters_test["W1"], b = parameters_test["b1"], f = "tanh")
print("A: ", A1_test)
print("df_dZ:", df_dZ1)



