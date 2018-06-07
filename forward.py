import numpy as np 


def forward(A, W, b):
	"""
	feed forward z = w * a + b
	Argument:
		A -- input from last layer
		W, b -- parameters
	Returns:
		Z = W * A + b
		cache = (A, W, b)
	"""

	Z = np.dot(W, A) + b
    
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b, Z)
    
	return (Z, cache)


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
		cache = (linear_cache, activation_cache)
			linear_cache = (A_prev, W, b)
			activation_cache = A
		df_dZ -- df/dZ
	"""

	Z, linear_cache = forward(A_prev, W, b)

	if f == "sigmoid":
		A = sigmoid(Z)
		activation_cache = sigmoid(Z)
		df_dZ = np.multiply(sigmoid(Z), 1 - sigmoid(Z))

	elif f == "relu":
		A = relu(Z)
		activation_cache = relu(Z)
		df_dZ = A
		df_dZ[df_dZ > 0] = 1
		
	elif f == "tanh":
		A = np.tanh(Z)
		activation_cache = np.tanh(Z)
		df_dZ = 4/(np.exp(Z) + np.exp(-Z))**2

	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return (A, df_dZ, cache)







#test
W_test = np.array([[0, 1], [1, 0]])
b_test = np.array([1, 1])
A_test = np.array([[1, 0], [0, 1]])
Z_test, linear_cache_test = forward(A = A_test, W = W_test, b = b_test)

print("Z: ", Z_test)
#print("linear_cache: ", linear_cache_test)

#A_test1, df_dZ1, cache_test = activation_forward(A_prev = A_test, W = W_test, b = b_test, f = "sigmoid")
#print("A: ", A_test1)
#print("cache: ", cache_test)
#print("df_dZ:", df_dZ1)

#A_test2, df_dZ2, cache_test = activation_forward(A_prev = A_test, W = W_test, b = b_test, f = "relu")
#print("A: ", A_test2)
#print("cache: ", cache_test)
#print("df_dZ:", df_dZ2)


A_test3, df_dZ3, cache_test = activation_forward(A_prev = A_test, W = W_test, b = b_test, f = "tanh")
print("A: ", A_test3)
print("cache: ", cache_test)
print("df_dZ:", df_dZ3)

print("Z:", cache_test[0][3])


