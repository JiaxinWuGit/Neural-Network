import numpy as np 

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


#test
AL_test = np.array([[0.0, 0.7, 0.3],[0.1, 0.3, 0.6]])
Y_test = np.array([[0, 1, 0],[0, 0, 1]])

cost_test1 = cost(AL = AL_test, Y = Y_test, cost_type = "quadratic")
cost_test2 = cost(AL = AL_test, Y = Y_test, cost_type = "cross_entropy")

print(cost_test1)
print(cost_test2)


