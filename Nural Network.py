import numpy as np

def sigimod(x):
	return 1/(1+np.exp(-x))

inputs = np.array([
            [0,0,1],
            [1,0,0],
            [1,0,1],
            [1,1,1],
            [1,1,1],
            [1,1,0],
            [0,0,1]
                  ]
                 )

outputs = np.array([
            [0],
            [1],
            [1],
            [0],
            [1],
            [0],
            [1]
                   ])

np.random.seed(5)

# randomly initialize weights
w0 = 2*np.random.random((3,7))
w1 = 2*np.random.random((7,1))

for j in range(200):

	# Feed forward 
    l0 = inputs
    l1 = sigimod(np.dot(l0,w0))
    l2 = sigimod(np.dot(l1,w1))

    # Error Rate calc
    l2_error = outputs - l2

    if (j% 10) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    
    l2_d = l2_error*sigimod(l2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_d.dot(w1.T)
    l1_d = l1_error * sigimod(l1)

    w1 += l1.T.dot(l2_d)
    w0 += l0.T.dot(l1_d)

#Print results
    print l2
