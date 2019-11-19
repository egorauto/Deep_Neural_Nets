import numpy as np

#
# (b)
#
# compute the softmax for the preactivations a.
# a is a numpy array
#
def softmax(b):
    # stabilization: b - max(b)
    b = b - np.max(b)
    exp = np.exp(b)
    denom = np.sum(exp)
    return exp / denom

#
# compute the softmax-cossentropy between the preactivations a and the
# correct class y.
# y is an integer indicating the correct class, 0 <= y < np.size(a, axis=-1).
#
def softmax_crossentropy(a, y):
    # stabilization as per HW3
    max_ = np.argmax(a)
    return - (a[y] - a[max_] - np.log(np.sum(np.exp(a - a[max_]))))


#
# (c)
#
# compute the gradient of the softmax-cossentropy between the
# preactivations a and the correct class y with respect to the preactivations
# a.
# y is an integer indicating the correct class, 0 <= y < np.size(a, axis=-1).
#
def grad_softmax_crossentropy(a, y):
    #based on the observation made in answer to 2. a)
    res = softmax(a)
    res[y] -= 1
    return res


#
# (d)
#

# To compute the numerical gradient at a point (a,y), for component i compute
# '(ce(a+da,y)-ce(a,y))/e' where 'da[i] = e' and the other entries of 'da' are
# zero and e is a small number, e.g. 0.0001 (i.e. use the finite differences
# method for each component of the gradient separately).
#
# implemented correctly, the difference between analytical and numerical
# gradient should be of the same magnitude as e
def numerical_gradient(a, y, e):
    grad = np.zeros(np.shape(a))
    for i in range(a.size):
        delta = np.zeros(np.shape(a))
        delta[i] = e
        grad [i] = ((softmax_crossentropy(a + delta, y) - softmax_crossentropy(a, y)) / e)
    return grad

### TESTING AREA ###

import random
for i in range (10):
    r = np.random.rand(10,1)
    y = random.randint(0,9)
    print( np.max(np.abs(numerical_gradient(r,y,0.1) -  grad_softmax_crossentropy(r,y))))
