import flakes
import numpy as np
import GPy
import string
import sys
import random

#SLEN = int(sys.argv[1])
#SLENS = range(10, 101, 10)
ALPHABET = 'actg'
#ALPHABET = string.ascii_lowercase
COEFS = [1.0] * 2
#OUTPUT_DIR = sys.argv[1]
random.seed(1000)
np.random.seed(1000)
SLEN = 40
SIZE = 200
NOISE = 0.01

x = np.array([[''.join([random.choice(ALPHABET) for j in xrange(SLEN)])] for i in xrange(SIZE)])
sk = flakes.string.StringKernel(gap_decay=0.1, match_decay=0.2, order_coefs=COEFS, alphabet=list(ALPHABET), mode='numpy')
gram = sk.K(x)
y = np.random.multivariate_normal([0] * SIZE, gram + (np.eye(x.shape[0]) * NOISE))

print y

sk = flakes.wrappers.gpy.GPyStringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='tf-batch', batch_size=100)
#sk = flakes.wrappers.gpy.GPyStringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='numpy')
model = GPy.models.GPRegression(x, y[:, None], kernel=sk)
print model
print model['.*coefs.*']
model.optimize(messages=True, max_iters=50)
print model
print model['.*coefs.*']
