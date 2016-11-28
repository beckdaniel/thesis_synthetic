import flakes
import sys
import numpy as np
import random
import datetime as dt

SLEN = int(sys.argv[1])
ALPHABET = 'actg'
COEFS = [1.0] * 5
random.seed(1000)

# Generate random instances
data = [[''.join([random.choice(ALPHABET) for j in xrange(SLEN)])] for i in xrange(100)]
#print data

# Naive SK
sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='naive')
before = dt.datetime.now()
print sk.K(data)
after = dt.datetime.now()
print "NAIVE SK: ", after-before

# Numpy SK
sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='numpy')
before = dt.datetime.now()
print sk.K(data)
after = dt.datetime.now()
print "NUMPY SK: ", after-before

# TF SK
sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='tf')
before = dt.datetime.now()
print sk.K(data)
after = dt.datetime.now()
print "TF SK: ", after-before

# TF BATCH SK
sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='tf-batch', batch_size=100)
before = dt.datetime.now()
print sk.K(data)
after = dt.datetime.now()
print "TF BATCH SK: ", after-before
