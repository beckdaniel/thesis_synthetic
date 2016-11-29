import flakes
import sys
import numpy as np
import random
import datetime as dt
import string

#SLEN = int(sys.argv[1])
SLENS = range(10, 11, 10)
#ALPHABET = 'actg'
ALPHABET = string.ascii_lowercase
COEFS = [1.0] * 5
random.seed(1000)

naive_times = []
numpy_times = []
for slen in SLENS:
    # Generate random instances
    data = [[''.join([random.choice(ALPHABET) for j in xrange(slen)])] for i in xrange(100)]

    # Naive SK
    sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='cynaive')
    before = dt.datetime.now()
    #print sk.K(data)
    sk.K(data)
    after = dt.datetime.now()
    naive_times.append((after-before).total_seconds())
    #print "CYTHON NAIVE SK: ", after-before

    # Numpy SK
    sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='numpy')
    before = dt.datetime.now()
    sk.K(data)
    after = dt.datetime.now()
    #print "NUMPY SK: ", after-before
    numpy_times.append((after-before).total_seconds())

print naive_times
print numpy_times

# TF SK
#sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='tf')
#before = dt.datetime.now()
#print sk.K(data)
#after = dt.datetime.now()
#print "TF SK: ", after-before

# TF BATCH SK
#sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='tf-batch', batch_size=100)
#before = dt.datetime.now()
#print sk.K(data)
#after = dt.datetime.now()
#print "TF BATCH SK: ", after-before
