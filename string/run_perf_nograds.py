import flakes
import sys
import numpy as np
import random
import datetime as dt
import string
import os

#SLEN = int(sys.argv[1])
SLENS = range(10, 101, 10)
#ALPHABET = 'actg'
ALPHABET = string.ascii_lowercase
COEFS = [1.0] * 5
OUTPUT_DIR = sys.argv[1]
random.seed(1000)


naive_times = []
numpy_times = []
for slen in SLENS:
    # Generate random instances
    data = [[''.join([random.choice(ALPHABET) for j in xrange(slen)])] for i in xrange(100)]

    # Naive SK
    #sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='cynaive')
    #before = dt.datetime.now()
    #print sk.K(data)
    #sk.K(data)
    #after = dt.datetime.now()
    #naive_times.append((after-before).total_seconds())
    #print "CYTHON NAIVE SK: ", after-before

    # Numpy SK
    sk = flakes.string.StringKernel(order_coefs=COEFS, alphabet=list(ALPHABET), mode='numpy-nograds')
    before = dt.datetime.now()
    sk.K(data)
    after = dt.datetime.now()
    #print "NUMPY SK: ", after-before
    numpy_times.append((after-before).total_seconds())

#np.savetxt(os.path.join(OUTPUT_DIR, 'sk_naive.tsv'), naive_times)
np.savetxt(os.path.join(OUTPUT_DIR, 'sk_numpy_nograds.tsv'), numpy_times)

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
