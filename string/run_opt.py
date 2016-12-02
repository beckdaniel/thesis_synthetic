import flakes
import numpy as np
import GPy
import string
import sys
import os
import random
from sklearn.metrics import mean_squared_error as MSE

#SLEN = int(sys.argv[1])
#SLENS = range(10, 101, 10)
#ALPHABET = 'ac'#tg'
ALPHABET = string.ascii_letters
COEFS = [1.0, 0.5, 0.25]
NOISE = 0.01
GAP = 0.01
MATCH = 0.2
#COEFS = [1.0]
OUTPUT_DIR = sys.argv[1]
random.seed(1000)
np.random.seed(1000)
SLEN = 40
SIZES = [10, 20, 50, 100, 200, 500]
#SIZES = [10, 20]



x_all = np.array([[''.join([random.choice(ALPHABET) for j in xrange(SLEN)])] for i in xrange(500)])
#sk = flakes.string.StringKernel(gap_decay=GAP, match_decay=MATCH, order_coefs=COEFS, alphabet=list(ALPHABET), mode='tf-batch')
#gram = sk.K(x_all)
#print gram + (np.eye(x.shape[0]) * NOISE)
#gram *= 2
#y_all = np.random.multivariate_normal([0] * 300, gram + (np.eye(x_all.shape[0]) * NOISE))[:, None]

x_test = x_all[-200:]
#y_test = y_all[-200:]


for size in SIZES:
    size_dir = os.path.join(OUTPUT_DIR, str(size))
    try:
        os.makedirs(size_dir)
    except:
        pass

    
    x_train = x_all[:size]
    #y_train = y_all[:size]
    
    
    gaps = []
    matches = []
    coefs_1 = []
    coefs_2 = []
    coefs_3 = []
    noises = []
    #rmses = []
    #nlpds = []
    for i in xrange(20):
        sk = flakes.string.StringKernel(gap_decay=GAP, match_decay=MATCH, order_coefs=COEFS, alphabet=list(ALPHABET), mode='tf-batch')
        gram = sk.K(x_train)
        y_train = np.random.multivariate_normal([0] * x_train.shape[0], gram + (np.eye(x_train.shape[0]) * NOISE))[:, None]

        sk2 = flakes.wrappers.gpy.GPyStringKernel(order_coefs=[1.0] * 3, alphabet=list(ALPHABET), mode='tf-batch', batch_size=100)
        model = GPy.models.GPRegression(x_train, y_train, kernel=sk2)
        model.randomize()
        #print model
        #print model['.*coefs.*']
        model.optimize(messages=True, max_iters=50)
        #print model
        #print model['.*coefs.*']

        gaps.append(model['string.gap_decay'])
        matches.append(model['string.match_decay'])
        coefs_1.append(model['string.coefs'][0])
        coefs_2.append(model['string.coefs'][1])
        coefs_3.append(model['string.coefs'][2])
        noises.append(model['Gaussian_noise.variance'])

        #y_pred = model.predict(x_test)[0]
        #rmses.append(np.sqrt(MSE(y_pred, y_test)))
        #nlpds.append(-np.mean(model.log_predictive_density(x_test, y_test)))

    np.savetxt(os.path.join(size_dir, 'gaps.tsv'), gaps)
    np.savetxt(os.path.join(size_dir, 'matches.tsv'), matches)
    np.savetxt(os.path.join(size_dir, 'coefs_1.tsv'), coefs_1)
    np.savetxt(os.path.join(size_dir, 'coefs_2.tsv'), coefs_2)
    np.savetxt(os.path.join(size_dir, 'coefs_3.tsv'), coefs_3)
    np.savetxt(os.path.join(size_dir, 'noises.tsv'), noises)
    print gaps
    print matches
    print coefs_1
    print coefs_2
    print coefs_3
    print noises
