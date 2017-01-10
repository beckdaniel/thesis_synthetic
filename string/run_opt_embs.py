from nltk.corpus import treebank as tb
import numpy as np
import GPy
import sys
import os
import flakes
from collections import defaultdict
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats.stats import pearsonr

SIZES = [10, 20, 50, 100, 200, 500, 1000]
#SIZES = [100, 200]
np.random.seed(1000)
OUTPUT_DIR = sys.argv[1]
EMBS = sys.argv[2]
COEFS = [1.0, 0.5, 0.25]
NOISE = 0.01
GAP = 0.01
MATCH = 0.2

def load_embs_matrix(filename):
    data = np.loadtxt(filename, delimiter=' ', comments=None, dtype=object)
    words = data[:, 0]
    embs = np.array(data[:, 1:], dtype=float)
    print embs.shape
    embs = np.concatenate(([[0.0] * embs.shape[1]], embs))
    words = defaultdict(int, [(word, i+1) for i, word in enumerate(words)])
    return embs, words

embs, words = load_embs_matrix(EMBS)

all_x = np.array([[[w.lower() for w in sent]] for sent in tb.sents()[:1200]])

for size in SIZES:
    size_dir = os.path.join(OUTPUT_DIR, str(size))
    try:
        os.makedirs(size_dir)
    except:
        pass

    #x_train = np.array([[[w.lower() for w in sent]] for sent in tb.sents()[:size]])
    x_train = all_x[:size]
    x_test = all_x[-200:]
    x = np.concatenate((x_train, x_test))
    gaps = []
    matches = []
    coefs_1 = []
    coefs_2 = []
    coefs_3 = []
    noises = []

    maes_b = []
    rmses_b = []
    pearsons_b = []
    nlpds_b = []

    maes_a = []
    rmses_a = []
    pearsons_a = []
    nlpds_a = []


    for i in xrange(20):
        sk = flakes.string.StringKernel(gap_decay=GAP, match_decay=MATCH, order_coefs=COEFS, embs=embs, index=words, mode='tf-batch', device='/gpu:0')

        #gram = sk.K(x_train)
        #y_train = np.random.multivariate_normal([0] * x_train.shape[0], gram + (np.eye(x_train.shape[0]) * NOISE))[:, None]
        gram = sk.K(x)
        y = np.random.multivariate_normal([0] * x.shape[0], gram + (np.eye(x.shape[0]) * NOISE))[:, None]
        y_train = y[:size]
        y_test = y[size:]
        
        sk2 = flakes.wrappers.gpy.GPyStringKernel(order_coefs=[1.0] * 3, embs=embs, index=words, mode='tf-batch', device='/gpu:0')

        model = GPy.models.GPRegression(x_train, y_train, kernel=sk2)
        model.randomize()
        preds, _ = model.predict(x_test)
        maes_b.append(MAE(preds, y_test))
        rmses_b.append(np.sqrt(MSE(preds, y_test)))
        pearsons_b.append(pearsonr(preds, y_test)[0])
        nlpds_b.append(-np.mean(model.log_predictive_density(x_test, y_test)))
        
        model.optimize(messages=True, max_iters=50)

        preds, _ = model.predict(x_test)
        maes_a.append(MAE(preds, y_test))
        rmses_a.append(np.sqrt(MSE(preds, y_test)))
        pearsons_a.append(pearsonr(preds, y_test)[0])
        nlpds_a.append(-np.mean(model.log_predictive_density(x_test, y_test)))
        
        gaps.append(model['string.gap_decay'])
        matches.append(model['string.match_decay'])
        coefs_1.append(model['string.coefs'][0])
        coefs_2.append(model['string.coefs'][1])
        coefs_3.append(model['string.coefs'][2])
        noises.append(model['Gaussian_noise.variance'])

    np.savetxt(os.path.join(size_dir, 'gaps.tsv'), gaps)
    np.savetxt(os.path.join(size_dir, 'matches.tsv'), matches)
    np.savetxt(os.path.join(size_dir, 'coefs_1.tsv'), coefs_1)
    np.savetxt(os.path.join(size_dir, 'coefs_2.tsv'), coefs_2)
    np.savetxt(os.path.join(size_dir, 'coefs_3.tsv'), coefs_3)
    np.savetxt(os.path.join(size_dir, 'noises.tsv'), noises)

    np.savetxt(os.path.join(size_dir, 'maes_b.tsv'), maes_b)
    np.savetxt(os.path.join(size_dir, 'rmses_b.tsv'), rmses_b)
    np.savetxt(os.path.join(size_dir, 'pearsons_b.tsv'), pearsons_b)
    np.savetxt(os.path.join(size_dir, 'nlpds_b.tsv'), nlpds_b)

    np.savetxt(os.path.join(size_dir, 'maes_a.tsv'), maes_a)
    np.savetxt(os.path.join(size_dir, 'rmses_a.tsv'), rmses_a)
    np.savetxt(os.path.join(size_dir, 'pearsons_a.tsv'), pearsons_a)
    np.savetxt(os.path.join(size_dir, 'nlpds_a.tsv'), nlpds_a)
