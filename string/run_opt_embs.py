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
DEVICE = sys.argv[3]
EMBS = sys.argv[2]
COEFS = [1.0, 0.5, 0.25]
NOISE = 0.01
GAP = 0.5
MATCH = 0.2

def load_embs_matrix(filename):
    data = np.loadtxt(filename, delimiter=' ', comments=None, dtype=object)
    words = data[:, 0]
    embs = np.array(data[:, 1:], dtype=float)
    print embs.shape
    embs = np.concatenate(([[0.0] * embs.shape[1]], embs))
    words = defaultdict(int, [(word, i+1) for i, word in enumerate(words)])
    return embs, words


def average_embs(x, embs, words):
    return np.array([np.mean([embs[words[w]] for w in sent[0]], axis=0) for sent in x])

embs, words = load_embs_matrix(EMBS)
all_x = np.array([[[w.lower() for w in sent]] for sent in tb.sents()[:1200]])
all_x_avg = average_embs(all_x, embs, words)
print all_x_avg.shape
#sys.exit(0)


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

    x_avg_train = all_x_avg[:size]
    x_avg_test = all_x_avg[-200:]
    
    gaps = []
    matches = []
    coefs_1 = []
    coefs_2 = []
    coefs_3 = []
    noises = []

    metrics_b = []
    #maes_b = []
    #rmses_b = []
    #pearsons_b = []
    #nlpds_b = []

    metrics_a = []
    #maes_a = []
    #rmses_a = []
    #pearsons_a = []
    #nlpds_a = []

    metrics_lin = []
    metrics_rbf = []
    metrics_mat32 = []
    metrics_mat52 = []
    
    

    for i in xrange(20):
        sk = flakes.string.StringKernel(gap_decay=GAP, match_decay=MATCH, order_coefs=COEFS, embs=embs, index=words, mode='tf-batch', device=DEVICE)

        #gram = sk.K(x_train)
        #y_train = np.random.multivariate_normal([0] * x_train.shape[0], gram + (np.eye(x_train.shape[0]) * NOISE))[:, None]
        gram = sk.K(x)
        y = np.random.multivariate_normal([0] * x.shape[0], gram + (np.eye(x.shape[0]) * NOISE))[:, None]
        y_train = y[:size]
        y_test = y[size:]
        
        sk2 = flakes.wrappers.gpy.GPyStringKernel(order_coefs=[1.0] * 3, embs=embs, index=words, mode='tf-batch', device=DEVICE)

        model = GPy.models.GPRegression(x_train, y_train, kernel=sk2)
        model.randomize()
        preds, _ = model.predict(x_test)
        metrics_b.append([-np.mean(model.log_predictive_density(x_test, y_test)),
                          MAE(preds, y_test),
                          pearsonr(preds, y_test)[0],
                          np.sqrt(MSE(preds, y_test))])

        model.optimize(messages=True, max_iters=50)

        preds, _ = model.predict(x_test)
        metrics_a.append([-np.mean(model.log_predictive_density(x_test, y_test)),
                          MAE(preds, y_test),
                          pearsonr(preds, y_test)[0],
                          np.sqrt(MSE(preds, y_test))])
        
        gaps.append(model['string.gap_decay'])
        matches.append(model['string.match_decay'])
        coefs_1.append(model['string.coefs'][0])
        coefs_2.append(model['string.coefs'][1])
        coefs_3.append(model['string.coefs'][2])
        noises.append(model['Gaussian_noise.variance'])

        # OTHER MODELS

        lin = GPy.kern.Linear(50)
        model_lin = GPy.models.GPRegression(x_avg_train, y_train, kernel=lin)
        model_lin.optimize(messages=True, max_iters=50)
        preds, _ = model_lin.predict(x_avg_test)
        metrics_lin.append([-np.mean(model_lin.log_predictive_density(x_avg_test, y_test)),
                          MAE(preds, y_test),
                          pearsonr(preds, y_test)[0],
                          np.sqrt(MSE(preds, y_test))])

        rbf = GPy.kern.RBF(50)
        model_rbf = GPy.models.GPRegression(x_avg_train, y_train, kernel=rbf)
        model_rbf.optimize(messages=True, max_iters=50)
        preds, _ = model_rbf.predict(x_avg_test)
        metrics_rbf.append([-np.mean(model_rbf.log_predictive_density(x_avg_test, y_test)),
                          MAE(preds, y_test),
                          pearsonr(preds, y_test)[0],
                          np.sqrt(MSE(preds, y_test))])

        mat32 = GPy.kern.Matern32(50)
        model_mat32 = GPy.models.GPRegression(x_avg_train, y_train, kernel=mat32)
        model_mat32.optimize(messages=True, max_iters=50)
        preds, _ = model_mat32.predict(x_avg_test)
        metrics_mat32.append([-np.mean(model_mat32.log_predictive_density(x_avg_test, y_test)),
                          MAE(preds, y_test),
                          pearsonr(preds, y_test)[0],
                          np.sqrt(MSE(preds, y_test))])

        mat52 = GPy.kern.Matern52(50)
        model_mat52 = GPy.models.GPRegression(x_avg_train, y_train, kernel=mat52)
        model_mat52.optimize(messages=True, max_iters=50)
        preds, _ = model_mat52.predict(x_avg_test)
        metrics_mat52.append([-np.mean(model_mat52.log_predictive_density(x_avg_test, y_test)),
                          MAE(preds, y_test),
                          pearsonr(preds, y_test)[0],
                          np.sqrt(MSE(preds, y_test))])
        

    np.savetxt(os.path.join(size_dir, 'gaps.tsv'), gaps)
    np.savetxt(os.path.join(size_dir, 'matches.tsv'), matches)
    np.savetxt(os.path.join(size_dir, 'coefs_1.tsv'), coefs_1)
    np.savetxt(os.path.join(size_dir, 'coefs_2.tsv'), coefs_2)
    np.savetxt(os.path.join(size_dir, 'coefs_3.tsv'), coefs_3)
    np.savetxt(os.path.join(size_dir, 'noises.tsv'), noises)

    np.savetxt(os.path.join(size_dir, 'metrics_b.tsv'), metrics_b)
    np.savetxt(os.path.join(size_dir, 'metrics_a.tsv'), metrics_a)

    np.savetxt(os.path.join(size_dir, 'metrics_lin.tsv'), metrics_lin)
    np.savetxt(os.path.join(size_dir, 'metrics_rbf.tsv'), metrics_rbf)
    np.savetxt(os.path.join(size_dir, 'metrics_mat32.tsv'), metrics_mat32)
    np.savetxt(os.path.join(size_dir, 'metrics_mat52.tsv'), metrics_mat52)
