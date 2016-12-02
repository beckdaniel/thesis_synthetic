import numpy as np
import matplotlib.pyplot as plt
import sys
import os

SIZES = [10, 20, 50, 100, 200, 500]
RESULTS_DIR = sys.argv[1]

hypers = {}
for size in SIZES:
    hypers[size] = {}
    size_dir = os.path.join(RESULTS_DIR, str(size))
    hyper_files = os.listdir(size_dir)
    for hfile in hyper_files:
        h = hfile.split('.')[0]
        hypers[size][h] = np.loadtxt(os.path.join(size_dir, hfile))
    
#print hypers

fig, ax = plt.subplots(3, 2)

def plot_hyper(hyper, ax):
    mean = [np.mean(hypers[size][hyper]) for size in SIZES]
    std2 = [2 * np.std(hypers[size][hyper]) for size in SIZES]
    print mean
    print SIZES
    print std2
    #ax.set_xscale('log')
    ax.errorbar(SIZES, mean, yerr=std2, fmt='o')
    ax.set_xlim([0, 600])


    

plot_hyper('gaps', ax[0, 0])
plot_hyper('matches', ax[1, 0])
plot_hyper('noises', ax[2, 0])
plot_hyper('coefs_1', ax[0, 1])
plot_hyper('coefs_2', ax[1, 1])
plot_hyper('coefs_3', ax[2, 1])
plt.tight_layout()
plt.show()

