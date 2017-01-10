import numpy as np
import matplotlib.pyplot as plt
import sys
import os

SIZES = [10, 20, 50, 100, 200, 500, 1000]
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

fig, ax = plt.subplots(2, 3, figsize=(10,7), sharey=True)

def plot_hyper(hyper, ax):
    mean = [np.median(hypers[size][hyper]) for size in SIZES]
    std2 = [2 * np.std(hypers[size][hyper]) for size in SIZES]
    print [hypers[size][hyper] for size in SIZES]
    #ax.set_xscale('log')
    #ax.errorbar(SIZES, mean, yerr=std2, fmt='o', lw=2, c='k', markersize='7', ecolor='g', capthick=2)
    bp = ax.boxplot([hypers[size][hyper] for size in SIZES], sym='', labels=SIZES, whis=[5, 95])
    for box in bp['boxes']:
        box.set(linewidth=2)
    for m in bp['medians']:
        m.set(linewidth=2)
    for w in bp['whiskers']:
        w.set(linewidth=2)
    for c in bp['caps']:
        c.set(linewidth=2)
    #ax.set_xlim([5, 1500])
    #ax.set_xticks(SIZES)
    #ax.set_xticklabels(SIZES)
    #ylim = ax.get_ylim()
    #ax.set_ylim([-0.25 * ylim[1], ylim[1]])
    ax.set_ylim([-0.25, 2.0])
    if hyper == 'gaps':
        ax.set_title(r'$\lambda_{g}$', fontsize=18)
        ax.axhline(0.01, lw=2, linestyle='-', c='k')
    elif hyper == 'matches':
        ax.set_title(r'$\lambda_{m}$', fontsize=18)
        #ax.set_title('match decay (0.2)')
        ax.axhline(0.2, lw=2, linestyle='-', c='k')
    elif hyper == 'noises':
        ax.set_title(r'$\sigma^2$', fontsize=18)
        #ax.set_title('noise (0.01)')
        ax.axhline(0.01, lw=2, linestyle='-', c='k')
        #ax.set_ylim([-0.5, 0.5])
    elif hyper == 'coefs_1':
        ax.set_title(r'$\mu_1$', fontsize=18)
        #ax.set_title('1-gram coefficient (1.0)')
        ax.axhline(1,0, lw=2, linestyle='-', c='k')
    elif hyper == 'coefs_2':
        ax.set_title(r'$\mu_2$', fontsize=18)
        #ax.set_title('2-gram coefficient (0.5)')
        ax.axhline(0.5, lw=2, linestyle='-', c='k')
    elif hyper == 'coefs_3':
        ax.set_title(r'$\mu_3$', fontsize=18)
        #ax.set_title('3-gram coefficient (0.25)')
        ax.axhline(0.25, lw=2, linestyle='-', c='k')


    

plot_hyper('gaps', ax[0, 0])
plot_hyper('matches', ax[0, 1])
plot_hyper('noises', ax[0, 2])
plot_hyper('coefs_1', ax[1, 0])
plot_hyper('coefs_2', ax[1, 1])
plot_hyper('coefs_3', ax[1, 2])


plt.tight_layout(pad=2)
fig.text(0.525, 0.01, 'number of instances', ha='center', fontsize=14)
fig.text(0.02, 0.525, 'hyperparameter value', ha='center', va='center', rotation='vertical', fontsize=14)
plt.show()

