import matplotlib.pyplot as plt
import numpy as np
import sys

NAIVE = np.loadtxt(sys.argv[1])
NUMPY = np.loadtxt(sys.argv[2])
LENS = range(10,101,10)

plt.plot(LENS, NAIVE, lw=2, label='Non-vectorised SK')
plt.plot(LENS, NUMPY, lw=2, label='VSK')
plt.xlabel('String length')
plt.ylabel('Wall-clock time (in seconds)')
plt.legend(loc=0)

plt.tight_layout()
plt.show()

