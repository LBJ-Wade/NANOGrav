import matplotlib.pyplot as plt
import corner
import numpy as np

chain = np.loadtxt('chains/mdc/open1/chain_1.txt')
#pars = sorted(xs.keys())
burn = int(0.25 * chain.shape[0])

#Plot corner plots
#truths = [1.0, 4.33, np.log10(5e-14)]
corner.corner(chain[burn:,:-4], 30)
plt.show()
