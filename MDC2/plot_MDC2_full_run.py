import matplotlib.pyplot as plt
import corner, os, glob, json
import numpy as np

#NEED TO CHANGE FILE ON DIFFERENT RUNS (ie full_run_1 -> full_run_2)
runname = 'full_run_1'

topdir = os.getcwd()
#Where the original data is
datadir = topdir + '/mdc2/group1/dataset_1b'
#Where the json noise file is
noisefile = topdir + '/mdc2/group1/challenge1_psr_noise.json'
#Where the everything should be saved to (chains, cornerplts, histograms, etc.)
outdir = topdir + '/dataset_1b/' + runname


#Load chains to make corner plots
chain = np.loadtxt(outdir + '/chain_1.txt')
#pars = sorted(xs.keys())
burn = int(0.25 * chain.shape[0])

print(chain.shape)
print(burn)
print(chain[burn:,-5].shape)
'''
#Plot and save corner plots
corner.corner(chain[burn:,-4], 30, labels=pars);
plt.show()
#plt.savefig(outdir + runname + '_cornerplt.png')
#plt.close()'''

#Plot upperlimit histogram on gwb
plt.hist(chain[burn:,-5], 50, density = False, histtype='step', lw=2);
#plt.xlabel(pars[-1]);
plt.show()
#plt.close()
