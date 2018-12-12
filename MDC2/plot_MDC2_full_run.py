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
#param json file with index in chain
paramfile = outdir + '/Search_params.json'


#Load chains to make corner plots
chain = np.loadtxt(outdir + '/chain_1.txt')
#pars = sorted(xs.keys())
burn = int(0.25 * chain.shape[0])

#Load param files to iterate through pulsars in pta
with open(paramfile) as pf:
	param_dict = json.load(pf)
	pf.close()

#Which pulsar do we want to look at?
psrs = []
for psr in param_dict.keys():
	psrs.append(psr)

plot_psr = 'J1939+2134'
#plot_psr = psrs[0]

psr_noise_names = []
psr_noise_idx = []

if plot_psr in param_dict:
	for psr, params in param_dict.items():
		if psr == plot_psr:
			for param_names, param_idx in params.items():
				psr_noise_names.append(param_names)
				psr_noise_idx.append(param_idx)
else:
	print('That pulsar is not in the pta.')

#pulsar corner plot
corner.corner(chain[burn:,psr_noise_idx], 30, labels=psr_noise_names);
plt.show()			
#Plot and save corner plots
#plt.savefig(outdir + runname + '_cornerplt.png')
#plt.close()

#Plot upperlimit histogram on gwb
pars = ['log10_A_gw','gamma_gw']
plt.hist(chain[burn:,-5], 30, density = True);
plt.xlabel(pars[-2]);
plt.show()
#plt.close()
