import matplotlib.pyplot as plt
import corner, os, glob, json
import numpy as np

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise.signals import signal_base

topdir = os.getcwd()
#Where the original data is
datadir = topdir + '/mdc2/group1/dataset_1b/'
#Where we save figures n stuff
figdir = topdir + '/dataset_1b/Cornerplts/'
#Where the json noise file is
noisefile = topdir + '/mdc2/group1/challenge1_psr_noise.json'
#Where the chains should be saved to
chaindir = topdir + '/dataset_1b/chains/'


parfiles = sorted(glob.glob(datadir + '*.par'))
timfiles = sorted(glob.glob(datadir + '*.tim'))

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)

##### parameters and priors #####

# Uniform prior on EFAC and EQUAD
efac = parameter.Uniform(0.1, 5.0)
log10_equad = parameter.Uniform(-10.0,-4.0)

# red noise parameters
# Uniform in log10 Amplitude and in spectral index
log10_A = parameter.Uniform(-18,-12)
gamma = parameter.Uniform(0,7)

##### Set up signals #####

# white noise
ef = white_signals.MeasurementNoise(efac=efac)
eq = white_signals.EquadNoise(log10_equad = log10_equad)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)

# timing model
tm = gp_signals.TimingModel()

# full model is sum of components
model = ef + rn + tm + eq

for psr in psrs:
	# initialize PTA
	pta = signal_base.PTA([model(psr)])
	#Pick random initial sampling
	xs = {par.name: par.sample() for par in pta.params}

	#Get true noise values for pulsar to plot in corner plot (truth values)
	with open(noisefile, 'r') as nf:
		inputvals = json.load(nf)
		nf.close()
	#Unpacking dictionaries in json file to get at noise values
	noise_vals = inputvals[psr.name]
	truths = [noise_vals["efac"], noise_vals["equad"], noise_vals["rn_spec_ind"], noise_vals["rn_log10_A"]]

	chain = np.loadtxt(chaindir + psr.name + '/chain_1.txt')
	pars = sorted(xs.keys())
	burn = int(0.25 * chain.shape[0])

	#Make corner plots
	corner.corner(chain[burn:,:-4], 30, truths = truths, labels=pars)
	plt.savefig(figdir + psr.name + '_cornerplt.png')
	plt.close()
