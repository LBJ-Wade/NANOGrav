from __future__ import division

import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg as sl
import os

import libstempo as libs
import libstempo.plot as libsplt

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise.signals import signal_base

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

#Where the original data is
datadir = os.getcwd() + '/mdc2/group1/dataset_1b'
#Where the refit par files are
pardir = os.getcwd() + '/dataset_1b_correctedpars/'
#Where the json noise file is
noisefile = os.getcwd() + '/mdc2/group1/challenge1_psr_noise.json'
#Where the chains should be saved to
#NEED TO CHANGE OUTDIR FILE ON DIFFERENT RUNS (ie open1 -> open2)
chaindir = os.getcwd() + '/chains/dataset_1b/'
#Where we save figures n stuff
figdir = os.getcwd() + '/Cornerplts/'

def Refit_pars(origdir,newdir):
	orig_parfiles = sorted(glob.glob(origdir + '/*.par'))
	orig_timfiles = sorted(glob.glob(origdir + '/*.tim'))
	#Load all of the Pulsars!
	orig_libs_psrs = []
	for p, t in zip(orig_parfiles, orig_timfiles):
		orig_libs_psr = libs.tempopulsar(p, t)
		orig_libs_psrs.append(orig_libs_psr)

	#Fit the par files again
	#Save them to new directory (Overwrites ones currently used in savedir)
	for orig_libs_psr in orig_libs_psrs:
		orig_libs_psr['DM'].fit = False
		orig_libs_psr['DM1'].fit = False
		orig_libs_psr['DM2'].fit = False
		orig_libs_psr.fit(iters=10)
		orig_libs_psr.savepar(newdir + orig_libs_psr.name + '.par')


parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)


# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

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

noise_params = ["efac", "equad",  "rn_log10_A", "rn_spec_idx"]

for psr in psrs:
	outdir = chaindir + psr.name + '/'
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	# initialize PTA
	pta = signal_base.PTA([model(psr)])

	#Pick random initial sampling
	xs = {par.name: par.sample() for par in pta.params}

	# dimension of parameter space
	ndim = len(xs)

	# initial jump covariance matrix
	cov = np.diag(np.ones(ndim) * 0.01**2)

	# Now we figure out which indices the red noise parameters have
	rn_idx1 = pta.param_names.index(psr.name + 'red_noise_log10_equad')
	rn_idx2 = pta.param_names.index(psr.name + 'red_noise_gamma')

	# set up jump groups by red noise groups
	ndim = len(xs)
	groups  = [range(0, ndim)]
	groups.extend([[rn_idx1,rn_idx2]])
As
	# intialize sampler
	sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir=outdir)

	# sampler for N steps
	N = 100000
	x0 = np.hstack(p.sample() for p in pta.params)
	sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)


	chain = np.loadtxt(outdir + 'chain_1.txt')
	pars = sorted(xs.keys())
	burn = int(0.25 * chain.shape[0])

	#Get true noise values for pulsar to plot in corner plot (truth values)
	with open(noisefile, 'r') as nf:
		inputvals = json.load(nf)
		nf.close()
	#Unpacking dictionaries in json file to get at noise values
	noise_vals = inputvals[psr.name]
	truths = [noise_vals["efac"], noise_vals["rn_spec_ind"], noise_vals["rn_log10_A"], noise_vals["equad"]]

	#Make corner plots
	corner.corner(chain[burn:,:-4], 30, truths = truths, labels=pars)
	plt.savefig(figdir + psr.name + '_cornerplt.pdf')
	plt.close()

	#make dictionary of pulsar parameters from these runs
	psr_dict[psr.name] = {}
	for p in noise_params:

