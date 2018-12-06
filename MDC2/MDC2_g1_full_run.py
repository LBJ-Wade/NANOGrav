from __future__ import division

import numpy as np
import glob, os, json
import matplotlib.pyplot as plt
import scipy.linalg as sl

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

#NEED TO CHANGE FILE ON DIFFERENT RUNS (ie full_run_1 -> full_run_2)
runname = 'full_run_1'

topdir = os.getcwd()
#Where the original data is
datadir = topdir + '/mdc2/group1/dataset_1b'
#Where the json noise file is
noisefile = topdir + '/mdc2/group1/challenge1_psr_noise.json'
#Where the refit par files are
pardir = topdir + '/dataset_1b/dataset_1b_correctedpars/'
#Where the everything should be saved to (chains, cornerplts, histograms, etc.)
outdir = topdir + '/dataset_1b/' + runname
#Where we save new json file
noisedir = topdir + '/dataset_1b/'

parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))

#Loading par and tim files into enterprise Pulsar class
psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)

'''
#Get true noise values for pulsar to plot in corner plot (truth values)
with open(noisefile, 'r') as nf:
	noise_dict = json.load(nf)
	nf.close()
#Unpacking dictionaries in json file to get at noise values
params = {}
for psr in psrs:
	params.update(noise_dict[psr.name])
'''

# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

##### parameters and priors #####

# white noise parameters
efac = parameter.Normal(1.0,0.1)
log10_equad = parameter.Uniform(-8.5,5)

# red noise parameters
red_noise_log10_A = parameter.LinearExp(-20,-12)
red_noise_gamma = parameter.Uniform(0,7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw = parameter.LinearExp(-18,-12)('log10_A_gw')
gamma_gw = parameter.Constant(13/3)('gamma_gw')

##### Set up signals #####

# white noise
ef = white_signals.MeasurementNoise(efac=efac)
eq = white_signals.EquadNoise(log10_equad = log10_equad)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=red_noise_log10_A, gamma=red_noise_gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

# gwb (no spatial correlations)
cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
# Hellings and Downs ORF
orf = utils.hd_orf()


gw = gp_signals.FourierBasisGP(cpl, orf, components=30, Tspan=Tspan, name = 'gw')

#crn = gp_signals.FourierBasisCommonGP(pl, orf, components=30, name='gw', Tspan=Tspan)

# timing model
tm = gp_signals.TimingModel(use_svd = True)

# full model is sum of components
model = ef + eq + rn + tm  + gw

# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])

#Setting white noise parameters to ones in the json file
#pta.set_default_params(params)

#Pick random initial sampling
xs = {}
for par in pta.params:
	print(par)
	xs[par.name] = par.sample()
	print(par.sample())
#xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups
ndim = len(xs)
groups  = [range(0, ndim)]
groups.extend(map(list, zip(range(0,ndim,2), range(1,ndim,2))))
groups.extend([[36]])

# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir = outdir)

# sampler for N steps
N = 100000
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

'''
#Load chains to make corner plots
chain = np.loadtxt(outdir + '/chain_1.txt')
pars = sorted(xs.keys())
burn = int(0.25 * chain.shape[0])

#Plot and save corner plots
corner.corner(chain[burn:,:-4], 30, labels=pars);
plt.savefig(outdir + runname + '_cornerplt.png')
plt.close()

#Plot upperlimit histogram on gwb
plt.hist(chain[burn:,-5], 50, normed=True, histtype='step', lw=2);
plt.xlabel(pars[-1]);
plt.close()
'''

