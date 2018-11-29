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


savetime = True


datadir = os.getcwd() + '/mdc2/group1/dataset_1b'
savedir = os.getcwd() + '/mdc2/group1/dataset_1b_correctedpars/'

orig_parfiles = sorted(glob.glob(datadir + '/*.par'))
orig_timfiles = sorted(glob.glob(datadir + '/*.tim'))

parfile_J0030 = datadir + '/J0030+0451.par'
timfile_J0030 = datadir + '/J0030+0451.tim'

#Load Pulsar into enterprise
psr_J0030 = Pulsar(parfile_J0030,timfile_J0030)
#Load Pulsar into libstempo
libs_psr_J0030 = libs.tempopulsar(parfile = parfile_J0030,timfile = timfile_J0030)


#All of the Pulsars!
orig_libs_psrs = []
for p, t in zip(orig_parfiles, orig_timfiles):
    orig_libs_psr = libs.tempopulsar(p, t)
    orig_libs_psrs.append(orig_libs_psr)


#Fit the par files again
#Save them to new directory (Overwrites ones currently used in savedir)
for orig_libs_psr in orig_libs_psrs:
    orig_libs_psr.fit()
    if savetime == True:
        #print(savedir + libs_psr.name + '.par')
        orig_libs_psr.savepar(savedir + orig_libs_psr.name + '.par')
        orig_libs_psr.savetim(savedir + orig_libs_psr.name + '.tim')


#Check new residuals
fit_parfiles = sorted(glob.glob(savedir + '/*.par'))
fit_timfiles = sorted(glob.glob(savedir + '/*.tim'))

fit_libs_psrs = []
for p, t in zip(fit_parfiles, fit_timfiles):
    fit_libs_psr = libs.tempopulsar(p, t)
    fit_libs_psrs.append(fit_libs_psr)


#libsplt.plotres(libs_psrs[7])


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

# initialize PTA
pta = signal_base.PTA([model(psr_J0030)])


#Pick random initial sampling
xs = {par.name: par.sample() for par in pta.params}


# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups
ndim = len(xs)
groups  = [range(0, ndim)]
groups.extend([[2,3]])

# intialize sampler
#NEED TO CHANGE OUTDIR FILE ON DIFFERENT RUNS (ie open1 -> open2)
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir='chains/mdc/open1/')


# sampler for N steps
N = 100000
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)


chain = np.loadtxt('chains/mdc/open1/chain_1.txt')
pars = sorted(xs.keys())
burn = int(0.25 * chain.shape[0])

#Plot corner plots
#truths = [1.0, 4.33, np.log10(5e-14)]
#corner.corner(chain[burn:,:-4], 30, labels=pars);

