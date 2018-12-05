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

topdir = os.getcwd()
#Where the original data is
datadir = topdir + '/mdc2/group1/dataset_1b'
#Where the json noise file is
noisefile = topdir + '/mdc2/group1/challenge1_psr_noise.json'
#Where the refit par files are
pardir = topdir + '/dataset_1b/dataset_1b_correctedpars/'
#Where the chains should be saved to
#NEED TO CHANGE OUTDIR FILE ON DIFFERENT RUNS (ie open1 -> open2)
chaindir = topdir + '/dataset_1b/chains/'
#Where we save figures n stuff
figdir = topdir + '/dataset_1b/Cornerplts/'
#Where we save new json file
noisedir = topdir + '/dataset_1b/'

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

# selection class to break white noise by backend
selection = selections.Selection(selections.by_backend)

##### parameters and priors #####

# white noise parameters
# since we are fixing these to values from the noise file we set
# them as constant parameters
efac = parameter.Constant()
log_10_equad = parameter.Constant()

# red noise parameters
red_noise_log10_A = parameter.LinearExp(-20,-12)
red_noise_gamma = parameter.Uniform(0,7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw = parameter.LinearExp(-18,-12)('log10_A_gw')
gamma_gw = parameter.Constant(4.33)('gamma_gw')

##### Set up signals #####

# white noise
ef = white_signals.MeasurementNoise(efac=efac)
eq = white_signals.EquadNoise(log10_equad = log10_equad)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=red_noise_log10_A, gamma=red_noise_gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

# gwb
# We pass this signal the power-law spectrum as well as the standard
# Hellings and Downs ORF
orf = utils.hd_orf()
crn = gp_signals.FourierBasisCommonGP(pl, orf, components=30, name='gw', Tspan=Tspan)

# timing model
tm = gp_signals.TimingModel()

# full model is sum of components
model = ef + rn + tm  + crn

# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])