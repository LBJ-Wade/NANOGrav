from __future__ import division
import numpy as np
import glob, os, json, pickle
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
runname = '/red_noise_with_corr_1'
group = '/group2'
dataset = '/dataset_2'

topdir = os.getcwd()
#Where the original data is
origdatadir = topdir + '/mdc2' + group + dataset
#Where the json noise file is
noisefile = topdir + '/mdc2' + group + '/challenge1_psr_noise.json'
#Where the dataset files are located
datadir = topdir + dataset
#Where the refit par files are saved to
pardir = datadir + '/corrected_pars/'
#Where the everything should be saved to (chains, cornerplts, histograms, etc.)
outdir = datadir + runname
#Where we save figures n stuff
figdir = datadir + '/Cornerplts/'
#The new json file we made
updatednoisefile = datadir + 'fit_psr_noise.json'
#The pickled pulsars
psr_obj_file = datadir + '/psr_objects.pickle'

if os.path.exists(datadir) == False:
    os.mkdir(datadir)
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

def Refit_pars(origdir,newdir):
    orig_parfiles = sorted(glob.glob(origdir + '/*.par'))
    orig_timfiles = sorted(glob.glob(origdir + '/*.tim'))
    #Load all of the Pulsars into libstempo
    orig_libs_psrs = []
    for p, t in zip(orig_parfiles, orig_timfiles):
        orig_libs_psr = libs.tempopulsar(p, t)
        orig_libs_psrs.append(orig_libs_psr)

    #Fit the par files again
    #Save them to new directory (Overwrites ones currently used in newdatadir)
    if os.path.exists(newdir) == False:
        os.mkdir(newdir)
    for new_libs_psr in orig_libs_psrs:
        new_libs_psr['DM'].fit = False
        new_libs_psr['DM1'].fit = False
        new_libs_psr['DM2'].fit = False
        try:
            new_libs_psr.fit(iters=3)
        except:
            continue
        new_libs_psr.savepar(newdir + new_libs_psr.name + '.par')

#Load all the pulsars if no pickle file
try:
    #Load pulsars from pickle file
    with open(psr_obj_file,'rb') as psrfile:
        psrs = pickle.load(psrfile)
        psrfile.close()
except:
    #If no pickle file, load and save pulsars

    #Load refit par files if they exist, else fit them and save them
    if os.path.exists(pardir) == True:
        parfiles = sorted(glob.glob(pardir + '/*.par'))
    else:
        #Refit par files using libstempo
        Refit_pars(origdatadir,pardir)
        parfiles = sorted(glob.glob(pardir + '/*.par'))

    #Loading tim files into enterprise Pulsar class
    timfiles = sorted(glob.glob(origdatadir + '/*.tim'))

    psrs = []
    for p, t in zip(parfiles,timfiles):
        psr = Pulsar(p, t)
        psrs.append(psr)
    #Save 9yr pulsars to a pickle file
    with open(psr_obj_file,'wb') as psrfile:
        pickle.dump(psrs,psrfile)
        psrfile.close()

# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

##### parameters and priors #####

# white noise parameters
efac = parameter.Uniform(0.5,4.0)
log10_equad = parameter.Uniform(-8.5,5)

# red noise parameters
red_noise_log10_A = parameter.Uniform(-20,-11)
red_noise_gamma = parameter.Uniform(0,7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
#Linear exp is upper limit run! Uniform is detection
log10_A_gw = parameter.Uniform(-20,-11)('zlog10_A_gw')
gamma_gw = parameter.Constant(13/3)('zgamma_gw')

##### Set up signals #####

# timing model
tm = gp_signals.TimingModel()

# white noise
ef = white_signals.MeasurementNoise(efac=efac)
eq = white_signals.EquadNoise(log10_equad = log10_equad)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=red_noise_log10_A, gamma=red_noise_gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)


cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)

#Common red noise process with no correlations
#crn = gp_signals.FourierBasisGP(spectrum = cpl, components=30, Tspan=Tspan, name = 'gw')

# gwb with Hellings and Downs correlations
# Hellings and Downs ORF
orf = utils.hd_orf()
gwb = gp_signals.FourierBasisCommonGP(cpl, orf, components=30, name='gw', Tspan=Tspan)

# full model is sum of components
model = ef + eq + rn + tm + gwb #+ crn

# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])

#make dictionary of pulsar parameters from these runs
param_dict = {}
for psr in pta.pulsars:
    param_dict[psr] = {}
    for param, idx in zip(pta.param_names,range(len(pta.param_names))):
        if param.startswith(psr):
            param_dict[psr][param] = idx
#Save to json file
with open(outdir + '/Search_params.json','w') as paramfile:
    json.dump(param_dict,paramfile,sort_keys = True,indent = 4)
    paramfile.close()

#Pick random initial sampling
xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups
groups  = [range(0, ndim)]
groups.extend(map(list, zip(range(0,ndim,2), range(1,ndim,2))))
groups.extend([[ndim-1]])

# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir = outdir)

# sampler for N steps
N = int(1e6)
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)