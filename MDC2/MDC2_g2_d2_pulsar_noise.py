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
runname = '/pulsar_noise_runs'
group = '/group2'
dataset = '/dataset_2'

topdir = os.getcwd()
#Where the original data is
origdatadir = topdir + '/mdc2' + group + dataset
#Where the json noise file is
noisefile = topdir + '/mdc2' + '/group1' + '/challenge1_psr_noise.json'
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
            print('Cannot refit Pulsar: ' + new_libs_psr.name)
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

psr_dict = {}

for psr in psrs:
	print('Working on ' + psr.name)
	psroutdir = outdir + '/' + psr.name + '/'
	if not os.path.exists(psroutdir):
		os.mkdir(psroutdir)

	# initialize PTA
	pta = signal_base.PTA([model(psr)])

	#Pick random initial sampling
	xs = {par.name: par.sample() for par in pta.params}

	# dimension of parameter space
	ndim = len(xs)

	# initial jump covariance matrix
	cov = np.diag(np.ones(ndim) * 0.01**2)

	# Now we figure out which indices the red noise parameters have
	rn_idx1 = pta.param_names.index(psr.name + '_red_noise_log10_A')
	rn_idx2 = pta.param_names.index(psr.name + '_red_noise_gamma')

	# set up jump groups by red noise groups
	ndim = len(xs)
	groups  = [range(0, ndim)]
	groups.extend([[rn_idx1,rn_idx2]])

	# intialize sampler
	sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir=psroutdir)

	# sampler for N steps
	N = int(1e6)
	x0 = np.hstack(p.sample() for p in pta.params)
	sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

	chain = np.loadtxt(psroutdir + 'chain_1.txt')
	pars = sorted(xs.keys())
	burn = int(0.25 * chain.shape[0])

	#make dictionary of pulsar parameters from these runs
	psr_dict[psr.name] = {}
	for idx,param_name in enumerate(pars):
		psr_dict[psr.name][param_name] = np.median(chain[burn:,idx])
	#Save to json file
	with open(psroutdir + '/fit_psr_noise.json','w') as paramfile:
		json.dump(psr_dict[psr.name],paramfile,sort_keys = True,indent = 4)
		paramfile.close()

# Now we want to save this all as a json file
with open(outdir+"/all_fit_psr_noise.json", 'w') as fpn:
    json.dump(psr_dict, fpn ,sort_keys = True,indent = 4)
    fpn.close()