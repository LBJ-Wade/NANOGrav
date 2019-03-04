from __future__ import division

import numpy as np
import glob, os, json, pickle
import matplotlib.pyplot as plt
import scipy.linalg as sl
from scipy.stats import skewnorm

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

import modded_cw_functions as fns

#NEED TO CHANGE FILE ON DIFFERENT RUNS (ie full_run_1 -> full_run_2)
runname = '/pulsar_noise_runs_with_cw'
group = '/group1'
dataset = '/dataset_3b'

topdir = os.getcwd()
#Where the original data is
origdatadir = topdir + '/mdc2' + group + dataset
#Where the json noise file is
noisefile = topdir + '/mdc2' + group + '/challenge1_psr_noise.json'
#Where the dataset files are located
datadir = topdir + dataset
#Where the everything should be saved to
outdir = datadir + runname
#The new json file we made
updatednoisefile = datadir + 'fit_psr_noise.json'
#The pickled pulsars
psr_obj_file = datadir + '/psr_objects.pickle'

if os.path.exists(datadir) == False:
    os.mkdir(datadir)
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

'''def Refit_pars(origdir,newdir):
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
		print('Refitting ' + orig_libs_psr.name)
		orig_libs_psr['DM'].fit = False
		orig_libs_psr['DM1'].fit = False
		orig_libs_psr['DM2'].fit = False
		orig_libs_psr.fit(iters=10)
		orig_libs_psr.savepar(newdir + orig_libs_psr.name + '.par')
		orig_libs_psr.savetim(newdir + orig_libs_psr.name + '.tim')
		if input('Continue?') == 'y':
			continue
		else:
			break'''


parfiles = sorted(glob.glob(origdatadir + '/*.par'))
timfiles = sorted(glob.glob(origdatadir + '/*.tim'))

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

#CW Stuff taken from caitlin's code
# cw params
cos_gwtheta = parameter.Uniform(-1,1)('cos_gwtheta')
gwphi = parameter.Uniform(0,2*np.pi)('gwphi')
#log10_mc = parameter.LinearExp(7,10)('log10_mc')
log10_mc = parameter.Uniform(7,10)('log10_mc')
#log10_fgw = BoundedNormal(log_f, log_f_err, -9, np.log10(3*10**(-7)))('log10_fgw')
log10_fgw = parameter.Uniform(-9, np.log10(3*10**(-7)))('log10_fgw')

phase0 = parameter.Uniform(0, 2*np.pi)('phase0')
psi = parameter.Uniform(0, np.pi)('psi')
cos_inc = parameter.Uniform(-1, 1)('cos_inc')

##sarah's change
p_phase = parameter.Uniform(0, 2*np.pi)
p_dist = parameter.Normal(0, 1)

#log10_h = parameter.LinearExp(-18, -11)('log10_h')
log10_h = parameter.Uniform(-18, -11)('log10_h')
#log10_dL = parameter.Constant(np.log10(85.8))('log10_dL')

cw_wf = fns.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc, 
                 log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0,
                 psi=psi, cos_inc=cos_inc, p_dist = p_dist, p_phase = p_phase, model = 'evolve', tref = np.max(tmax))
cw = fns.CWSignal(cw_wf, inc_psr_term=True)

# full model is sum of components
model = ef + rn + tm + eq + cw

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