import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import sys,os,glob,json,pickle
from collections import OrderedDict

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection

from enterprise_extensions import models,model_utils

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

def get_noise_from_pal2(noisefile):
    psrname = noisefile.split('/')[-1].split('_noise.txt')[0]
    fin = open(noisefile, 'r')
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if 'efac' in line:
            par = 'efac'
            flag = ln[0].split('efac-')[-1]
        elif 'equad' in line:
            par = 'log10_equad'
            flag = ln[0].split('equad-')[-1]
        elif 'jitter_q' in line:
            par = 'log10_ecorr'
            flag = ln[0].split('jitter_q-')[-1]
        elif 'RN-Amplitude' in line:
            par = 'red_noise_log10_A'
            flag = ''
        elif 'RN-spectral-index' in line:
            par = 'red_noise_gamma'
            flag = ''
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = '_'.join(name)
        params.update({pname: float(ln[1])})
    return params

#Where the enterprise 11yr datafiles are
topdir = os.getcwd()

runname = '/simGWB_3'
#Where the everything should be saved to (chains, cornerplts, histograms, etc.)
simdir = topdir + '/SimRuns'
outdir = simdir + runname
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

#noise files of 11yr set
noisepath = topdir + '/nano11/noisefiles_new/'
#psrs used for analysis
psrlistpath = topdir + '/nano11/psrlist_Tg3yr.txt'
#The pickled pulsars
psr_pickle_file = simdir + '/enterprise_sim_pulsars.pickle'

encoding = "utf-8"
psrlist_bytes = np.loadtxt(psrlistpath,dtype='S42')
psrlist = []
for psr in psrlist_bytes:
    psrlist.append(psr.decode(encoding))

#Load noisefiles to set pulsar white noise values
noisefiles = sorted(glob.glob(noisepath+'*.txt'))
noisefiles = [x for x in noisefiles if x.split('/')[-1].split('_')[0] in psrlist]

noise_params = {}
for nf in noisefiles:
    noise_params.update(get_noise_from_pal2(nf))

#Load pulsars from pickle file
with open(psr_pickle_file,'rb') as psrfile:
    psrs = pickle.load(psrfile)
    psrfile.close()


# find the maximum time span to set GW frequency sampling
selection = Selection(selections.by_backend)

tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

##### parameters and priors #####

# white noise parameters
'''
efac = parameter.Uniform(0.5,4.0)
log10_equad = parameter.Uniform(-10,-5)
log10_ecorr = parameter.Uniform(-10,-5)
'''
efac = parameter.Constant()
log10_equad = parameter.Constant()
log10_ecorr = parameter.Constant()

# red noise parameters
red_noise_log10_A = parameter.Uniform(-18,-13)
red_noise_gamma = parameter.Uniform(0,7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw_1 = parameter.Uniform(-18,-13)('zlog10_A_gw')
gamma_gw_1 = parameter.Constant(13/3)('zgamma_gw')

# Second GW parameters
log10_A_gw_2 = parameter.Uniform(-18,-13)('zlog10_A_other_gw')
gamma_gw_2 = parameter.Constant(7/3)('zgamma_other_gw')

##### Set up signals #####

# timing model
tm = gp_signals.TimingModel()

# white noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.EquadNoise(log10_equad = log10_equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr = log10_ecorr, selection=selection)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=red_noise_log10_A, gamma=red_noise_gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

cpl_1 = utils.powerlaw(log10_A=log10_A_gw_1, gamma=gamma_gw_1)
cpl_2 = utils.powerlaw(log10_A=log10_A_gw_2, gamma=gamma_gw_2)

#Common red noise process with no correlations
crn_1 = gp_signals.FourierBasisGP(spectrum = cpl_1, components=30, Tspan=Tspan, name = 'gw')
crn_2 = gp_signals.FourierBasisGP(spectrum = cpl_2, components=30, Tspan=Tspan, name = 'other_gw')

# gwb with Hellings and Downs correlations
# Hellings and Downs ORF
#orf = utils.hd_orf()
#gwb_1 = gp_signals.FourierBasisCommonGP(cpl_1, orf, components=30, name='gw', Tspan=Tspan)
#gwb_2 = gp_signals.FourierBasisCommonGP(cpl_2, orf, components=30, name='other_gw', Tspan=Tspan)


# full model is sum of components
model = ef + eq + ec + rn + tm + crn_1 + crn_2  #+ crn

# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])


with open(outdir + '/parameters.json', 'w') as fp:
    json.dump(pta.param_names, fp)


#Set Default PTA parameters to the ones in the noisefiles
pta.set_default_params(noise_params)


#Pick random initial sampling
xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

groups = model_utils.get_parameter_groups(pta)
groups.append([ndim-2,ndim-1])


# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir = outdir,resume=True)

#define your JP here
jp = model_utils.JumpProposal(pta)

sampler.addProposalToCycle(jp.draw_from_prior, 15)
sampler.addProposalToCycle(jp.draw_from_red_prior, 15)

# sampler for N steps
N = int(1e6)
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)