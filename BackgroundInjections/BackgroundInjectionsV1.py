import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import sys,os,glob,json
from collections import OrderedDict

import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP

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


# In[50]:


#Where the enterprise 11yr datafiles are
topdir = os.getcwd()

runname = '/simGWB_1'
#Where the everything should be saved to (chains, cornerplts, histograms, etc.)
outdir = topdir + '/SimRuns' + runname
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

parpath = topdir + '/nano11/partim_new/'
timpath = topdir + '/nano11/partim_new/'
noisepath = topdir + '/nano11/noisefiles_new/'
psrlistpath = topdir + '/nano11/psrlist_Tg3yr.txt'


J1909_par = parpath + 'J1909-3744' + '_NANOGrav_11yv0.gls.par'
J1909_tim = timpath + 'J1909-3744' + '_NANOGrav_11yv0.tim'


J1909 = T2.tempopulsar(parfile = J1909_par, timfile = J1909_tim, maxobs=30000, ephem='DE436',clk=None)


t = np.arange(53000,56650,30.0) #observing dates for 10 years
t += np.random.randn(len(t)) #observe every 30+/-1 days



J0613_par = parpath +'J0613-0200' + '_NANOGrav_11yv0.gls.par'
fake_J0613=LT.fakepulsar(J0613_par,obstimes=t,toaerr=0.5)



encoding = "utf-8"
psrlist_bytes = np.loadtxt(psrlistpath,dtype='S42')
psrlist = []
for psr in psrlist_bytes:
    psrlist.append(psr.decode(encoding))


parfiles = sorted(glob.glob(parpath+'*.par'))
timfiles = sorted(glob.glob(timpath+'*.tim'))
noisefiles = sorted(glob.glob(noisepath+'*.txt'))

parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0].split('_')[0] in psrlist]
timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0].split('_')[0] in psrlist]
noisefiles = [x for x in noisefiles if x.split('/')[-1].split('_')[0] in psrlist]

print(len(parfiles),len(timfiles),len(noisefiles))


#######################################
# PASSING THROUGH TEMPO2 VIA libstempo
#######################################

t2psr = []
for ii in range(len(parfiles)):
    
    t2psr.append( T2.tempopulsar(parfile = parfiles[ii], timfile = timfiles[ii],
                                 maxobs=30000, ephem='DE436') )
    
    if np.any(np.isfinite(t2psr[ii].residuals())==False)==True:
        t2psr[ii] = T2.tempopulsar(parfile = parfiles[ii], timfile = timfiles[ii])
                 

    print('\r{0} of {1}'.format(ii+1,len(parfiles)),flush=True,end='')


noise_params = {}
for nf in noisefiles:
    noise_params.update(get_noise_from_pal2(nf))


#Now parse this large dictionary so that we can call noise parameters as noise_dict[pulsar name][noise type]
#Returns either floats or 2 column arrays of flags and values. 

noise_dict = {}
for p in psrlist:
    noise_dict[p]={}
    noise_dict[p]['equads'] = []
    noise_dict[p]['efacs'] = []
    noise_dict[p]['ecorrs'] = []
    for ky in list(noise_params.keys()):
        if p in ky:
            if 'equad' in ky:
                noise_dict[p]['equads'].append([ky.replace(p + '_' , ''), noise_params[ky]])
            if 'efac' in ky:
                noise_dict[p]['efacs'].append([ky.replace(p + '_' , ''), noise_params[ky]])
            if 'ecorr' in ky:
                noise_dict[p]['ecorrs'].append([ky.replace(p + '_' , ''), noise_params[ky]])
            if 'gamma' in ky:
                noise_dict[p]['RN_gamma'] = noise_params[ky]
            if 'log10_A' in ky:
                noise_dict[p]['RN_Amp'] = 10**noise_params[ky]
                
    noise_dict[p]['equads'] = np.array(noise_dict[p]['equads'])
    noise_dict[p]['efacs'] = np.array(noise_dict[p]['efacs'])
    noise_dict[p]['ecorrs'] = np.array(noise_dict[p]['ecorrs'])    
    
    if len(noise_dict[p]['ecorrs'])==0: #Easier to just delete these dictionary items if no ECORR values. 
        noise_dict[p].__delitem__('ecorrs')


#By using seeds we can  reproduce the dataset if need be. 
seed_efac = 1066
seed_equad = 1492
seed_jitter = 1776
seed_red = 1987
seed_gwb_1 = 1667
seed_gwb_2 = 1980


for ii,p in enumerate(t2psr):

    ## make ideal
    LT.make_ideal(p)

    ## add efacs
    LT.add_efac(p, efac = noise_dict[p.name]['efacs'][:,1], 
                flagid = 'f', flags = noise_dict[p.name]['efacs'][:,0], 
                seed = seed_efac + ii)

    ## add equads
    LT.add_equad(p, equad = noise_dict[p.name]['equads'][:,1], 
                 flagid = 'f', flags = noise_dict[p.name]['equads'][:,0], 
                 seed = seed_equad + ii)

    ## add jitter
    try: #Only NANOGrav Pulsars have ECORR
        LT.add_jitter(p, ecorr = noise_dict[p.name]['ecorrs'][:,1], 
                      flagid='f', flags = noise_dict[p.name]['ecorrs'][:,0], 
                      coarsegrain = 1.0/86400.0, seed=seed_jitter + ii)
    except KeyError:
        pass

    ## add red noise
    LT.add_rednoise(p, noise_dict[p.name]['RN_Amp'], noise_dict[p.name]['RN_gamma'], 
                    components = 30, seed = seed_red + ii)
    
    print(ii, p.name)


# Create GWB
# Takes a list of libstempo pulsar objects as input.
LT.createGWB(t2psr, Amp=1.5e-15, gam=13./3., seed=seed_gwb_1)
LT.createGWB(t2psr, Amp=9.0e-15, gam=10./3., seed=seed_gwb_2)


psrs = []
for p in t2psr:
    psrs.append(Pulsar(p))


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
log10_A_gw_1 = parameter.Uniform(-18,-13)('zlog10_A_gw_1')
gamma_gw_1 = parameter.Constant(13/3)('zgamma_gw_1')

# Second GW parameters
log10_A_gw_2 = parameter.Uniform(-18,-13)('zlog10_A_gw_2')
gamma_gw_2 = parameter.Constant(10/3)('zgamma_gw_2')

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

cpl_1 = utils.powerlaw(log10_A=log10_A_gw_1, gamma=gamma_gw_2)
cpl_2 = utils.powerlaw(log10_A=log10_A_gw_2, gamma=gamma_gw_2)

#Common red noise process with no correlations
#crn = gp_signals.FourierBasisGP(spectrum = cpl, components=30, Tspan=Tspan, name = 'gw')

# gwb with Hellings and Downs correlations
# Hellings and Downs ORF
orf = utils.hd_orf()
gwb_1 = gp_signals.FourierBasisCommonGP(cpl_1, orf, components=30, name='gw_1', Tspan=Tspan)
gwb_2 = gp_signals.FourierBasisCommonGP(cpl_2, orf, components=30, name='gw_2', Tspan=Tspan)

# full model is sum of components
model = ef + eq + ec + rn + tm + gwb_1 + gwb_2  #+ crn

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

# sampler for N steps
N = int(1e5)
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

