import numpy as np
import sys,os,glob,json,pickle
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
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import sim_gw as SG

scratch_dir = '/scratch/ark0015/background_injections'

runname = '/simGWB_1'
#Where the everything should be saved to (chains,etc.)
simdir = scratch_dir + '/SimRuns'
outdir = simdir + runname
if os.path.exists(simdir) == False:
    os.mkdir(simdir)
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

#The pulsars
psrs_wn_only_dir = scratch_dir + '/FakePTA'
#noise11yr_path = scratch_dir + '/nano11/noisefiles_new'
#psrlist11yr_path = scratch_dir + '/nano11/psrlist_Tg3yr.txt'



# #### Get par and tim files
parfiles = sorted(glob.glob(psrs_wn_only_dir+'*.par'))
timfiles = sorted(glob.glob(psrs_wn_only_dir+'*.tim'))


# #### Instantiate a "Simulation class"
sim = SG.Simulation(parfiles,timfiles)


# #### Inject 2 backgrounds
background_amp_1 = 1.3e-15
background_amp_2 = 5.0e-15

background_gamma_1 = 13./3.
background_gamma_2 = 7./3.

background_seed_1 = 1986
background_seed_2 = 1667

#Save Injection Parameters
injection_parameters = {}
injection_parameters['Background_1'] = {'log_10_amp':np.log10(background_amp_1),\
                                        'gamma':background_gamma_1,\
                                        'seed':background_seed_1}
injection_parameters['Background_2'] = {'log_10_amp':np.log10(background_amp_2),\
                                        'gamma':background_gamma_2,\
                                        'seed':background_seed_2}
with open(outdir + '/injection_parameters.json', 'w') as fp:
    json.dump(injection_parameters, fp, sort_keys=True)

sim.createGWB(background_amp_1,gamma_gw=background_gamma_1,seed=background_seed_1)
sim.createGWB(background_amp_2,gamma_gw=background_gamma_2,seed=background_seed_2,noCorr=True)


# ### Get pulsars as enterprise pulsars
sim.init_ePulsars()


# #### Use Simple 2 GWB model to instantiate enterprise PTA
background_gammas = [background_gamma_1, background_gamma_2]
pta = SG.model_simple_multiple_gwbs(sim.psrs,gammas=background_gammas)

# #### Save params for plotting
with open(outdir + '/sample_parameters.json', 'w') as fp:
    json.dump(pta.param_names, fp)


# #### Set up sampler and initial samples

#Pick random initial sampling
xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

groups = model_utils.get_parameter_groups(pta)
groups.append([ndim-2,ndim-1])

# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir = outdir,resume=False)


# # Sample!
# sampler for N steps
N = int(1e2)
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

