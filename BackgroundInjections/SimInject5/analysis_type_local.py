import numpy as np
import sys,os,json,pickle

from enterprise_extensions import model_utils
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

current_path = os.getcwd()
splt_path = current_path.split("/")

top_path_idx = splt_path.index('nanograv')
top_dir = "/".join(splt_path[0:top_path_idx+1])

background_injection_dir = top_dir + '/NANOGrav/BackgroundInjections'
pta_sim_dir = top_dir + '/pta_sim/pta_sim'

sys.path.insert(0,pta_sim_dir)
import sim_gw as SG
import noise

noise_mdc2 =  top_dir + '/NANOGrav/MDC2/mdc2/group1/group1_psr_noise.json'

simdir = current_path + '/SimRuns'
injection_combination_subdirectory = simdir + '/simGWB_1'

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN ANALYSES"""
runname = '/analysis_1'
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN ANALYSES"""

#Where the everything should be saved to (chains,etc.)
outdir = injection_combination_subdirectory + runname
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

#Get the pickled pulsars
with open(injection_combination_subdirectory + '/enterprise_pickled_psrs.pickle', "rb") as f:
        psrs = pickle.load(f)

##### Use create pta analysis to instantiate enterprise PTA
#Fixed GWB power law
#Varied RN gamma and amplitude
background_gammas = [13./3.]
pta = SG.create_pta_analysis(psrs, gammas = background_gammas, psd='powerlaw', components=30, freqs=None,
                 upper_limit=False, bayesephem=False, select=None,
                 white_noise=True, red_noise=True, Tspan=None,orf=None)

# #### Save params for plotting
with open(outdir + '/sample_parameters.json', 'w') as fp:
    json.dump(pta.param_names, fp)

#Get Noise Values
with open(noise_mdc2, 'rb') as fin:
    noise_json =json.load(fin)

noiseparams = noise.handle_noise_parameters(noise_json)

#Set Fixed WN values
pta.set_default_params(noiseparams)

# #### Set up sampler and initial samples

#Pick random initial sampling
xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

groups = model_utils.get_parameter_groups(pta)
#groups.append([ndim-2,ndim-1])

# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir = outdir,resume=False)

# # Sample!
# sampler for N steps
N = int(1e3)
x0 = np.hstack(p.sample() for p in pta.params)
#sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

