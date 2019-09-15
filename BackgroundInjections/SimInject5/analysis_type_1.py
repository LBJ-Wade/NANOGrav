import numpy as np
import os,json,pickle

import sim_gw as SG

scratch_directory = '/scratch/ark0015/background_injections'
analysis_directory = scratch_directory + '/analysis_type_1'

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""
injection_combination_directory = scratch_directory + '/injection_combination_1'
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""
injection_combination_subdirectory = injection_combination_directory + '/simulation_1'
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN ANALYSES"""
runname = '/analysis_1'
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN ANALYSES"""

#Where the everything should be saved to (chains,etc.)
outdir = analysis_directory + runname
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
with open(scratch_dir + '/challenge1_psr_noise.json', 'rb') as fin:
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
N = int(1e5)
x0 = np.hstack(p.sample() for p in pta.params)
#sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

