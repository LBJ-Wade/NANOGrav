from __future__ import division

import numpy as np
import glob, os, sys, pickle, json


from enterprise.pulsar import Pulsar


import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index("nanograv")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
noise_path = top_dir + "/pta_sim/pta_sim"
sys.path.insert(0, noise_path)
sys.path.insert(0, e_e_path)
import enterprise_extensions as e_e
from enterprise_extensions import models
from enterprise_extensions.sampler import JumpProposal
import noise

psrlist = ["J2317+1439"]
datadir = top_dir + "/5yr/NANOGrav_dfg+12_20120911"
# outdir = current_path + "/chains/" + psrlist[0] + "_red_var_white_fixed/"
outdir = current_path + "/chains/" + "messing_around/"

parfiles = sorted(glob.glob(datadir + "/par/*.par"))
timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))

noisefile = top_dir + "/5yr/noisefiles/{}_noise.txt".format(psrlist[0].split("J")[1])
tmpnoisedict = noise.get_noise_from_file(noisefile)
noisedict = {}
for key in tmpnoisedict.keys():
    noisedict["_".join(key.split("-"))] = tmpnoisedict[key]

# filter
parfiles = [
    x for x in parfiles if x.split("/")[-1].split(".")[0].split("_")[0] in psrlist
]
timfiles = [
    x for x in timfiles if x.split("/")[-1].split(".")[0].split("_")[0] in psrlist
]

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem="DE436", clk=None, drop_t2pulsar=False)
    psrs.append(psr)

tmparams_nodmx = []
for psr in psrs:
    for par in psr.fitpars:
        if "DMX" in ["".join(list(x)[0:3]) for x in par.split("_")][0]:
            pass
        elif "JUMP" in ["".join(list(x)[0:4]) for x in par.split("_")][0]:
            pass
        elif par == "Offset":
            pass
        else:
            tmparams_nodmx.append(par)
# tmparam_list = [ 'PB', 'A1', 'XDOT', 'EPS1', 'EPS2', 'EPS1DOT', 'EPS2DOT']
# tmparam_list = [ 'PB', 'A1', 'EPS1', 'EPS2']
tmparam_list = tmparams_nodmx
tmparam_list = ["F0", "F1"]
print("Sampling these values: ", tmparam_list)

"""noisefiles = sorted(glob.glob(top_dir + '/12p5yr/*.json'))
params = {}
for nf in noisefiles:
    with open(nf, 'r') as fin:
        params.update(json.load(fin))"""


pta = e_e.models.model_general(
    psrs,
    tm_var=True,
    tm_linear=False,
    tmparam_list=tmparam_list,
    common_psd="powerlaw",
    red_psd="powerlaw",
    orf=None,
    common_var=False,
    common_components=30,
    red_components=30,
    dm_components=30,
    modes=None,
    wgts=None,
    logfreq=False,
    nmodes_log=10,
    noisedict=noisedict,
    tm_svd=False,
    tm_norm=True,
    gamma_common=None,
    upper_limit=False,
    upper_limit_red=None,
    upper_limit_dm=None,
    upper_limit_common=None,
    bayesephem=False,
    be_type="orbel",
    wideband=False,
    dm_var=False,
    dm_type="gp",
    dm_psd="powerlaw",
    dm_annual=False,
    white_vary=False,
    gequad=False,
    dm_chrom=False,
    dmchrom_psd="powerlaw",
    dmchrom_idx=4,
    red_var=True,
    red_select=None,
    red_breakflat=False,
    red_breakflat_fq=None,
    coefficients=False,
)

# dimension of parameter space
params = pta.param_names
print(params)
ndim = len(params)
# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.1 ** 2)

# parameter groupings
groups = e_e.sampler.get_parameter_groups(pta)
tm_groups = e_e.sampler.get_timing_groups(pta)
for tm_group in tm_groups:
    groups.append(tm_group)

print('Before Sampler')
sampler = ptmcmc(
    ndim,
    pta.get_lnlikelihood,
    pta.get_lnprior,
    cov,
    groups=groups,
    outDir=outdir,
    resume=False,
)
print('After Sampler')
np.savetxt(outdir + "/pars.txt", list(map(str, pta.param_names)), fmt="%s")
np.savetxt(
    outdir + "/priors.txt",
    list(map(lambda x: str(x.__repr__()), pta.params)),
    fmt="%s",
)
# sampler = e_e.sampler.setup_sampler(
#    pta, outdir=outdir, resume=False, empirical_distr=None
# )
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_signal("timing_model"), 80)
for p in pta.params:
    for cat in ["pos", "pm", "spin", "kep", "gr"]:
        if cat in p.name.split("_"):
            sampler.addProposalToCycle(jp.draw_from_par_prior(p.name), 20)

# sampler for N steps
N = int(10)
x0 = np.hstack(p.sample() for p in pta.params)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
