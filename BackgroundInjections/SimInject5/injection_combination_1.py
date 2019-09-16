import numpy as np
import sys,os,glob,json,pickle

#import sim_gw as SG

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""
injection_combination = '_1'
#GWB Properties
background_amp_1 = 1.3e-15
background_gamma_1 = 13./3.
background_seed_1 = 1986
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""
runname = '/simulation_1'
#Red Noise Amplitude
background_amp_2 = 5.0e-15
background_gamma_2 = 7./3.
background_seed_2 = 1667
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index('nanograv')
top_dir = "/".join(splt_path[0:top_path_idx+1])

background_injection_dir = top_dir + '/NANOGrav/BackgroundInjections'
pta_sim_dir = top_dir + '/pta_sim/pta_sim'

sys.path.insert(0,pta_sim_dir)
import sim_gw as SG
import noise

scratch_dir = '/scratch/ark0015/background_injections'

runname = '/simGWB_1'
#Where the everything should be saved to (chains,etc.)
simdir = current_path + '/SimRuns'
outdir = simdir + runname
if os.path.exists(simdir) == False:
    os.mkdir(simdir)
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

#The pulsars
#psrs_wn_only_dir = background_injection_dir + '/FakePTA/'
noise_mdc2 =  top_dir + '/NANOGrav/MDC2/mdc2/group1/group1_psr_noise.json'
mdc2_psrs = top_dir + '/NANOGrav/MDC2/mdc2/group1/dataset_1a/'
#noise11yr_path = background_injection_dir + '/nano11/noisefiles_new/'
#psrlist11yr_path = backgrouninjection_dir + '/nano11/psrlist_Tg3yr.txt'



# #### Get par and tim files
parfiles = sorted(glob.glob(mdc2_psrs+'*.par'))
timfiles = sorted(glob.glob(mdc2_psrs+'*.tim'))
"""
noisefiles = sorted(glob.glob(noise11yr_path+'*.txt'))

noise_params = {}
for nf in noisefiles:
    psrname = nf.split('/')[-1].split('_noise.txt')[0]
    noise_params[psrname] = {}
    params = noise.get_noise_from_file(nf)
    noise_params[psrname].update(params)

with open(outdir + '/noise_parameters.json', 'w') as fp:
    json.dump(noise_params, fp, sort_keys=True,indent=4)
"""
noise_params = noise.load_noise_files(noisepath=noise_mdc2)


##### Instantiate a "Simulation class"
sim = SG.Simulation(parfiles,timfiles)

#Save Injection Parameters
injection_parameters = {}
injection_parameters['Background_1'] = {'log_10_amp':np.log10(background_amp_1),\
                                        'gamma':background_gamma_1,\
                                        'seed':background_seed_1}
injection_parameters['Background_2'] = {'log_10_amp':np.log10(background_amp_2),\
                                        'gamma':background_gamma_2,\
                                        'seed':background_seed_2}

with open(outdir + '/injection_parameters.json', 'w') as fp:
    json.dump(injection_parameters, fp, sort_keys=True,indent=4)

sim.add_white_noise(noise_params)

sim.createGWB(background_amp_1,gamma_gw=background_gamma_1,seed=background_seed_1)
sim.createGWB(background_amp_2,gamma_gw=background_gamma_2,seed=background_seed_2,noCorr=True)


# ### Get pulsars as enterprise pulsars
sim.init_ePulsars()

#Save sim pulsars to a pickle file
with open(outdir + '/enterprise_pickled_psrs.pickle','wb') as psrfile:
    pickle.dump(sim.psrs,psrfile)
    psrfile.close()