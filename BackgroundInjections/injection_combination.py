import numpy as np
import os,glob,json,pickle,sys

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index('background_injections')
top_dir = "/".join(splt_path[0:top_path_idx+1])

sys.path.insert(0,top_dir)
import sim_gw as SG
import noise

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""
injection_combination = '_1'
#GWB Properties
background_amp_1 = 1.3e-15
background_gamma_1 = 13./3.
background_seed_1 = 1986
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""
runname = '/simulation_6'
#Red Noise Amplitude
background_amp_2 = 10**-14.5
background_gamma_2 = 3.0
background_seed_2 = 1667
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""

#Where the everything should be saved to (chains,etc.)
simdir = top_dir + '/injection_combination' + injection_combination
outdir = simdir + runname
if os.path.exists(simdir) == False:
    os.mkdir(simdir)
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

#The pulsars
#psrs_wn_only_dir = scratch_dir + '/FakePTA'
noise_file_mdc2 =  top_dir + '/group1_psr_noise.json'
psrs_mdc2 = top_dir + '/mdc2/'
#noise11yr_path = scratch_dir + '/nano11/noisefiles_new'
#psrlist11yr_path = scratch_dir + '/nano11/psrlist_Tg3yr.txt'


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

# #### Get par and tim files
parfiles = sorted(glob.glob(psrs_mdc2+'*.par'))
timfiles = sorted(glob.glob(psrs_mdc2+'*.tim'))
noise_params = noise.load_noise_files(noisepath=noise_file_mdc2)

# #### Instantiate a "Simulation class"
sim = SG.Simulation(parfiles,timfiles)

sim.add_white_noise(noise_params)

sim.createGWB(background_amp_1,gamma_gw=background_gamma_1,seed=background_seed_1)
sim.createGWB(background_amp_2,gamma_gw=background_gamma_2,seed=background_seed_2,noCorr=True)

# ### Get pulsars as enterprise pulsars
sim.init_ePulsars()

#Save sim pulsars to a pickle file
with open(outdir + '/enterprise_pickled_psrs.pickle','wb') as psrfile:
    pickle.dump(sim.psrs,psrfile)
    psrfile.close()
