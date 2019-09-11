import numpy as np
import os,glob,json,pickle

import sim_gw as SG

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

scratch_dir = '/scratch/ark0015/background_injections'
#Where the everything should be saved to (chains,etc.)
simdir = scratch_dir + '/injection_combination' + injection_combination
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

#Save sim pulsars to a pickle file
with open(outdir + '/enterprise_pickled_psrs.pickle','wb') as psrfile:
    pickle.dump(sim.psrs,psrfile)
    psrfile.close()