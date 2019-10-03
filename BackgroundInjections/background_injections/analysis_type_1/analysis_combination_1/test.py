import numpy as np
import os,sys,json,pickle

current_path = os.getcwd()
splt_path = current_path.split("/")

top_path_idx = splt_path.index('background_injections')
top_dir = "/".join(splt_path[0:top_path_idx+1])

print(top_dir)

analysis_path_idx = splt_path.index('analysis_type_1')
analysis_directory = "/".join(splt_path[0:analysis_path_idx+1])

print(analysis_directory)
print(splt_path[-1])
print(splt_path[-1].split('_'))

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""
injection_combination_directory = top_dir + '/injection_combination_1'
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN COMBINATIONS"""

"""BELOW ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""
injection_combination_subdirectories = os.listdir(injection_combination_directory)
"""ABOVE ARE THE PARAMETERS TO CHANGE BETWEEN INJECTIONS"""


print(injection_combination_subdirectories.split('_'))

