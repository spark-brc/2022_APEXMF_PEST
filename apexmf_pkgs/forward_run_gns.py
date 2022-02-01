import os
from datetime import datetime
import pyemu
from apexmf_pst_utils import extract_month_str, extract_watertable_sim, extract_month_sed
from apexmf_pst_par import riv_par
import subprocess
import numpy as np
import pandas as pd

# Set path to working directory
wd = os.getcwd()
os.chdir(wd)
print(wd)
mf_wd = wd + "\MODFLOW"


# file path
rch_file = 'SITE64.RCH'

# reach numbers that are used for calibration
subs = [21, 26, 64, 67, 84, 111]
# Get MODFLOW grids used for calibration
# get cal only
grid_ids = [8489]

# modify river parameters
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  modifying river parameters...')
print(30*'+ ' + '\n')
# os.chdir(riv_wd)
# riv_par(mf_wd)

os.chdir(mf_wd)
# modify MODFLOW parameters
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print(time + ' |  modifying MODFLOW HK, VHK, and SY parameters...')
data_fac = ['hk0pp.dat', 'sy0pp.dat']
for i in data_fac:
    outfile = i + '.ref'
    pyemu.utils.geostats.fac2real(i, factors_file=i+'.fac', out_file=outfile)
    # # Create vertical k file
    # if i[:2] == 'hk':
    #     vk = np.loadtxt(outfile)
    #     np.savetxt('v' + outfile, vk/10, fmt='%.12e', delimiter='\t')

# Run model
os.chdir(wd)
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  running model...')
print(30*'+ ' + '\n')
# pyemu.os_utils.run('SWAT-MODFLOW3.exe >_s+m.stdout', cwd='.')
p = subprocess.Popen('APEX1501.exe' , cwd = '.')
p.wait()
# pyemu.os_utils.run('APEX-MODFLOW.exe', cwd=wd)

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')

print('\n' + 50*'+ ')
print(time + ' | simulation successfully completed | extracting simulated values...')
print(50*'+ ' + '\n')
extract_month_str(rch_file, subs, '1/1/2007', '1/1/2011', '12/31/2017')
extract_watertable_sim(grid_ids, '1/1/2007', '12/31/2017')
print(time + ' | Complete ...')


