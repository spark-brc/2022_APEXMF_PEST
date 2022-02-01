import os
from datetime import datetime
import pyemu
from apexmf_pst_utils import extract_month_str, extract_watertable_sim, extract_month_sed


wd = os.getcwd()
os.chdir(wd)
print(wd)

# file path
rch_file = 'SITE75.RCH'
# reach numbers that are used for calibration
subs = [12, 57, 75]
grid_ids = [5895, 6273]


time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  modifying parameters...')
print(30*'+ ' + '\n')

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  running model...')
print(30*'+ ' + '\n')
# pyemu.os_utils.run('SWAT-MODFLOW3.exe >_s+m.stdout', cwd='.')
pyemu.os_utils.run('APEX-MODFLOW_ani.exe', cwd='.')
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')

print('\n' + 35*'+ ')
print(time + ' | simulation successfully completed | extracting simulated values...')
print(35*'+ ' + '\n')
extract_month_str(rch_file, subs, '1/1/1980', '1/1/1992', '12/31/1999')
extract_month_sed(rch_file, subs, '1/1/1980', '1/1/1992', '12/31/1999')
extract_watertable_sim(grid_ids, '1/1/1980', '12/31/1999')
print(time + ' | Waiting for other workers to be completed ...')


