import os
import shutil
import glob
from datetime import datetime
import pandas as pd
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT


wd = os.getcwd()
os.chdir(wd)

def create_riv_par(wd, chns, chg_type=None, rivcd=None, rivbot=None, val=None):
    os.chdir(wd)
    if rivcd is None:
        rivcd =  ['rivcd_{}'.format(x) for x in chns]
    if rivbot is None:
        rivbot =  ['rivbot_{}'.format(x) for x in chns]
    if chg_type is None:
        chg_type = 'unfchg'
    if val is None:
        val = 0.001
    riv_f = rivcd + rivbot
    df = pd.DataFrame()
    df['parnme'] = riv_f
    df['chg_type'] = chg_type
    df['val'] = val
    df.index = df.parnme
    with open('mf_riv.par', 'w') as f:
        f.write("# modflow_par file.\n")
        f.write("NAME   CHG_TYPE    VAL\n")
        f.write(
            df.loc[:, ["parnme", "chg_type", "val"]].to_string(
                                                        col_space=0,
                                                        formatters=[SFMT, SFMT, SFMT],
                                                        index=False,
                                                        header=False,
                                                        justify="left")
            )
    print("'mf_riv.par' file has been exported to the MODFLOW working directory!")
    return df


def read_modflow_par(wd):
    os.chdir(wd)
    # read mf_riv_par.par
    riv_pars = pd.read_csv('mf_riv.par', sep=r'\s+', comment='#', index_col=0)
    # get parameter types and channel number
    riv_pars['par_type'] = [x.split('_')[0] for x in riv_pars.index]
    riv_pars['chn_no'] = [x.split('_')[1] for x in riv_pars.index]
    return riv_pars


def riv_par(wd):
    """change river parameters in *.riv file (river package).

    Args:
        - wd (`str`): the path and name of the existing output file
    Reqs:
        - 'modflow.par'
    Opts:
        - 'riv_package.org'
    Vars:
        - pctchg: provides relative percentage changes
        - absval: provides absolute values
        - unfchg: provides uniform changes
    """

    os.chdir(wd)
    riv_files = [f for f in glob.glob(wd + "/*.riv")]
    if len(riv_files) == 1:
        riv_f = os.path.basename(riv_files[0])
        # duplicate original riv file
        if not os.path.exists('riv_package.org'):
            shutil.copy(riv_f, 'riv_package.org')
            print('The original river package "{}" has been backed up...'.format(riv_f))
        else:
            print('The "riv_package.org" file already exists...')

        with open('riv_package.org') as f:
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()

        # read riv pacakge
        df_riv = pd.read_csv('riv_package.org', sep=r'\s+', skiprows=3, header=None)

        # read mf_riv_par.par
        riv_pars = read_modflow_par(wd)

        # Select rows based on channel number
        for i in range(len(riv_pars)):
            if riv_pars.iloc[i, 2] == 'rivcd':
                subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivcd = subdf.iloc[:, 4] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivcd = subdf.iloc[:, 4] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 4] = float(riv_pars.iloc[i, 1])
                    new_rivcd = subdf.iloc[:, 4]
                count = 0
                for j in range(len(df_riv)):
                    if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                        df_riv.iloc[j, 4] = new_rivcd.iloc[count]
                        count += 1
            elif riv_pars.iloc[i, 2] == 'rivbot':
                subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivbot = subdf.iloc[:, 5] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivbot = subdf.iloc[:, 5] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 5] = float(riv_pars.iloc[i, 1])
                    new_rivbot = subdf.iloc[:, 5]
                count = 0
                for j in range(len(df_riv)):
                    if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                        df_riv.iloc[j, 5] = new_rivbot.iloc[count]
                        count += 1

        df_riv.iloc[:, 4] = df_riv.iloc[:, 4].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 3] = df_riv.iloc[:, 3].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 5] = df_riv.iloc[:, 5].map(lambda x: '{:.10e}'.format(x))

        # ------------ Export Data to file -------------- #
        version = "version 1.2."
        time = datetime.now().strftime('- %m/%d/%y %H:%M:%S -')

        with open(riv_f, 'w') as f:
            f.write("# RIV: River package file is parameterized. " + version + time + "\n")
            f.write(line1)
            f.write(line2)
            f.write(line3)
            df_riv.to_csv(
                        f, sep='\t',
                        header=False,
                        index=False,
                        line_terminator='\n',
                        encoding='utf-8'
                        )
        print(os.path.basename(riv_f) + " file is overwritten successfully!")

    elif len(riv_files) > 1:
        print(
                "You have more than one River Package file!("+str(len(riv_files))+")"+"\n"
                + str(riv_files)+"\n"
                + "Solution: Keep only one file!")
    else:
        print("File Not Found! - We couldn't find your *.riv file.")


def riv_par_more_detail(wd):
    """change river parameters in *.riv file (river package).

    Args:
        - wd (`str`): the path and name of the existing output file
    Reqs:
        - 'modflow.par'
    Opts:
        - 'riv_package.org'
    Vars:
        - pctchg: provides relative percentage changes
        - absval: provides absolute values
        - unfchg: provides uniform changes
    """

    os.chdir(wd)
    riv_files = [f for f in glob.glob(wd + "/*.riv")]
    if len(riv_files) == 1:
        riv_f = os.path.basename(riv_files[0])
        # duplicate original riv file
        if not os.path.exists('riv_package.org'):
            shutil.copy(riv_f, 'riv_package.org')
            print('The original river package "{}" has been backed up...'.format(riv_f))
        else:
            print('The "riv_package.org" file already exists...')

        with open('riv_package.org') as f:
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()

        # read riv pacakge
        df_riv = pd.read_csv('riv_package.org', sep=r'\s+', skiprows=3, header=None)

        # read mf_riv_par.par
        riv_pars = read_modflow_par(wd)

        ''' this block used for whole change in river parameter
        # Change river conductance
        if riv_pars.loc['riv_cond', 'CHG_TYPE'].lower() == 'pctchg':
            riv_cond_v = riv_pars.loc['riv_cond', 'VAL']
            df_riv.iloc[:, 4] = df_riv.iloc[:, 4] + (df_riv.iloc[:, 4] * riv_cond_v / 100)
        else:
            riv_cond_v = riv_pars.loc['riv_cond', 'VAL']
            df_riv.iloc[:, 4] = riv_cond_v

        # Change river bottom elevation
        riv_bot_v = riv_pars.loc['riv_bot', 'VAL']
        if riv_pars.loc['riv_bot', 'CHG_TYPE'].lower() == 'pctchg':
            df_riv.iloc[:, 5] = df_riv.iloc[:, 5] + (df_riv.iloc[:, 5] * riv_bot_v / 100)
        elif riv_pars.loc['riv_bot', 'CHG_TYPE'].lower() == 'absval':
            df_riv.iloc[:, 5] = riv_bot_v
        else:
            df_riv.iloc[:, 5] = df_riv.iloc[:, 5] + riv_bot_v

        df_riv.iloc[:, 4] = df_riv.iloc[:, 4].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 3] = df_riv.iloc[:, 3].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 5] = df_riv.iloc[:, 5].map(lambda x: '{:.10e}'.format(x))
        '''
        # Select rows based on channel number
        for i in range(len(riv_pars)):
            if riv_pars.iloc[i, 2] == 'rivcd':
                if riv_pars.iloc[i, 3][0] != 'g':
                    subdf = df_riv[df_riv.iloc[:, -3] == int(riv_pars.iloc[i, 3])]
                else:
                    subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivcd = subdf.iloc[:, 4] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivcd = subdf.iloc[:, 4] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 4] = float(riv_pars.iloc[i, 1])
                    new_rivcd = subdf.iloc[:, 4]
                if riv_pars.iloc[i, 3][0] != 'g':         
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -3] == int(riv_pars.iloc[i, 3]):
                            df_riv.iloc[j, 4] = new_rivcd.iloc[count]
                            count += 1
                else:
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                            df_riv.iloc[j, 4] = new_rivcd.iloc[count]
                            count += 1                
            elif riv_pars.iloc[i, 2] == 'rivbot':
                if riv_pars.iloc[i, 3][0] != 'g':
                    subdf = df_riv[df_riv.iloc[:, -3] == int(riv_pars.iloc[i, 3])]
                else:
                    subdf = df_riv[df_riv.iloc[:, -1] == riv_pars.iloc[i, 3]]
                if riv_pars.iloc[i, 0] == 'pctchg':
                    new_rivbot = subdf.iloc[:, 5] + (subdf.iloc[:, 4] * float(riv_pars.iloc[i, 1]) / 100)
                elif riv_pars.iloc[i, 0] == 'unfchg':
                    new_rivbot = subdf.iloc[:, 5] + float(riv_pars.iloc[i, 1])
                else:
                    subdf.iloc[:, 5] = float(riv_pars.iloc[i, 1])
                    new_rivbot = subdf.iloc[:, 5]
                if riv_pars.iloc[i, 3][0] != 'g':
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -3] == int(riv_pars.iloc[i, 3]):
                            df_riv.iloc[j, 5] = new_rivbot.iloc[count]
                            count += 1
                else:
                    count = 0
                    for j in range(len(df_riv)):
                        if df_riv.iloc[j, -1] == riv_pars.iloc[i, 3]:
                            df_riv.iloc[j, 5] = new_rivbot.iloc[count]
                            count += 1
                            
        df_riv.iloc[:, 4] = df_riv.iloc[:, 4].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 3] = df_riv.iloc[:, 3].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, 5] = df_riv.iloc[:, 5].map(lambda x: '{:.10e}'.format(x))
        df_riv.iloc[:, -2] = df_riv.iloc[:, -2].map(lambda x: '{:.10e}'.format(x))

        # ------------ Export Data to file -------------- #
        version = "version 1.2."
        time = datetime.now().strftime('- %m/%d/%y %H:%M:%S -')

        with open(riv_f, 'w') as f:
            f.write("# RIV: River package file is parameterized. " + version + time + "\n")
            f.write(line1)
            f.write(line2)
            f.write(line3)
            df_riv.to_csv(
                        f, sep='\t',
                        header=False,
                        index=False,
                        line_terminator='\n',
                        encoding='utf-8'
                        )
        print(os.path.basename(riv_f) + " file is overwritten successfully!")

    elif len(riv_files) > 1:
        print(
                "You have more than one River Package file!("+str(len(riv_files))+")"+"\n"
                + str(riv_files)+"\n"
                + "Solution: Keep only one file!")
    else:
        print("File Not Found! - We couldn't find your *.riv file.")


