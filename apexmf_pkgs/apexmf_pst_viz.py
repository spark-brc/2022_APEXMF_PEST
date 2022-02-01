""" PEST support visualizations: 10/22/2020 created by Seonggyu Park
    last modified day: 10/22/2020 by Seonggyu Park
"""

from http.client import METHOD_NOT_ALLOWED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from hydroeval import evaluator, nse, rmse, pbias
import numpy as np

def str_df(rch_file, start_date, rch_num, obd_nam, time_step=None):
    
    if time_step is None:
        time_step = "D"
        strobd_file = "stf_day.obd"
    else:
        time_step = "M"
        strobd_file = "stf_mon.obd"
    output_rch = pd.read_csv(
                        rch_file, delim_whitespace=True, skiprows=9,
                        usecols=[0, 1, 8], names=["idx", "sub", "simulated"], index_col=0
                        )
    df = output_rch.loc["REACH"]
    str_obd = pd.read_csv(
                        strobd_file, sep=r'\s+', index_col=0, header=0,
                        parse_dates=True, delimiter="\t",
                        na_values=[-999, ""]
                        )
    # Get precipitation data from *.DYL
    prep_file = 'sub{}.DLY'.format(rch_num)
    with open(prep_file) as f:
        content = f.readlines()    
    year = content[0][:6].strip()
    mon = content[0][6:10].strip()
    day = content[0][10:14].strip()
    prep = [float(i[32:38].strip()) for i in content]
    prep_stdate = "/".join((mon,day,year))
    prep_df =  pd.DataFrame(prep, columns=['prep'])
    prep_df.index = pd.date_range(prep_stdate, periods=len(prep))
    prep_df = prep_df.replace(9999, np.nan)
    if time_step == "M":
        prep_df = prep_df.resample('M').mean()
    df = df.loc[df['sub'] == int(rch_num)]
    df = df.drop('sub', axis=1)
    df.index = pd.date_range(start_date, periods=len(df), freq=time_step)
    df = pd.concat([df, str_obd[obd_nam], prep_df], axis=1)
    plot_df = df[df['simulated'].notna()]
    return plot_df


def str_plot(plot_df):

    colnams = plot_df.columns.tolist()
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.grid(True)
    ax.plot(plot_df.index, plot_df.iloc[:, 0], label='Simulated', color='green', marker='^', alpha=0.7)
    ax.scatter(
        plot_df.index, plot_df.iloc[:, 1], label='Observed',
        # color='red',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        # zorder=2,
        )
    ax.plot(plot_df.index, plot_df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    ax2=ax.twinx()
    ax2.bar(
        plot_df.index, plot_df.prep, label='Precipitation',
        width=20,
        color="blue", align='center', alpha=0.5, zorder=0)
    ax2.set_ylabel("Precipitation $(mm)$",color="blue",fontsize=14)
    ax.set_ylabel("Stream Discharge $(m^3/day)$",fontsize=14)
    ax2.invert_yaxis()
    ax2.set_ylim(plot_df.prep.max()*3, 0)
    ax.margins(y=0.2)
    ax.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)    
    # add stats
    org_stat = plot_df.dropna()

    sim_org = org_stat.iloc[:, 0].to_numpy()
    obd_org = org_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim_org, obd_org)
    df_rmse = evaluator(rmse, sim_org, obd_org)
    df_pibas = evaluator(pbias, sim_org, obd_org)
    r_squared = (
        ((sum((obd_org - obd_org.mean())*(sim_org-sim_org.mean())))**2)/
        ((sum((obd_org - obd_org.mean())**2)* (sum((sim_org-sim_org.mean())**2))))
        )    
    ax.text(
        0.95, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(df_nse[0], df_rmse[0], df_pibas[0], r_squared),
        horizontalalignment='right',fontsize=10,
        bbox=dict(facecolor='green', alpha=0.5),
        transform=ax.transAxes
        )     
    fig.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    ax.legend(
        lines, labels, loc = 'lower left', ncol=5,
        # bbox_to_anchor=(0, 0.202),
        fontsize=12)
    # plt.legend()
    plt.show()
    return df_nse[0]


def get_stats(df):
    df_stat = df.dropna()

    sim = df_stat.iloc[:, 0].to_numpy()
    obd = df_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim, obd)
    df_rmse = evaluator(rmse, sim, obd)
    df_pibas = evaluator(pbias, sim, obd)
    r_squared = (
        ((sum((obd - obd.mean())*(sim-sim.mean())))**2)/
        ((sum((obd - obd.mean())**2)* (sum((sim-sim.mean())**2))))
        )
    return df_nse, df_rmse, df_pibas, r_squared


def str_plot_test(plot_df, cal_period=None, val_period=None):

    if cal_period:
        cal_df = plot_df[cal_period[0]:cal_period[1]]
    if val_period:
        val_df = plot_df[val_period[0]:val_period[1]]
    colnams = plot_df.columns.tolist()
    # plot
    fig, ax = plt.subplots(figsize=(16, 4))
    
    ax.grid(True)
    # cali
    ax.plot(cal_df.index, cal_df.iloc[:, 0], label='Calibrated', color='green', marker='^', alpha=0.7)
    ax.plot(val_df.index, val_df.iloc[:, 0], label='Validated', color='m', marker='x', alpha=0.7)
    ax.scatter(
        plot_df.index, plot_df.iloc[:, 1], label='Observed',
        # color='red',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        # zorder=2,
        )
    ax.plot(plot_df.index, plot_df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    ax2=ax.twinx()
    ax2.bar(
        plot_df.index, plot_df.prep, label='Precipitation',
        width=20,
        color="blue", align='center', alpha=0.5, zorder=0)
    ax2.set_ylabel("Precipitation $(mm)$",color="blue",fontsize=14)
    ax.set_ylabel("Stream Discharge $(m^3/s)$",fontsize=14)
    ax2.invert_yaxis()
    ax2.set_ylim(plot_df.prep.max()*3, 0)
    ax.margins(y=0.2)
    ax.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)    
    # add stats cal
    cal_nse, cal_rmse, cal_pbias, cal_rsquared = get_stats(cal_df)
    ax.text(
        0.63, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(cal_nse[0], cal_rmse[0], cal_pbias[0], cal_rsquared),
        horizontalalignment='right',fontsize=12,
        bbox=dict(facecolor='lightgreen', alpha=0.5),
        transform=ax.transAxes
        )
    # add stats val
    val_nse, val_rmse, val_pbias, val_rsquared = get_stats(val_df)
    ax.text(
        0.65, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(val_nse[0], val_rmse[0], val_pbias[0], val_rsquared),
        horizontalalignment='left',fontsize=12,
        bbox=dict(facecolor='m', alpha=0.5),
        transform=ax.transAxes
        )
    fig.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(
        lines+h2, labels+l2, loc = 'upper right', ncol=4,
        bbox_to_anchor=(1, 1.13),
        fontsize=12)
    # plt.legend()
    plt.savefig('temp.png', dpi=600, bbox_inches="tight")
    print(os.getcwd())
    
    plt.show()



def obds_df(strobd_file, wt_obd_file):
    str_obd = pd.read_csv(
                        strobd_file, sep=r'\s+', index_col=0, header=0,
                        parse_dates=True, delimiter="\t",
                        na_values=[-999, ""]
                        )
    wt_obd = pd.read_csv(
                        'MODFLOW/' + wt_obd_file, sep=r'\s+', index_col=0, header=0,
                        parse_dates=True, delimiter="\t",
                        na_values=[-999, ""]
                        )
    if strobd_file == 'streamflow_month.obd':
        str_obd = str_obd.resample('M').mean()
    if wt_obd_file == 'modflow_month.obd':
        wt_obd = wt_obd.resample('M').mean()

    df = pd.concat([str_obd, wt_obd], axis=1)
    return df
    

def wt_df(start_date, grid_id, obd_nam, time_step=None, prep_sub=None):
    
    if time_step is None:
        time_step = "D"
        mfobd_file = "modflow.obd"
    else:
        time_step = "M"
        mfobd_file = "modflow_month.obd."

    mf_obs = pd.read_csv(
                        "MODFLOW/modflow.obs",
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")

    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    output_wt = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)])
    output_wt.index = pd.date_range(start_date, periods=len(output_wt))

    if time_step == 'M':
        output_wt = output_wt.resample('M').mean()
    if prep_sub is not None:
        # Get precipitation data from *.DYL
        prep_file = 'sub{}.DLY'.format(prep_sub)
        with open(prep_file) as f:
            content = f.readlines()    
        year = content[0][:6].strip()
        mon = content[0][6:10].strip()
        day = content[0][10:14].strip()
        prep = [float(i[32:38].strip()) for i in content]
        prep_stdate = "/".join((mon,day,year))
        prep_df =  pd.DataFrame(prep, columns=['prep'])
        prep_df.index = pd.date_range(prep_stdate, periods=len(prep))
        prep_df = prep_df.replace(9999, np.nan)
        # if time_step == "M":
        prep_df = prep_df.resample('M').mean()
        output_wt = pd.concat([output_wt, mfobd_df[obd_nam], prep_df], axis=1)
    else:
        output_wt = pd.concat([output_wt, mfobd_df[obd_nam]], axis=1)
    output_wt = output_wt[output_wt[str(grid_id)].notna()]

    return output_wt        


def wt_df2(start_date, grid_id, obd_nam, time_step=None, prep_sub=None):
    
    if time_step is None:
        time_step = "D"
        mfobd_file = "modflow.obd"
    else:
        time_step = "M"
        mfobd_file = "modflow_month.obd."

    mf_obs = pd.read_csv(
                        "MODFLOW/modflow.obs",
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")


    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    output_wt = output_wt[str(grid_id)]
    output_wt.index = pd.date_range(start_date, periods=len(output_wt))
    mfobd_df = float(mf_obs.loc[int(grid_id)]) + mfobd_df[obd_nam] 

    if time_step == 'M':
        output_wt = output_wt.resample('M').mean()
    if prep_sub is not None:
        # Get precipitation data from *.DYL
        prep_file = 'sub{}.DLY'.format(prep_sub)
        with open(prep_file) as f:
            content = f.readlines()    
        year = content[0][:6].strip()
        mon = content[0][6:10].strip()
        day = content[0][10:14].strip()
        prep = [float(i[32:38].strip()) for i in content]
        prep_stdate = "/".join((mon,day,year))
        prep_df =  pd.DataFrame(prep, columns=['prep'])
        prep_df.index = pd.date_range(prep_stdate, periods=len(prep))
        prep_df = prep_df.replace(9999, np.nan)
        # if time_step == "M":
        prep_df = prep_df.resample('M').mean()
        output_wt = pd.concat([output_wt, mfobd_df, prep_df], axis=1)
    else:
        output_wt = pd.concat([output_wt, mfobd_df], axis=1)
    output_wt = output_wt[output_wt[str(grid_id)].notna()]

    return output_wt



def wt_plot(plot_df):

    colnams = plot_df.columns.tolist()
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.grid(True)
    ax.plot(plot_df.index, plot_df.iloc[:, 0], label='Simulated', color='green', marker='^', alpha=0.7)
    ax.scatter(
        plot_df.index, plot_df.iloc[:, 1], label='Observed',
        # color='red',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        # zorder=2,
        )
    ax.plot(plot_df.index, plot_df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    ax2=ax.twinx()
    ax2.bar(
        plot_df.index, plot_df.prep, label='Precipitation',
        width=20,
        color="blue", align='center', alpha=0.5, zorder=0)
    ax2.set_ylabel("Precipitation $(mm)$",color="blue",fontsize=14)
    ax.set_ylabel("Depth to Water $(m)$",fontsize=14)
    ax2.invert_yaxis()
    ax2.set_ylim(plot_df.prep.max()*3, 0)
    ax.margins(y=0.2)
    ax.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)    
    
    # add stats
    plot_df = plot_df.drop('prep', axis=1)
    org_stat = plot_df.dropna()

    sim_org = org_stat.iloc[:, 0].to_numpy()
    obd_org = org_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim_org, obd_org)
    df_rmse = evaluator(rmse, sim_org, obd_org)
    df_pibas = evaluator(pbias, sim_org, obd_org)
    r_squared = (
        ((sum((obd_org - obd_org.mean())*(sim_org - sim_org.mean())))**2)/
        ((sum((obd_org - obd_org.mean())**2)* (sum((sim_org - sim_org.mean())**2))))
        )      
    ax.text(
        0.95, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(df_nse[0], df_rmse[0], df_pibas[0], r_squared),
        horizontalalignment='right',fontsize=10,
        bbox=dict(facecolor='green', alpha=0.5),
        transform=ax.transAxes
        )  
    ax.set_title(colnams[0], loc='center', fontsize=12)   
    fig.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    ax.legend(
        lines, labels, loc = 'lower left', ncol=5,
        # bbox_to_anchor=(0, 0.202),
        fontsize=12)
    # plt.legend()
    plt.show()


def wt_tot_df(sim_start, df_start, df_end, grid_ids, obd_nams, time_step=None):
    """combine all groundwater outputs to provide a dataframe for 1 to 1 plot

    Args:
        start_date (str): simulation start date 
        grid_ids (list): list of grid ids used for plot
        obd_nams (list): list of column names in observed data and in accordance with grid ids
        time_step (str, optional): simulation time step (day, month, annual). Defaults to None.

    Returns:
        dataframe: dataframe for all simulated depth to water and observed data
    """
    if time_step is None:
        time_step = "D"
        mfobd_file = "modflow.obd"
    else:
        time_step = "M"
        mfobd_file = "modflow_month.obd."
    # read obs and obd files to get grid ids, elev, and observed values
    mf_obs = pd.read_csv(
                        "MODFLOW/modflow.obs",
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")
    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    # read simulated water elevation
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    # append data to big dataframe
    tot_df = pd.DataFrame()
    for grid_id, obd_nam in zip(grid_ids, obd_nams):
        df = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)]) # calculate depth to water
        df.index = pd.date_range(sim_start, periods=len(df))
        df = df[df_start:df_end]
        if time_step == 'M':
            df = df.resample('M').mean()
        df = pd.concat([df, mfobd_df[obd_nam]], axis=1) # concat sim with obd
        df = df.dropna() # drop nan
        new_cols ={x:y for x, y in zip(df.columns, ['sim', 'obd'])} #replace col nams with new nams
        tot_df = tot_df.append(df.rename(columns=new_cols))  
    return tot_df


def wt_tot_df2(sim_start, df_start, df_end, grid_ids, obd_nams, time_step=None):
    """combine all groundwater outputs to provide a dataframe for 1 to 1 plot

    Args:
        start_date (str): simulation start date 
        grid_ids (list): list of grid ids used for plot
        obd_nams (list): list of column names in observed data and in accordance with grid ids
        time_step (str, optional): simulation time step (day, month, annual). Defaults to None.

    Returns:
        dataframe: dataframe for all simulated depth to water and observed data
    """
    if time_step is None:
        time_step = "D"
        mfobd_file = "modflow.obd"
    else:
        time_step = "M"
        mfobd_file = "modflow_month.obd."
    # read obs and obd files to get grid ids, elev, and observed values
    mf_obs = pd.read_csv(
                        "MODFLOW/modflow.obs",
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")
    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    # read simulated water elevation
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    # append data to big dataframe

    tot_df = pd.DataFrame()
    for grid_id, obd_nam in zip(grid_ids, obd_nams):
        # df = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)]) # calculate depth to water
        df = output_wt[str(grid_id)]
        df.index = pd.date_range(sim_start, periods=len(df))
        df = df[df_start:df_end]
        if time_step == 'M':
            df = df.resample('M').mean()
        # mfobd_df = float(mf_obs.loc[int(grid_id)]) + mfobd_df[obd_nam]
        mfobd_dff = mfobd_df[obd_nam] + float(mf_obs.loc[int(grid_id)])
        df = pd.concat([df, mfobd_dff], axis=1) # concat sim with obd
        df = df.dropna() # drop nan
        new_cols ={x:y for x, y in zip(df.columns, ['sim', 'obd'])} #replace col nams with new nams
        tot_df = tot_df.append(df.rename(columns=new_cols))  
    return tot_df  


def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
    suffix  = ["G", "M", "k", "" , "m" , "u", "n"  ]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >=d:
            val = y/float(d)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                # return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
                return '{val:d}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    # print (val, signf)
                    if str(val).split(".")[1] == "0":
                    #    return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i])
                        return '{val:d}'.format(val=int(round(val)), suffix=suffix[i]) 
                tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                return tx.format(val=val, suffix=suffix[i])

                #return y
    return y


def dtw_sim_obd(st_date, ed_date, grid_id, time_step=None):
    
    if time_step is None:
        obd_file = 'dtw_day.obd'
    if time_step == 'month':
        obd_file = 'dtw_mon.obd'
    else:
        obd_file = 'dtw_day.obd'
    dtw_sim = pd.read_csv(
                        'wt_{}.txt'.format(grid_id),
                        sep=r'\s+',
                        index_col=0,
                        parse_dates=True,
                        names=['date', grid_id]
                        )
    dtw_obd = pd.read_csv(
                        'MODFLOW/' + obd_file,
                        sep='\t',
                        usecols=['date', 'wt{:05d}'.format(grid_id)],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, ""]
                        )
    df = pd.concat([dtw_sim, dtw_obd], axis=1)
    # NOTE: temp
    df['wt{:05d}'.format(grid_id)].fillna(0, inplace=True)
    #

    df = df[st_date:ed_date]
    return df

def dtw_hydrograph(df):
    fig, ax = plt.subplots()

    ax.plot(df.index, df.iloc[:, 0], label='Simulated', color='green', marker='^', alpha=0.7)
    ax.scatter(
        df.index, df.iloc[:, 1], label='Observed',
        # color='red',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        # zorder=2,
        )
    ax.plot(df.index, df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    cal_nse, cal_rmse, cal_pbias, cal_rsquared = get_stats(df)
    ax.text(
        1., 1.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(cal_nse[0], cal_rmse[0], cal_pbias[0], cal_rsquared),
        horizontalalignment='right',fontsize=12,
        bbox=dict(facecolor='green', alpha=0.5),
        transform=ax.transAxes
        )
    ax.text(
        0., 1.1,
        '{}'.format(df.columns[0]),
        horizontalalignment='left',fontsize=12,
        bbox=dict(facecolor='white', alpha=0.5),
        transform=ax.transAxes
        )
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    plt.savefig('dtw_{}.png'.format(df.columns[0]), dpi=150, bbox_inches="tight")
    print('saved dtw_{}.png ...'.format(df.columns[0]))
    # plt.show()


def phi_progress_plot(filename):
    rec_file = filename[:-3] + 'rec' 
    with open(rec_file, "r") as f:
        model_calls = []
        phis = []
        for line in f.readlines():
            if line.strip().startswith("Model calls so far"):
                model_calls.append(int(line.replace('\n', '').split()[5]))
            if line.strip().startswith("Starting phi for this iteration"):
                phis.append(float(line.replace('\n', '').split()[6]))
    
    df = pd.DataFrame({'Model Runs': model_calls, 'Phi': phis})
    df = df.set_index('Model Runs')
    df.plot(figsize=(5,5), grid=True)

if __name__ == '__main__':
    wd = "D:/Workshops/2022_Webinar_apexmf_opt_BLM/apexmf_1st_cal\APEX-MODFLOW"
    os.chdir(wd)
    phi_progress_plot('ani_pest_rw.pst')
