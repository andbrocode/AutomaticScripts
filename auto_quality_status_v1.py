#!/usr/bin/env python
# coding: utf-8
#
# # ROMY Status File
#
# ___________________________________________________

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy as obs
import matplotlib.colors

from pandas import DataFrame
from andbro__save_to_pickle import __save_to_pickle

#import matplotlib
#matplotlib.use('Agg')

# ___________________________________________________

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'
elif os.uname().nodename == 'lin-ffb-01':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'

# ___________________________________________________
# ## Configurations

config = {}

# extract ring
config['ring'] = sys.argv[1]

# extract time interval
config['tbeg'] = obs.UTCDateTime(sys.argv[2])
config['tend'] = obs.UTCDateTime(sys.argv[2]) +86400

# specify if it has differential transmission with signal combination
# for common mode rejection
config['commonmode'] = {"U":False, "V":False, "W":False, "Z":True}

# specify path to data
config['path_to_autodata'] = archive_path+f"romy_autodata/"

# specify path to output figures
config['path_to_figures'] = archive_path+f"romy_plots/{config['tbeg'].year}/R{config['ring']}/status/"

# specify path to output data
config['path_to_output'] = archive_path+f"temp_archive/{config['tbeg'].year}/BW/R{config['ring']}/"

# set nominal sagnac frequency of rings
config['fsagnac_rings'] = {"U":302.5, "V":447.7, "W":447.7, "Z":553.5}
config['fsagnac_nominal'] = config['fsagnac_rings'][config['ring']]

# set threshold for DC
config['DC_threshold'] = 0.1

# set threshold for AC
config['AC_threshold'] = 0.15

# set threshold for Sagnac frequency deviation
config['delta_fsagnac'] = 2.0

# ___________________________________________________

def __get_mlti_intervals(mlti_times, time_delta=60):

    from obspy import UTCDateTime
    from numpy import array

    if len(mlti_times) == 0:
        return array([]), array([])

    t1, t2 = [], []
    for k, _t in enumerate(mlti_times):

        _t = UTCDateTime(_t)

        if k == 0:
            _tlast = _t
            t1.append(UTCDateTime(str(_t)[:16]))

        if _t -_tlast > time_delta:
            t2.append(UTCDateTime(str(_tlast)[:16])+60)
            t1.append(UTCDateTime(str(_t)[:16]))

        _tlast = _t

    t2.append(UTCDateTime(str(_t)[:16])+60)
    # t2.append(mlti_times[-1])

    return array(t1), array(t2)

def __load_mlti(tbeg, tend, ring, path_to_archive):

    from obspy import UTCDateTime
    from pandas import read_csv, concat

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    rings = {"U":"03", "Z":"01", "V":"02", "W":"04"}

    if tbeg.year == tend.year:
        year = tbeg.year

        path_to_mlti = path_to_archive+f"romy_archive/{year}/BW/CROMY/{year}_romy_{rings[ring]}_mlti.log"

        mlti = read_csv(path_to_mlti, names=["time_utc","Action","ERROR"])

    else:

        path_to_mlti1 = path_to_archive+f"romy_archive/{tbeg.year}/BW/CROMY/{tbeg.year}_romy_{rings[ring]}_mlti.log"
        mlti1 = read_csv(path_to_mlti1, names=["time_utc","Action","ERROR"])

        path_to_mlti2 = path_to_archive+f"romy_archive/{tend.year}/BW/CROMY/{tend.year}_romy_{rings[ring]}_mlti.log"
        mlti2 = read_csv(path_to_mlti2, names=["time_utc","Action","ERROR"])

        mlti = concat([mlti1, mlti2])

    mlti = mlti[(mlti.time_utc > tbeg) & (mlti.time_utc < tend)]

    return mlti

def __load_beat(tbeg, tend, ring, path_to_data):

    from datetime import date
    from pandas import read_pickle, concat, DataFrame, date_range
    from obspy import UTCDateTime

    # convert dates
    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    dd1 = date.fromisoformat(str(tbeg.date))
    dd2 = date.fromisoformat(str(tend.date))

    df = DataFrame()
    for dat in date_range(dd1, dd2):
    
        # define file name
        filename = f"{str(dat)[:4]}/R{ring}/FJ{ring}_"+str(dat)[:10].replace("-", "")+".pkl"

        # loading data
        try:
            df0 = read_pickle(path_to_data+filename)
            df = concat([df, df0])
        except:
            print(f"error for {file}")

    # check for empty DataFrame
    if df.empty:
        print(" -> empty dataframe!")
        return df

    ## trim to defined times
    df = df[(df.times_utc >= tbeg) & (df.times_utc < tend)]

    ## correct seconds
    df['times_utc_sec'] = [abs(tbeg - UTCDateTime(_t))  for _t in df['times_utc']]

    return df

def main(config):

    # _____________________________________
    # ### Load MLTI Logs

    try:
        mlti_log = __load_mlti(config['tbeg'], config['tend'], config['ring'], archive_path)
    except:
        print(f"no MLTI log: {config['tbeg']}")
        quit()

    try:
        mlti_t1, mlti_t2 = __get_mlti_intervals(mlti_log.time_utc, time_delta=100)
    except:
        print("mlti intervals failed!")
        mlti_t1, mlti_t2 = np.array([]), np.array([])

    # _____________________________________
    # ### Load Beat Data
    print(f"\n loading R{config['ring']}: {config['tbeg'].date} ...")

    # loading data
    try:
        beat = __load_beat(config['tbeg'], config['tend'], config['ring'], config['path_to_autodata'])
    except:
        print(f" -> failed to load data: {config['tbeg']}")
        quit()

    # check length of DataFrame
    if len(beat) == 0:
        print(f" -> no beat file: {config['tbeg']}")
        quit()

    # _____________________________________
    # ### Define Variables

    # prepare output dataframe
    status = DataFrame()
    status['times_utc'] = beat.times_utc
    status['times_utc_sec'] = beat.times_utc_sec

    N = status.shape[0]

    mlti = np.ones(N)
    quality = np.ones(N)
    fsagnac = np.ones(N)
    dc_threshold = np.ones(N)
    ac_threshold = np.ones(N)

    # _____________________________________
    # ## Determine Status

    idx_mlti = 0

    for idx in range(beat.shape[0]):

        _time = obs.UTCDateTime(status.times_utc.iloc[idx])

        ## check if time conincides with MLTI
        # print(_time, mlti_t1[idx_mlti], mlti_t2[idx_mlti])
        if len(mlti_t1) > 0 and len(mlti_t2) > 0:
            if _time >= mlti_t1[idx_mlti] and _time <= mlti_t2[idx_mlti]:
                quality[idx] = 0
                mlti[idx] = 0

            ## update mlti interval
            if _time > mlti_t2[idx_mlti] and idx_mlti < len(mlti_t1)-1:
                idx_mlti += 1

        # check if Sagnac frequency in band
        flower = config['fsagnac_nominal'] + config['delta_fsagnac']
        fupper = config['fsagnac_nominal'] - config['delta_fsagnac']
        if beat.fj.iloc[idx] < fupper or beat.fj.iloc[idx] > flower:
            quality[idx] = 0
            fsagnac[idx] = 0

        # only if signal has a DC offset, thus not common mode
        if not config['commonmode']:
            # check against DC threshold
            if beat.dc_z.iloc[idx] < config['DC_threshold']:
                quality[idx] = 0
                dc_threshold[idx] = 0

        # check against AC threshold
        if beat.ac_z.iloc[idx] < config['AC_threshold']:
            quality[idx] = 0
            ac_threshold[idx] = 0

    # add to dictionary
    status['quality'] = quality
    status['fsagnac'] = fsagnac
    status['mlti'] = mlti
    status['ac_threshold'] = ac_threshold
    status['dc_threshold'] = dc_threshold

    # store output to file
    outfilename = f"R{config['ring']}_{config['tbeg'].date}_status"
    __save_to_pickle(status, config['path_to_output'], outfilename)

    # _____________________________________
    # ### Plotting

    arr = np.ones((3, status['quality'].size))

    arr[0] *= status['quality']
    arr[1] *= status['fsagnac']
    arr[2] *= status['mlti']

    try:
        names = ["quality", "fsagnac", "mlti", "ac_threshold", "dc_threshold"]
        bars = np.ones(len(names))-0.5

        arr = np.ones((len(names), status['quality'].size))

        for _n, name in enumerate(names):
            arr[_n] *= status[name]


        cmap = matplotlib.colors.ListedColormap(['darkred', 'green'])

        fig = plt.figure(figsize=(15, 4))

        c = plt.pcolormesh(np.arange(0, arr.shape[1]), names, arr, cmap=cmap, rasterized=True, alpha=0.8)

        for _k, bar in enumerate(bars):
            plt.axhline(bar+_k, color="k", alpha=0.5)

        plt.xlabel("Time (min)")


        plt.title(f"Quality Status of R{config['ring']} on {config['tbeg'].date}")

        print(f" -> stored: {config['path_to_figures']}R{config['ring']}_{config['tbeg'].date}_status.png")
        fig.savefig(config['path_to_figures']+f"R{config['ring']}_{config['tbeg'].date}_status.png", format="png", dpi=100, bbox_inches='tight')

        plt.close();

    except:
        print(" -> failed to plot: {config['tbeg']")

# ___________________________________________________

if __name__ == "__main__":
    main(config)

# End of File
