#!/usr/bin/env python
# coding: utf-8

# ## Compute Sagnac Frequency


import os, gc, json, shutil, sys
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging

from pprint import pprint
from obspy import UTCDateTime, Stream
from scipy.signal import welch, periodogram
from numpy import zeros, argmax, arange
from tqdm import tqdm
from pandas import DataFrame, date_range
from datetime import datetime, date

# sys.path.append("/home/brotzer/andbro_python")

# from andbro__querrySeismoData import __querrySeismoData
from andbro__utc_to_mjd import __utc_to_mjd
from andbro__read_sds import __read_sds


archive_path = "/import/freenas-ffb-01-data/"

## Configuration

config = {}

config['ring'] = sys.argv[1]
# config['ring'] = "V"

## specify seed id for [sagnac, monobeam1, monobeam2]
config['seeds'] = [f"BW.DROMY..FJ{config['ring']}", f"BW.DROMY..F1{config['ring']}", f"BW.DROMY..F2{config['ring']}"]
if config['ring'] == "Z":
    config['seeds'] = [f"BW.DROMY..FJ{config['ring']}", f"BW.DROMY..F1V", f"BW.DROMY..F2V"]

config['n_cpu'] = 3

if len(sys.argv) >= 3:
    config['tbeg'] = UTCDateTime(sys.argv[-1]).date
else:
    config['tbeg'] = (UTCDateTime.now()-40000).date
    # config['tbeg'] = UTCDateTime("2023-08-01").date

config['tend'] = config['tbeg']

## define appendix for output file
config['outfile_appendix'] = ""

## specify path to raw romy data as sds archive
config['path_to_sds'] = archive_path+f"romy_archive/"

## specify path for output data
config['outpath_data'] = archive_path+f"romy_autodata/{config['tbeg'].year}/R{config['ring']}/"

## specify path for logfile
config['outpath_logs'] = archive_path+f"romy_autodata/{config['tbeg'].year}/logfiles/"

## create logfile name
config['logfile'] = f"{config['ring']}_{config['tbeg'].year}_autodata.log"

## obsidian convert from V to count  [0.59604645ug  from obsidian]
config['conversion'] = 0.59604645e-6

## specify repository to request data from [archive, george, ..]
config['repository'] = "archive"

## select method to compute Sagnac frequency
## "hilbert" | "multitaper_hilbert" | "welch" | "periodogram" | multitaper | multitaper_periodogram
config['method'] = "hilbert"

## define and select Sagnac frequency for rings
rings = {"Z":553, "U":302, "V":448,"W":448}
config['f_expected'] = rings[config['ring']]  ## expected sagnac frequency in Hz

## specify frequency band around Sagnac frequency (f_expected +- f_band)
config['f_band'] = 3 ## +- frequency band in Hz

## specify number of windows for multitaper
#config['n_windows'] = 10

## specify the time steps (equivalent to time delta / sampling rate) [ -> default = 60 ]
config['t_steps'] = 60  ## seconds

## specify time overlap for window to consider for frequency estimation [ -> default = 180 ]
config['t_overlap'] = 180 ## seconds

## specify time interval for data chuncks to load [-> default = 3600 ]
config['time_interval'] = 3600  ## seconds

## adjust number of samples
config['NN'] = int(config['time_interval']/config['t_steps'])




## Methods

def __hilbert_frequency_estimator(config, st, fs):

    from scipy.signal import hilbert
    import numpy as np

    st0 = st.copy()


    f_lower = config['f_expected'] - config['f_band']
    f_upper = config['f_expected'] + config['f_band']


    ## bandpass with butterworth
    st0.detrend("linear")
    st0.taper(0.1)
    st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)


    ## estimate instantaneous frequency with hilbert
    signal = st0[0].data

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

    ## cut first and last 5% (corrupted)

    dd = int(0.05*len(instantaneous_frequency))

    t = st0[0].times()
    t1 = st0[0].times()[1:]
    t2 = t1[dd:-dd]

    t_mid = t[int((len(t))/2)]

    insta_f_cut = instantaneous_frequency[dd:-dd]

    ## averaging
    insta_f_cut_mean = np.mean(insta_f_cut)
#    insta_f_cut_mean = np.median(insta_f_cut)

    return t_mid, insta_f_cut_mean, np.mean(amplitude_envelope), np.std(insta_f_cut)


def __compute_ac_dc_contrast(config, st):

    from numpy import percentile, nanmean

    st0 = st.copy()
    dat = st0[0].data

    percentiles = percentile(dat, [2.5, 97.5])
    ac = percentiles[1]-percentiles[0]
    dc = nanmean(dat)

    # contrast(max(dat)-min(dat))/(max(dat)+min(dat))
    con = (percentiles[1]-percentiles[0])/(percentiles[1]+percentiles[0])


    return ac, dc, con


def __compute_phase(config, st0):


    from numpy import argmax, pi
    from obspy.signal.cross_correlation import correlate, xcorr_max

    st_tmp = st0.copy()

    df_new = st_tmp[0].stats.sampling_rate * 10

    mono1 = config['seeds'][1].split(".")[3]
    mono2 = config['seeds'][2].split(".")[3]


    f1 = st_tmp.select(channel=mono1).detrend("demean").normalize().interpolate(df_new)[0]
    f2 = st_tmp.select(channel=mono2).detrend("demean").normalize().interpolate(df_new)[0]

    # cc = correlate(f1.data[:N], f2.data[:N], int(len(f1.data[:N])/2))
    cc = correlate(f1.data, f2.data, int(len(f1.data)/2))

    shift, value = xcorr_max(cc)

    time_shift = shift/df_new

    return time_shift, value


def __loop(config, st0, starttime):

    from numpy import nan, zeros

    NN = config['NN']

    ii = 0
    n1 = 0
    n2 = config['t_steps']

    ph = zeros(NN)

    while n2 <= config['time_interval']:

        try:

            ## cut stream to chuncks
            st_tmp = st0.copy().trim(starttime+n1-config['t_overlap']/2, starttime+n1+config['t_steps']+config['t_overlap']/2)

            ## compute time shift with crosscorrelation
            time_shift, cc_max = __compute_phase(config, st_tmp)

            ## append values to arrays
            ph[ii] = time_shift


        except Exception as e:
            ph[ii] = nan
            logging.info(" -> computing time shift failed")
            logging.error(e)

        ii += 1
        n1 += config['t_steps']
        n2 += config['t_steps']

    return ph


def __compute(config, st0, starttime, method="hilbert"):

    from scipy.signal import find_peaks, peak_widths, welch, periodogram
    from numpy import nan, zeros

    NN = config['NN']

    ii = 0
    n1 = 0
    n2 = config['t_steps']

    tt0, tt1, tt2, ff, hh, pp = zeros(NN), zeros(NN).astype(str), zeros(NN), zeros(NN), zeros(NN), zeros(NN)
    ac, dc, con = zeros(NN), zeros(NN), zeros(NN)


    while n2 <= config['time_interval']:

        try:

            ## cut stream to chuncks
            st_tmp = st0.copy().trim(starttime+n1-config['t_overlap']/2, starttime+n1+config['t_steps']+config['t_overlap']/2)

            ## get time series from stream
            # times = st_tmp[0].times(reftime=UTCDateTime("2016-01-01T00"))

            ## get sampling rate from stream
            df = st_tmp[0].stats.sampling_rate


            if method == "hilbert":

                f_tmp, f_max, p_max, h_tmp = __hilbert_frequency_estimator(config, st_tmp, df)

            else:
                print(" -> unkown method")
                continue

            times_utc = st_tmp[0].times("utcdatetime")
            times_sec = st_tmp[0].times()

            ## copmute AC, DC and Contrast
            ac0, dc0, con0 = __compute_ac_dc_contrast(config, st_tmp)


            ## append values to arrays
            tt0[ii] = times_sec[int(len(times_sec)/2)]
            tt1[ii] = str(times_utc[int(len(times_utc)/2)])
            tt2[ii] = __utc_to_mjd(times_utc[int(len(times_utc)/2)])
            ff[ii] = f_max
            pp[ii] = p_max
            hh[ii] = h_tmp
            ac[ii] = ac0
            dc[ii] = dc0
            con[ii] = con0


        except Exception as e:
            tt0[ii], tt1[ii], tt2[ii], ff[ii], pp[ii], hh[ii] = nan, nan, nan, nan, nan, nan
            logging.info(" -> computing failed")
            logging.error(e)

        ii += 1
        n1 += config['t_steps']
        n2 += config['t_steps']

    return tt0, tt1, tt2, ff, hh, pp, ac, dc, con


def __store_as_pickle(obj, filename):

    import pickle, isfile

    ofile = open(filename, 'wb')
    pickle.dump(obj, ofile)

    if isfile(filename):
        print(f"created: {filename}")


def __join_pickle_files(config):

    import pickle
    from numpy import sort
    from pandas import concat, read_pickle

    files = sort(os.listdir(config['outpath_data']+"tmp/"))

    data = concat([read_pickle(config['outpath_data']+"tmp/"+filex) for filex in files if ".pkl" in filex])

    data.sort_values(by="times_mjd")
    data.reset_index(drop=True, inplace=True)

    return data


## _________________________________________________
## Looping in Main

def main(times):

    ## extract arguments
    jj, tbeg, tend = times

    ## adjust length of string
    jj = str(jj).rjust(3,"0")

    ## amount of samples
    # NNN = int(config['time_interval']/config['t_steps'])

    ## prepare empty arrays
    # t_utc, t_mjd = zeros(NNN), zeros(NNN)
    # fj, f1, f2 = zeros(NNN), zeros(NNN), zeros(NNN)
    # pz, p1, p2 = zeros(NNN), zeros(NNN), zeros(NNN)
    # ac_z, ac_1, ac_2 = zeros(NNN), zeros(NNN), zeros(NNN)
    # dc_z, dc_1, dc_2 = zeros(NNN), zeros(NNN), zeros(NNN)
    # con_z, con_1, con_2 = zeros(NNN), zeros(NNN), zeros(NNN)


    ## set times t1 and t2 as time interval to request
    t1, t2 = UTCDateTime(tbeg), UTCDateTime(tend)

    ## loop for sagnac and monobeams
    for seed in config['seeds']:

        loading_data_status = True

        try:
            ## load data for current time window
#                    print(" -> loading data ...")
#            st, inv = __querrySeismoData(
#                                        seed_id=seed,
#                                        starttime=t1-2*config['t_overlap'],
#                                        endtime=t2+2*config['t_overlap'],
#                                        repository=config['repository'],
#                                        path=None,
#                                        restitute=None,
#                                        detail=None,
#                                        )

            st = __read_sds(config['path_to_sds'], seed, t1-2*config['t_overlap'], t2+2*config['t_overlap'], data_format='MSEED')

            ## convert from V to count  [0.59604645ug  from obsidian]
            st[0].data = st[0].data*config['conversion']

        except Exception as e:
            logging.error(f" -> failed to load data for {seed}!")
            logging.error(e)
            loading_data_status = False
            # continue


        ## compute signal values
        if loading_data_status:
            try:
                tt_sec, tt_utc, tt_mjd, ff, hh, pp, ac, dc, con = __compute(config, st, t1, method=config['method'])
            except:
                logging.info(" -> computation failed")
                #tt_utc, tt_mjd, ff, hh, pp, ac, dc, con = [],[],[],[],[],[],[],[]
                tt_utc, tt_sec, tt_mjd, ff, hh, pp, ac, dc, con = zeros(60), zeros(60), zeros(60), zeros(60), zeros(60), zeros(60), zeros(60), zeros(60), zeros(60)



        monos = Stream()
        if seed.split(".")[3] == config['seeds'][0].split(".")[3]:
            fj, pz, ac_z, dc_z, con_z = ff, pp, ac, dc, con
        elif seed.split(".")[3] == config['seeds'][1].split(".")[3]:
            if not loading_data_status:
                f1, p1, ac_1, dc_1, con_1 =  zeros(60), zeros(60), zeros(60), zeros(60), zeros(60)
            else:
                monos += st[0]
                f1, p1, ac_1, dc_1, con_1 = ff, pp, ac, dc, con
        elif seed.split(".")[3] == config['seeds'][2].split(".")[3]:
            if not loading_data_status:
                f2, p2, ac_2, dc_2, con_2 = zeros(60), zeros(60), zeros(60), zeros(60), zeros(60)
            else:
                monos += st[0]
                f2, p2, ac_2, dc_2, con_2 = ff, pp, ac, dc, con



    ## create and write a dataframe
    df = DataFrame()
    df['times_utc'] = tt_utc
    df['times_utc_sec'] = tt_sec
    df['times_mjd'] = tt_mjd
    df['fj'], df['f1'], df['f2'] = fj, f1, f2
    df['pz'], df['p1'], df['p2'] = pz, p1, p2
    df['ac_z'], df['ac_1'], df['ac_2'] = ac_z, ac_1, ac_2
    df['dc_z'], df['dc_1'], df['dc_2'] = dc_z, dc_1, dc_2
    df['contrast_z'], df['contrast_1'], df['contrast_2'] = con_z, con_1, con_2
#    df['time_shift'] =  t_shift

    try:
        del st, tt_sec, tt_utc, tt_mjd, ff, hh, pp
        gc.collect()
    except:
        pass

    ## write data frame to tmp file
    df.to_pickle(config['outpath_data']+f"tmp/tmp_{tbeg}_{tend}_{jj}.pkl", protocol=4)



## ________ MAIN  ________
if __name__ == '__main__':

    pprint(config)

    ## start logging
    if os.path.isfile(config['outpath_data']+config['logfile']):
        os.remove(config['outpath_data']+config['logfile'])

    logging.basicConfig(filename=config['outpath_logs']+config['logfile'], format="%(asctime)s,%(levelname)s,%(message)s", datefmt="%m/%d/%Y %I:%M:%S", level=logging.DEBUG)
    logging.info(" __________ START ______________")
    logging.info(f" -> {config['tbeg']}: start at {UTCDateTime.now()}")

    ## create tmp directory
    if os.path.isdir(config['outpath_data']+"tmp/"):
        shutil.rmtree(config['outpath_data']+"tmp/")
        os.mkdir(config['outpath_data']+"tmp/")
    else:
        os.mkdir(config['outpath_data']+"tmp/")

    ## generate arguments for final parallel loop
    list_of_times = []

    for j, date in enumerate(date_range(config['tbeg'], config['tend'])):

        date = UTCDateTime(UTCDateTime(date).date)

        hh, counter = 0, 0

        while hh < 86400:

            tbeg = date - config['t_overlap'] + hh
            tend = date + config['t_overlap'] + hh + config['time_interval']

            list_of_times.append((counter, tbeg, tend))

            hh += config['time_interval']
            counter += 1


    ## launch parallel processes
    with mp.Pool(processes=config['n_cpu']) as pool:

        list(tqdm(pool.imap_unordered(main, list_of_times), total=len(list_of_times)))

    pool.close()
    pool.join()


    ## join tmp files to a master data frame
    logging.info(" -> joining pickle files to master file ...")
    df_all = __join_pickle_files(config)

    ## store master data frame as pickle file
    date_str = str(config['tbeg']).replace("-","")
    logging.info(f" -> writing: {config['outpath_data']}FJ{config['ring']}_{date_str}{config['outfile_appendix']}.pkl")
    df_all.to_pickle(f"{config['outpath_data']}FJ{config['ring']}_{date_str}{config['outfile_appendix']}.pkl")


    logging.info(f" -> Done {UTCDateTime.now()}")



## END OF FILE
