#!/usr/bin/env python
# coding: utf-8

## _______________________________________________
# ## Compute sagnac spectra and helicorder of ROMY components daily and save to archive


import os, gc, json, sys
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from scipy.signal import welch, periodogram
from numpy import zeros, argmax, arange, array, linspace, shape, nanmean, ones
from tqdm import tqdm
from pandas import DataFrame, date_range
from datetime import datetime, date

from andbro__querrySeismoData import __querrySeismoData
from andbro__calculate_propabilistic_distribution import __calculate_propabilistic_distribution
from andbro__cut_frequencies_array import __cut_frequencies_array
from andbro__read_sds import __read_sds

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


#%%
## _______________________________________________
## Configuration

freenas_path = "/import/freenas-ffb-01-data/"
#freenas_path = "/home/andbro/freenas/"


config = {}

## select ring laser
# config['ring'] = "Z"
config['ring'] = sys.argv[1]

## specify location code of rings
config['location'] = {"Z":"10", "U":"", "V":"", "W":""}

## specify expected Sagnac frequency of rings
config['rings'] = {"Z":553, "U":302, "V":448,"W":448}
config['f_expected'] = config['rings'][config['ring']]  ## expected sagnac frequency

## specify seed for raw sagnac data
config['seed_raw'] = f"BW.DROMY..FJ{config['ring']}"

## specify seed for rotation rate data
config['seed_rot'] = f"BW.ROMY.{config['location'][config['ring']]}.BJ{config['ring']}"

## get date time of yesterday
if len(sys.argv) >= 3:
    config['tbeg'] = str(sys.argv[-1])
else:
    config['tbeg'] = str((UTCDateTime.now()-40000).date)

# set date manual
# config['tbeg'] = "2023-08-09"

config['tend'] = config['tbeg']

## conversion from volts to counts of obsidian data logger
config['conversion'] = 0.59604645e-6

## specify output paths
config['outpath_data'] = freenas_path+f"romy_autodata/{UTCDateTime(config['tbeg']).year}/R{config['ring']}/spectra/"
config['outpath_figs'] = freenas_path+f"romy_plots/{UTCDateTime(config['tbeg']).year}/R{config['ring']}/spectra/"

config['path_to_sds'] = freenas_path+f"romy_archive"
config['repository'] = "archive"  ## "george"

## set saving options
config['save_plots'] = True
config['save_data'] = True

## select method for PSD computation
config['method'] = "welch" ## "welch" | "periodogram" | multitaper

## specify frequency band
config['f_band'] = 30 ## +- frequency band

## define lenght of segments for Welch PSD in seconds
config['segment_factor'] = 600 ## seconds

## _____________________
## Variables for looping

config['threshold'] = -10

## define time offset (represents an overlap of the time period)
config['time_offset'] = 30 ## seconds

## select time period of data to consider (3600 s = 1 hr)
config['interval'] = 3600  ## seconds



#%%______________________________________________
## Methods



def __makeplot_colorlines(config, ff, data, smooth=None):

    from numpy import log10, median

    def __get_median_psd(psds):

        from numpy import median, zeros, isnan

        med_psd = zeros(psds.shape[1])

        for f in range(psds.shape[1]):
            a = psds[:,f]
            med_psd[f] = median(a[~isnan(a)])

        return med_psd

    def __smooth(y, box_pts):
        from numpy import ones, convolve, hanning

        win = hanning(box_pts)
        y_smooth = convolve(y, win/sum(win), mode='same')

        return y_smooth


    cols = plt.cm.jet_r(linspace(0,1,shape(data)[0]+1))
#     cols = plt.cm.viridis(linspace(0,1,shape(data)[0]+1))

    ## ____________________________________________

    fig, ax = plt.subplots(1,1, figsize=(15,10))

    font = 14

    data_min = min([min(d) for d in data])
    data_max = max([max(d) for d in data])

    for i, psdx in enumerate(data):

        if smooth is not None:
            ax.plot(ff, __smooth(psdx,smooth), color=cols[i], zorder=2, label=i,  alpha=0.3)
        else:
            ax.plot(ff, psdx, color=cols[i], zorder=2, label=i,  alpha=0.3)

    ## select only psds above a median threshold for median computation
    psd_select = array([dat for dat in data if median(log10(dat)) > config['threshold']])
    try:
        psd_median = __get_median_psd(psd_select)
        ax.plot(ff, psd_median, color='k', lw=1, zorder=2)
    except:
        print(" -> median computation failed!")

    ax.set_xlim(config['f_expected']-config['f_band'], config['f_expected']+config['f_band'])

    ax.set_yscale("log")
    ax.set_ylim(data_min-0.01*data_min, data_max+0.5*data_max)

    leg = ax.legend(ncol=2)

    # change the line width for the legend
    [line.set_linewidth(3.0) for line in leg.get_lines()]


    ax.grid(ls='--', zorder=1)

    ax.set_xlabel("Frequency (Hz)", fontsize=font)
    ax.set_ylabel(f"PSD (V$^2$ /Hz) ", fontsize=font)
    ax.set_title(f"Sagnac Spetra on {date} ({config['interval']}s windows) ", fontsize=font+2)

    ax.tick_params(axis='both', labelsize=font-2)

#     plt.show();
    return fig


def __makeplot_colorlines_and_helicorder(config, ff, data, traces, peaks=None, smooth=None):

    from numpy import log10, median, flipud, nanmax, nanmin

    def __get_median_psd(psds):

        from numpy import median, zeros, isnan, nanmean

        med_psd = zeros(psds.shape[1])

        for f in range(psds.shape[1]):
            a = psds[:,f]
            med_psd[f] = median(a[~isnan(a)])

        return med_psd

    ## extract colors from colormap
    cols = plt.cm.jet_r(linspace(0,1,shape(data)[0]+1))

    ## ____________________________________________

    fig, ax = plt.subplots(1,2, figsize=(18,8))

    plt.subplots_adjust(wspace=0.15)

    font = 14

    data_min = nanmin([nanmin(d) for d in data])
    data_max = nanmax([nanmax(d) for d in data])

    data = flipud(data)

    for i, psdx in enumerate(data):

        if smooth is not None:
            ax[0].plot(ff, __smooth(psdx,smooth), color=cols[i], zorder=2, label=23-i,  alpha=0.3)
        else:
            ax[0].plot(ff, psdx, color=cols[i], zorder=2, label=23-i,  alpha=0.3)

    ## select only psds above a median threshold for median computation
    psd_select = array([dat for dat in data if median(log10(dat)) > config['threshold']])
    try:
        psd_median = __get_median_psd(psd_select)
        ax[0].plot(ff, psd_median, color='k', lw=1, zorder=2)
    except:
        print(" -> median computation failed!")

    ax[0].set_xlim(config['f_expected']-config['f_band'], config['f_expected']+config['f_band'])

    ax[0].set_yscale("log")
    ax[0].set_ylim(data_min-0.01*data_min, data_max+0.5*data_max)

    ## insert legend
    leg = ax[0].legend(ncol=2)

    # change the line width for the legend
    [line.set_linewidth(3.0) for line in leg.get_lines()]


    ax[0].grid(ls='--', zorder=1)

    ax[0].set_xlabel("Frequency (Hz)", fontsize=font)
    ax[0].set_ylabel(f"PSD (V$^2$ /Hz) ", fontsize=font)
    ax[0].set_title(f"Sagnac Spetra of R{config['ring']} on {config['tbeg']} ({config['interval']}s windows) ", fontsize=font+2)

    ax[0].tick_params(axis='both', labelsize=font-2)

    ## ___________________________________
    ## PLOT 2

#     norm_st_max = np.max(traces)
    timeaxis = linspace(0, 60, len(traces[0]))

    ## reverse list to have a downward helicorder
    traces.reverse()

    for m, tr in enumerate(traces):

        norm_tr_max = nanmax(abs(tr))
        try:
            ax[1].plot(timeaxis, tr/norm_tr_max - nanmean(tr/norm_tr_max) + m, color=cols[m], alpha=0.3)
        except:
            ax[1].plot(timeaxis, ones(len(timeaxis))*m, color=cols[m], alpha=0)

    ax[1].set_yticks(linspace(0,23,24))

    tck_lbls = [str(int(tt)).rjust(2,"0")+":00" for tt in linspace(0,23,24)]
    tck_lbls.reverse()
    ax[1].set_yticklabels(tck_lbls)

    ax[1].set_ylim(-1, 24)
    ax[1].set_xlabel("Time (min)", fontsize=font)
    ax[1].set_title(f"Helicorder of R{config['ring']} (trace normalized)", fontsize=font+2)
    ax[1].tick_params(axis='both', labelsize=font-2)

#    plt.show();
    return fig


def __makeplot_distribution(config, xx, yy, dist, overlay=False):

    from numpy import nanmax, nanmin
    from matplotlib import colors

    def __smooth(y, box_pts):
        from numpy import ones, convolve, hanning

        win = hanning(box_pts)
        y_smooth = convolve(y, win/sum(win), mode='same')

        return y_smooth



    cmap = plt.cm.get_cmap("YlOrRd")
#     cmap = plt.cm.get_cmap("viridis")
    cmap.set_bad("white")
    cmap.set_under("white")

    max_psds = nanmax(dist)
    min_psds = nanmin(dist)


    ## ____________________________________________

    fig, ax = plt.subplots(1,1, figsize=(15,10))

    font = 14

    im = ax.pcolormesh( xx, yy, dist.T,
                        cmap=cmap,
                        vmax=max_psds,
                        vmin=min_psds+0.01*min_psds,
                        norm=colors.LogNorm(),
                        )

    if overlay is not None:
        ax.plot(xx, __smooth(10**overlay, 50), color='k', alpha=0.6, lw=1, zorder=2, label="maxima")

    ax.set_xlim(config['f_expected']-config['f_band'], config['f_expected']+config['f_band'])

    ax.set_yscale("log")

    ax.legend(ncol=2)

    ax.grid(ls='--', zorder=1)

    ax.set_xlabel("Frequency (Hz)", fontsize=font)
    ax.set_ylabel(f"PSD (V$^2$ /Hz) ", fontsize=font)
    ax.set_title(f"Sagnac Spetra on {config['xdate']} ({config['interval']}s windows) ", fontsize=font+2)

    ax.tick_params(axis='both', labelsize=font-2)

    cb = plt.colorbar(im, ax=ax, anchor=(0.0, -0.5))
    cb.set_label("Propability Density", fontsize=font, labelpad=-60)

#     plt.show();
    return fig


def __get_welch_psd(config, arr, df):

    from scipy.signal import get_window

    segments = int(df*config['segment_factor'])

    f0, psd0 = welch(
                    arr,
                    fs=df,
                    window=get_window('hann', segments),
                    nperseg=segments,
                    noverlap=int(segments/2),
                    nfft=None,
                    detrend='constant',
                    return_onesided=True,
                    scaling='density',
                    )

    return f0, psd0

def __save_to_pickle(obj, filename):

    import pickle

    if not filename.split("/")[-1].split(".")[-1] == "pkl":
        filename = filename+".pkl"

    with open(filename, 'wb') as ofile:
        pickle.dump(obj, ofile)

    if os.path.isfile(filename):
        print(f" -> created: {filename}")


def __smooth(y, box_pts):
    from numpy import ones, convolve, hanning

    win = hanning(box_pts)
    y_smooth = convolve(y, win/sum(win), mode='same')

    return y_smooth


def __check_path(path):
    created=False
    if not os.path.exists(path):
        os.mkdir(path)
        created=True
    if created and os.path.exists(path):
        print(f" -> created: {path}")


#%%

###########################################################
########################### MAIN ##########################
###########################################################

def main():


    tbeg = date.fromisoformat(str(config['tbeg']))
    tend = date.fromisoformat(str(config['tend']))

    print(json.dumps(config, indent=4, sort_keys=True))

    __check_path(config['outpath_data'])
    __check_path(config['outpath_figs'])

    ### ---------------------------------------------
    ## looping days
    for xdate in date_range(tbeg, tend):

        config['xdate'] = xdate

        idx_count=0
        NNN = int(86400/config['interval'])

        psds, traces = [], []

        ### ---------------------------------------------
        ## looping hours
        for hh in tqdm(range(NNN)):

            ## define current time window
            dh = hh*config['interval']

            t1, t2 = UTCDateTime(xdate)+dh, UTCDateTime(xdate)+config['interval']+dh

            try:
                ## load data for current time window
#                print(" -> loading data ...")
                # st_raw, inv_raw = __querrySeismoData(
                #                                      seed_id=config['seed_raw'],
                #                                      starttime=t1-2*config['time_offset'],
                #                                      endtime=t2+2*config['time_offset'],
                #                                      repository=config['repository'],
                #                                      path=None,
                #                                      restitute=None,
                #                                      detail=None,
                #                                     )
                st_raw = __read_sds(config['path_to_sds'], config['seed_raw'], t1-2*config['time_offset'], t2+2*config['time_offset'])
            except:
                print(" -> failed to load raw data!")
                continue

            try:
                ## load data for current time window
#                print(" -> loading data ...")
                # st_rot, inv_rot = __querrySeismoData(
                #                                      seed_id=config['seed_rot'],
                #                                      starttime=t1-2*config['time_offset'],
                #                                      endtime=t2+2*config['time_offset'],
                #                                      repository=config['repository'],
                #                                      path=None,
                #                                      restitute=True,
                #                                      detail=None,
                #                                     )
                st_rot = __read_sds(config['path_to_sds'], config['seed_rot'], t1-2*config['time_offset'], t2+2*config['time_offset'])
            except:
                print(" -> failed to load rot data!")
                continue

            if len(st_rot) == 0 or len(st_raw) == 0:
                print(f" -> detected empty stream!")
                continue

            st_rot[0].trim(t1, t2)
            st_raw[0].trim(t1, t2)

            ## convert from counts to volts
            st_raw[0].data = st_raw[0].data * config['conversion']


#            print(" -> computing welch ...")
            try:
                ff, psd = __get_welch_psd(config, st_raw[0].data, st_raw[0].stats.sampling_rate)
            except Exception as e:
                print(f" -> failed to compute psd")
                print(e)

            psds.append(psd)
            traces.append(st_rot[0].data)

            del st_raw, st_rot
            gc.collect()

        if len(psds) == 0:
            continue

        ## generate output object
        output = {}
        output['frequencies'] = ff
        output['psds'] = array(psds)

        ## store output
        if config['save_data']:
            date_str = str(xdate)[:10].replace("-","")

            __save_to_pickle(output, f"{config['outpath_data']}R{config['ring']}_{date_str}_spectra.pkl")


        ## limit frequency range for plotting
        try:
            f_min , f_max = config['f_expected']-config['f_band'], config['f_expected']+config['f_band']
            psds, ff = __cut_frequencies_array(array(psds), ff, f_min, f_max)
        except:
            print(f" -> failed to cut frequeny range!")

        ## Plotting
#         try:
#             colorlines = __makeplot_colorlines(config, ff, array(psds), smooth=None);
#         except Exception as e:
#             print(" -> failed to plot colorlines!")
#             print(e)

#         try:
#             colorlines_smooth = __makeplot_colorlines(config, ff, array(psds), smooth=20);
#         except Exception as e:
#             print(" -> failed to plot colorlines smooth!")
#             print(e)

        try:
            colorlines_heli = __makeplot_colorlines_and_helicorder(config, ff, array(psds), traces, peaks=None, smooth=None);
        except Exception as e:
            print(" -> failed to plot colorlines heli!")
            print(e)

#         out = __calculate_propabilistic_distribution(psds, bins=50, density=True, y_log_scale=True, axis=0)
#         distribution = __makeplot_distribution(config, ff, out['bin_mids'], out['dist'], overlay=out['bins_maximas']);


        if config['save_plots']:

            ## make date string
            date_str = str(xdate)[:10].replace("-","")

#               ### PLOT 1 -----------------
#             try:
#                 outname = f"plot_sagnacspectra_{date_str}_{config['interval']}_colorlines.png"

#                 colorlines.savefig(
#                                     f"{config['outpath_figs']}{subdir}{outname}",
#                                     dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
#                                     format="png", transparent=False, bbox_inches="tight", pad_inches=0.2,
#                                    )
#                 print(f" -> saving: {config['outpath_figs']}{subdir}{outname}...")
#             except Exception as e:
#                 print(e)
#                 pass

#               ### PLOT 2 -----------------
#             try:
#                 outname = f"plot_sagnacspectra_{date_str}_{config['interval']}_colorlines_smooth.png"

#                 colorlines_smooth.savefig(
#                                           f"{config['outpath_figs']}{subdir}{outname}",
#                                           dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
#                                           format="png", transparent=False, bbox_inches="tight", pad_inches=0.2,
#                                          )
#                 print(f" -> saving: {config['outpath_figs']}{subdir}{outname}...")
#             except Exception as e:
#                 print(e)
#                 pass

            ### PLOT 3 -----------------
            try:
                outname = f"{date_str}_sagnacspectra_R{config['ring']}.png"

                colorlines_heli.savefig(
                                    f"{config['outpath_figs']}{outname}",
                                    dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                                    format="png", transparent=False, bbox_inches="tight", pad_inches=0.2,
                                    )
                print(f" -> saving: {config['outpath_figs']}{outname} ...")
            except Exception as e:
                print(e)
                pass

#%%

if __name__ == "__main__":
    main()


## END OF FILE
