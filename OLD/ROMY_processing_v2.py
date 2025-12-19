#!/bin/python3

'''
ROMY Processing

- load ROMY ZUV data
- remove sensitivity and rotate to ZNE
- mask with MLTI log to exclude MLTI boosts
- store as masked streams

'''

import os
import sys
import obspy as obs
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from numpy import where


import warnings
warnings.filterwarnings('ignore')

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/ontap-ffb-bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'
elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'


config = {}

config['path_to_sds'] = archive_path+"romy_archive/"

config['path_to_sds_out'] = archive_path+"temp_archive/"

config['path_to_inventory'] = root_path+"Documents/ROMY/stationxml_ringlaser/dataless/"

config['path_to_figs'] = data_path+"2delete/testfigs/"

config['store_figure'] = False

if len(sys.argv) > 1:
    config['tbeg'] = obs.UTCDateTime(sys.argv[1])
config['tend'] = config['tbeg'] + 86400

config['sampling_rate'] = 20 # Hz

config['time_offset'] = 60 # seconds

config['t1'] = config['tbeg']-config['time_offset']
config['t2'] = config['tend']+config['time_offset']

config['Nexpected'] = int((config['t2'] - config['t1']) * config['sampling_rate'])



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

        if _t - _tlast > time_delta:
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

def __load_lxx(tbeg, tend, path_to_archive):

    from obspy import UTCDateTime
    from pandas import read_csv, concat

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if tbeg.year == tend.year:
        year = tbeg.year

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{year}/logfiles/LXX_maintenance.log"

        lxx = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

    else:

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tbeg.year}/logfiles/LXX_maintenance.log"
        lxx1 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tend.year}/logfiles/LXX_maintenance.log"
        lxx2 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        lxx = concat([lxx1, lxx2])

    lxx = lxx[(lxx.datetime > tbeg) & (lxx.datetime < tend)]

    return lxx

def __rotate_romy_ZUV_ZNE(st, inv, keep_z=True):

    from obspy.signal.rotate import rotate2zne

    ori_z = inv.get_orientation("BW.ROMY.10.BJZ")
    ori_u = inv.get_orientation("BW.ROMY..BJU")
    ori_v = inv.get_orientation("BW.ROMY..BJV")

    romy_z = st.select(channel="*Z")[0].data
    romy_u = st.select(channel="*U")[0].data
    romy_v = st.select(channel="*V")[0].data


    romy_z, romy_n, romy_e = rotate2zne(
                                        romy_z, ori_z['azimuth'], ori_z['dip'],
                                        romy_u, ori_u['azimuth'], ori_u['dip'],
                                        romy_v, ori_v['azimuth'], ori_v['dip'],
                                        inverse=False
                                       )

    st_new = st.copy()

    if keep_z:
        st_new.select(channel="*Z")[0].data = st.select(channel="*Z")[0].data
    else:
        st_new.select(channel="*Z")[0].data = romy_z

    st_new.select(channel="*U")[0].data = romy_n
    st_new.select(channel="*V")[0].data = romy_e

    ch = st_new.select(channel="*U")[0].stats.channel[:2]

    st_new.select(channel="*U")[0].stats.channel = f"{ch}N"
    st_new.select(channel="*V")[0].stats.channel = f"{ch}E"

    return st_new

def __write_stream_to_sds(st, cha, path_to_sds):

    import os

    # check if output path exists
    if not os.path.exists(path_to_sds):
        print(f" -> {path_to_sds} does not exist!")
        return

    tr = st.select(channel=cha)[0]

    nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
    yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

    if not os.path.exists(path_to_sds+f"{yy}/"):
        os.mkdir(path_to_sds+f"{yy}/")
        print(f"creating: {path_to_sds}{yy}/")
    if not os.path.exists(path_to_sds+f"{yy}/{nn}/"):
        os.mkdir(path_to_sds+f"{yy}/{nn}/")
        print(f"creating: {path_to_sds}{yy}/{nn}/")
    if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/"):
        os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/")
        print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/")
    if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D"):
        os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D")
        print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/{cc}.D")

    st.write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")

    print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")

def __mlti_intervals_to_zero(dat, times, mlti_t1, mlti_t2, t_offset_sec=120):

    from numpy import nan, where, full, array

    dat = array(dat)

    _mask = full((len(times)), 0, dtype=int)

    idx = 0
    for nn, tt in enumerate(times):

        if idx >= len(mlti_t1):
            continue
        else:
            t1, t2 = (mlti_t1[idx]-t_offset_sec), (mlti_t2[idx]+t_offset_sec)

        if tt >= t1:
            _mask[nn] = 1
        if tt > t2:
            idx += 1

    dat = where(_mask == 1, 1, dat)

    return dat

def __get_trace(seed):

    from numpy import zeros

    net, sta, loc, cha = seed.split('.')

    trr = obs.Trace()
    trr.stats.starttime = config['t1']
    trr.data = zeros(config['Nexpected'])
    trr.stats.network = net
    trr.stats.station = sta
    trr.stats.location = loc
    trr.stats.channel = cha
    trr.stats.sampling_rate = config['sampling_rate']

    return trr

def despike_sta_lta(arr, df, t_lta=50, t_sta=1, threshold_upper=40, plot=False):

    from obspy.signal.trigger import recursive_sta_lta
    from obspy.signal.trigger import classic_sta_lta
    from obspy.signal.trigger import plot_trigger
    import numpy as np

    def smooth(y, npts):
        '''
        moving average of 1d signal for n samples
        '''
        win = np.hanning(npts)
        y_smooth = np.convolve(y, win/np.sum(win), mode='same')
        y_smooth[:npts//2] = np.nan
        y_smooth[-npts//2:] = np.nan
        return y_smooth

    # apply sta lta
    # cft = classic_sta_lta(arr, int(t_sta * df), int(t_lta * df))

    cft = recursive_sta_lta(arr, int(t_sta * df), int(t_lta * df))

    # plot trigger result
    if plot:
        plot_trigger(arr, cft, threshold_upper, 1, show=plot)

    # create mask from triggers
    spikes = np.where(cft < threshold_upper, 0, cft)
    spikes = np.where(spikes > threshold_upper, 1, spikes)

    # create mask with nan
    mask2 = where(spikes == 1, np.nan, spikes)
    mask2 += 1

    mask2 = smooth(mask2, 20)

    mask1 = np.nan_to_num(mask2, nan=0)

    return spikes, mask1, mask2

def __checkup_plot(_st, _masks, _spikes):

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15, 12))

    plt.subplots_adjust(hspace=0.15)

    for i, tr in enumerate(_st):

        cha = tr.stats.channel[-1]

        data_before = tr.copy().data
        data_after = data_before*_masks[cha]

        ax[i].plot(_spikes[tr.stats.channel[-1]], color="grey", alpha=0.6, zorder=1, label="spikes")
        ax[i].plot(data_before, label=f"{cha} before")
        ax[i].plot(data_after, label=f" {cha} after")

        ax[i].set_ylim(-np.nanmax(abs(data_after))*1.1, np.nanmax(abs(data_after))*1.1)

        ax[i].legend(loc=1)

    # plt.show();
    return fig

def __read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED"):

    '''
    VARIABLES:
     - path_to_archive
     - seed
     - tbeg, tend
     - data_format

    DEPENDENCIES:
     - from obspy.core import UTCDateTime
     - from obspy.clients.filesystem.sds import Client

    OUTPUT:
     - stream

    EXAMPLE:
    >>> st = __read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED")

    '''

    import os
    from obspy.core import UTCDateTime, Stream
    from obspy.clients.filesystem.sds import Client

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if not os.path.exists(path_to_archive):
        print(f" -> {path_to_archive} does not exist!")
        return

    ## separate seed id
    net, sta, loc, cha = seed.split(".")

    ## define SDS client
    client = Client(path_to_archive, sds_type='D', format=data_format)

    ## read waveforms
    try:
        st = client.get_waveforms(net, sta, loc, cha, tbeg, tend, merge=-1)
    except:
        print(f" -> failed to obtain waveforms!")
        st = Stream()

    return st


def main(config):

    print(f"processing: {config['tbeg'].date} ...")

    # load MLTI logs
    mltiU = __load_mlti(config['t1'], config['t2'], "U", archive_path)
    mltiV = __load_mlti(config['t1'], config['t2'], "V", archive_path)
    mltiZ = __load_mlti(config['t1'], config['t2'], "Z", archive_path)

    # compute intervals for MLTI
    mltiU_t1, mltiU_t2 = __get_mlti_intervals(mltiU.time_utc)
    mltiV_t1, mltiV_t2 = __get_mlti_intervals(mltiV.time_utc)
    mltiZ_t1, mltiZ_t2 = __get_mlti_intervals(mltiZ.time_utc)

    # load maintenance file
    # lxx = __load_lxx(config['t1'], config['t2'], archive_path)

    # load inventory
    romy_inv = obs.read_inventory(config['path_to_inventory']+"dataless.seed.BW_ROMY")

    # load ROMY data
    st0 = obs.Stream()
    st0 += __read_sds(config['path_to_sds'], "BW.ROMY.10.BJZ", config['t1'], config['t2'])
    st0 += __read_sds(config['path_to_sds'], "BW.ROMY..BJU", config['t1'], config['t2'])
    st0 += __read_sds(config['path_to_sds'], "BW.ROMY..BJV", config['t1'], config['t2'])

    # remove sensitivity
    st0 = st0.remove_sensitivity(romy_inv)

    # check if merging is required
    if len(st0) > 3:
        print(f" -> merging required!")
        st0 = st0.merge(fill_value="interpolate")

    st0 = st0.trim(config['tbeg'], config['tend'])

    # check if data has same length
    for tr in st0:
        Nreal = len(tr.data)
        if Nreal != config['Nexpected']:
            tr.data = tr.data[:config['Nexpected']]
            # print(f" -> adjust length: {tr.stats.station}.{tr.stats.channel}:  {Nreal} -> {config['Nexpected']}")

    # remove trend
    st0 = st0.detrend("linear")

    # rotate streams
    st0 = __rotate_romy_ZUV_ZNE(st0, romy_inv, keep_z=True)

    # prepare MLTI masks
    tr_mltiU = __get_trace("BW.ROMY.40.MLT")

    tr_mltiU.data = __mlti_intervals_to_zero(tr_mltiU.data,
                                             tr_mltiU.times(reftime=config['t1'], type="utcdatetime"),
                                             mltiU_t1,
                                             mltiU_t2,
                                             t_offset_sec=60
                                             )

    tr_mltiV = __get_trace("BW.ROMY.40.MLT")

    tr_mltiV.data = __mlti_intervals_to_zero(tr_mltiV.data,
                                             tr_mltiV.times(reftime=config['t1'], type="utcdatetime"),
                                             mltiV_t1,
                                             mltiV_t2,
                                             t_offset_sec=60
                                             )

    tr_mltiZ = __get_trace("BW.ROMY.40.MLT")

    tr_mltiZ.data = __mlti_intervals_to_zero(tr_mltiZ.data,
                                             tr_mltiZ.times(reftime=config['t1'], type="utcdatetime"),
                                             mltiZ_t1,
                                             mltiZ_t2,
                                             t_offset_sec=60
                                             )

    # make mlti trace for horizontals (since rotated and intermixed)
    tr_mltiH = tr_mltiU.copy()

    # combine mlti periods
    tr_mltiH.data = tr_mltiU.data + tr_mltiV.data

    # remove periods with value 2 due to summation
    tr_mltiH.data = where(tr_mltiH.data >= 1, 1, tr_mltiH.data)

    # prepare maintenance mask
#     lxx_t1, lxx_t2 = __get_mlti_intervals(lxx.datetime)

#     tr_lxx = __get_trace("BW.ROMY.30.LXX")

#     tr_lxx.data = __mlti_intervals_to_zero(tr_lxx.data,
#                                            tr_lxx.times(reftime=config['t1'], type="utcdatetime"),
#                                            lxx_t1,
#                                            lxx_t2,
#                                            t_offset_sec=60
#                                            )

    # write output Z
    outZ = obs.Stream()

    outZ += st0.select(component="Z").copy()
    outZ.select(component="Z")[0].stats.location = "40"

    data = outZ.select(component="Z")[0].data
    mask = tr_mltiZ.data

    if len(data) != len(mask):
        mask = mask[:len(data)]

    # apply mlti mask
    dat1 = ma.masked_array(data, mask=mask)

    # check if all data is masked -> replace with nan values
    if dat1.mask.all():
        outZ.select(component="Z")[0].data = np.nan*np.ones(len(dat1))
    else:
        # otherwise overwrite data with masked data
        outZ.select(component="Z")[0].data = dat1

    # trim to defined interval
    outZ = outZ.trim(config['tbeg'], config['tend'], nearest_sample=False)

    # split into several traces since masked array cannot be stored as mseed
    outZ = outZ.split()

    __write_stream_to_sds(outZ, "BJZ", config['path_to_sds_out'])

    # write output N
    outN = obs.Stream()

    outN += st0.select(component="N").copy()
    outN.select(component="N")[0].stats.location = "40"

    data = outN.select(component="N")[0].data
    mask = tr_mltiH.data

    if len(data) != len(mask):
        mask = mask[:len(data)]

    # apply mlti mask
    dat1 = ma.masked_array(data, mask=mask)

    # check if all data is masked -> replace with nan values
    if dat1.mask.all():
        outN.select(component="N")[0].data = np.nan*np.ones(len(dat1))
    else:
        # otherwise overwrite data with masked data
        outN.select(component="N")[0].data = dat1

    # trim to defined interval
    outN = outN.trim(config['tbeg'], config['tend'], nearest_sample=False)

    # split into several traces since masked array cannot be stored as mseed
    outN = outN.split()

    __write_stream_to_sds(outN, "BJN", config['path_to_sds_out'])

    # write output E
    outE = obs.Stream()

    outE += st0.select(component="E").copy()
    outE.select(component="E")[0].stats.location = "40"

    data = outE.select(component="E")[0].data
    mask = tr_mltiH.data

    if len(data) != len(mask):
        mask = mask[:len(data)]

    # apply mlti mask
    dat1 = ma.masked_array(data, mask=mask)

    # check if all data is masked -> replace with nan values
    if dat1.mask.all():
        outE.select(component="E")[0].data = np.nan*np.ones(len(dat1))
    else:
        # otherwise overwrite data with masked data
        outE.select(component="E")[0].data = dat1

    # adjust time period
    outE = outE.trim(config['tbeg'], config['tend'], nearest_sample=False)

    # split into several traces since masked array cannot be stored as mseed
    outE = outE.split()

    __write_stream_to_sds(outE, "BJE", config['path_to_sds_out'])




if __name__ == "__main__":
    main(config)

# End of File
