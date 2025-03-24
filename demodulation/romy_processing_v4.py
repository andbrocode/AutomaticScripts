#!/bin/python3

"""
ROMY Processing Module

This module provides functionality for processing ROMY ring laser data:
- Loading ROMY ZUV data
- Removing sensitivity and rotating to ZNE
- Masking with MLTI log to exclude MLTI boosts
- Deglitching data using pattern matching
- Storing as masked streams

Author: Andreas Brotzer
"""

import os
import sys
import numpy as np
import argparse
import numpy.ma as ma
import matplotlib.pyplot as plt
import obspy as obs
from typing import Dict, List, Tuple, Optional, Union
from obspy import UTCDateTime, Stream, Inventory
from obspy.core.trace import Trace

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure paths based on hostname
PATHS = {
    'lighthouse': {
        'root': '/home/andbro/',
        'data': '/home/andbro/kilauea-data/',
        'archive': '/home/andbro/freenas/',
        'bay': '/home/andbro/ontap-ffb-bay200/'
    },
    'kilauea': {
        'root': '/home/brotzer/',
        'data': '/import/kilauea-data/',
        'archive': '/import/freenas-ffb-01-data/',
        'bay': '/import/ontap-ffb-bay200/'
    }
}

# Add other hostnames with same paths as kilauea
for host in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    PATHS[host] = PATHS['kilauea']

# Get paths for current hostname
hostname = os.uname().nodename
paths = PATHS.get(hostname, PATHS['lighthouse'])

# Configurations
config = {
    # specify three seeds of ROMY
    'seeds': ["BW.ROMY.60.BJZ", "BW.ROMY.60.BJU", "BW.ROMY.60.BJV"],
    # path to inventory file
    'path_to_inventory': paths['root'] + "Documents/ROMY/stationxml_ringlaser/dataless/",
    # path to mlti logs
    'path_to_mlti_logs': paths['archive'] + "romy_archive/",
    # inventory filename
    'inventory_file': "dataless.seed.BW_ROMY",
    # path to sds archive
    'path_to_sds': paths['archive'] + "temp_archive/",
    # path to output sds
    'path_to_sds_out': paths['archive'] + "temp_archive/",
    # time offset for mlti intervals
    'time_offset': 60,  # seconds
    # output location code
    'loc_out': "00",
    # apply MLTI mask
    'apply_mlti_mask': True,
    # apply LXX mask
    'apply_lxx_mask': False,
    # keep Z component of horizontal ring
    'keep_z': True,
    # deglitch detection used for additional mask
    'apply_deglitching': False, # not working yet on production....
    # remove sensitivity
    'remove_sensitivity': False,
    # verbose output
    'verbose': True,
}

def deglitch_trace(tr: Trace, 
                   lowpass_freq: float = 0.1,
                   min_amplitude: float = 1e-11,
                   threshold_multiplier: float = 4,
                   corr_threshold: float = 0.6,
                   window_length: int = 300,
                   glitch_width: int = 21,
                   fit_window: int = 1000,
                   plot_detection: bool = True,
                   correct_orig_data: bool = True) -> Tuple[Trace, List[int], List[float]]:
    """
    Detect and remove glitches from a seismic trace using pattern matching.
    
    Parameters
    ----------
    tr : obspy.core.trace.Trace
        Input trace to deglitch
    lowpass_freq : float, optional
        Lowpass filter frequency in Hz
    min_amplitude : float, optional
        Minimum glitch amplitude to detect
    threshold_multiplier : float, optional
        Multiplier for detection threshold
    corr_threshold : float, optional
        Correlation threshold for pattern matching
    window_length : int, optional
        Analysis window length in samples
    glitch_width : int, optional
        Expected glitch width in samples
    fit_window : int, optional
        Window for fitting glitch model
    plot_detection : bool, optional
        Plot detection results
    correct_orig_data : bool, optional
        Correct original data
        
    Returns
    -------
    Tuple[Trace, List[int], List[float]]
        Deglitched trace, glitch indices, glitch amplitudes
    """
    from deglitch_utils import detect_and_correct_glitches
    
    # Create stream with single trace
    st = Stream([tr])
    
    # Detect and correct glitches
    st_corrected, indices, amplitudes = detect_and_correct_glitches(
        st,
        lowpass_freq=lowpass_freq,
        min_amplitude=min_amplitude,
        threshold_multiplier=threshold_multiplier,
        corr_threshold=corr_threshold,
        window_length=window_length,
        glitch_width=glitch_width,
        fit_window=fit_window,
        plot_detection=plot_detection,
        correct_orig_data=correct_orig_data
    )
    
    return st_corrected[0], indices, amplitudes

def get_mlti_intervals(mlti_times: List[UTCDateTime], 
                        time_delta: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get start and end times of MLTI intervals.
    
    Parameters
    ----------
    mlti_times : list of UTCDateTime
        List of MLTI event times
    time_delta : int, optional
        Time difference threshold in seconds
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of start and end times
    """
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

def load_mlti(tbeg: Union[str, UTCDateTime], tend: Union[str, UTCDateTime], ring: str, path_to_archive: str):
    """
    Load MLTI data from archive
    
    Parameters
    ----------
    tbeg : Union[str, UTCDateTime]
        Start time of the data to load
    tend : Union[str, UTCDateTime]
        End time of the data to load
    ring : str
        Ring letter to load
    path_to_archive : str
        Path to the archive directory
    """
    from obspy import UTCDateTime
    from pandas import read_csv, concat

    # Convert tbeg and tend to UTCDateTime objects
    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    # Define the mapping of ring letters to their corresponding numbers
    rings = {"U":"03", "Z":"01", "V":"02", "W":"04"}

    if tbeg.year == tend.year:
        year = tbeg.year

        # Define the path to the MLTI log file
        file_name = f"{year}_romy_{rings[ring]}_mlti.log"
        path_to_mlti = path_to_archive+f"{year}/BW/CROMY/{file_name}"

        # Check if the file exists
        if not os.path.exists(path_to_mlti):
            print(f"  -> {file_name} does not exist!")
            return None
        else:
            # Read the MLTI log file
            mlti = read_csv(path_to_mlti, names=["time_utc","Action","ERROR"])

    else:

        # Define the path to the MLTI log file for the start year
        file_name = f"{tbeg.year}_romy_{rings[ring]}_mlti.log"
        path_to_mlti1 = path_to_archive+f"romy_archive/{tbeg.year}/BW/CROMY/{file_name}"

        # Check if the file exists
        if not os.path.exists(path_to_mlti1):
            print(f"  -> {file_name} does not exist!")
            return None
        else:
            # Read the MLTI log file for first year
            mlti1 = read_csv(path_to_mlti1, names=["time_utc","Action","ERROR"])

        # Define the path to the MLTI log file for the end year
        file_name = f"{tend.year}_romy_{rings[ring]}_mlti.log"
        path_to_mlti2 = path_to_archive+f"romy_archive/{tend.year}/BW/CROMY/{file_name}"

        # Check if the file exists
        if not os.path.exists(path_to_mlti2):
            print(f" - > {file_name} does not exist!")
            return None
        else:
            # Read the MLTI log file for second year
            mlti2 = read_csv(path_to_mlti2, names=["time_utc","Action","ERROR"])

        # Concatenate the MLTI log files
        mlti = concat([mlti1, mlti2])

    # Filter the MLTI log file to only include data between tbeg and tend
    mlti = mlti[(mlti.time_utc > tbeg) & (mlti.time_utc < tend)]

    return mlti

def load_lxx(tbeg: Union[str, UTCDateTime], tend: Union[str, UTCDateTime], path_to_archive: str):
    """
    Load LXX data from archive
    
    Parameters
    ----------
    tbeg : Union[str, UTCDateTime]
        Start time of the data to load
    tend : Union[str, UTCDateTime]
        End time of the data to load
    path_to_archive : str
        Path to the archive directory
    """
    from obspy import UTCDateTime
    from pandas import read_csv, concat

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if tbeg.year == tend.year:
        year = tbeg.year

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{year}/logfiles/LXX_maintenance.log"

        # Check if the file exists
        if not os.path.exists(path_to_lx_maintenance):
            print(f" -> {path_to_lx_maintenance} does not exist!")
            return None
        else:
            # Read the LXX maintenance log file
            lxx = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

    else:

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tbeg.year}/logfiles/LXX_maintenance.log"
        if not os.path.exists(path_to_lx_maintenance):
            print(f" -> {path_to_lx_maintenance} does not exist!")
            return None
        else:
            # Read the LXX maintenance log file for first year
            lxx1 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tend.year}/logfiles/LXX_maintenance.log"
        if not os.path.exists(path_to_lx_maintenance):
            print(f" -> {path_to_lx_maintenance} does not exist!")
            return None
        else:
            # Read the LXX maintenance log file for second year
            lxx2 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        lxx = concat([lxx1, lxx2])

    lxx = lxx[(lxx.datetime > tbeg) & (lxx.datetime < tend)]

    return lxx

def rotate_romy_ZNE(st: Stream, inv: Inventory, use_components: List[str] = ["Z", "U", "V"], keep_z: bool = True) -> Stream:

    from obspy.signal.rotate import rotate2zne

    locs = {"Z":"10", "U":"", "V":"", "W":""}

    # make dictionary for components with data, azimuth and dip
    components = {}
    for comp in use_components:
        loc = locs[comp]
        components[comp] = {
            'data': st.select(component=comp)[0].data,
            'azimuth': inv.get_orientation(f"BW.ROMY.{loc}.BJ{comp}")['azimuth'],
            'dip': inv.get_orientation(f"BW.ROMY.{loc}.BJ{comp}")['dip']
        }

    # Rotate to ZNE
    romy_z, romy_n, romy_e = rotate2zne(
                                        components[use_components[0]]['data'], components[use_components[0]]['azimuth'], components[use_components[0]]['dip'],
                                        components[use_components[1]]['data'], components[use_components[1]]['azimuth'], components[use_components[1]]['dip'],
                                        components[use_components[2]]['data'], components[use_components[2]]['azimuth'], components[use_components[2]]['dip'],
                                        inverse=False
                                       )

    # create new stream with ZNE components
    st_new = st.copy()

    for c, tr in zip(['Z', 'N', 'E'], st_new):
        tr.stats.channel = tr.stats.channel[:2]+c
    
    if keep_z and 'Z' in use_components:
        st_new.select(component='Z')[0].data = components['Z']['data']
    else:
        st_new.select(component='Z')[0].data = romy_z

    st_new.select(component='N')[0].data = romy_n
    st_new.select(component='E')[0].data = romy_e

    return st_new

def write_stream_to_sds(st: Stream, loc: str, path_to_sds: str):
    """
    Write a stream to SDS

    Parameters
    ----------
    st : Stream
        Stream to write
    cha : str
        Channel to write
    """
    import os
    import gc
    # check if output path exists
    if not os.path.exists(path_to_sds):
        print(f" -> {path_to_sds} does not exist!")
        return

    # select trace
    for tr in st:

        # get station information
        nn, ss, cc = tr.stats.network, tr.stats.station, tr.stats.channel

        # get location code
        ll = loc

        # reset location code
        tr.stats.location = loc

        # get year and julday
        yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

        # create directory structure using os.makedirs
        os.makedirs(path_to_sds+f"{yy}/", exist_ok=True)
        os.makedirs(path_to_sds+f"{yy}/{nn}/", exist_ok=True)
        os.makedirs(path_to_sds+f"{yy}/{nn}/{ss}/", exist_ok=True)
        os.makedirs(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/", exist_ok=True)

        # splite trace if masked array
        if isinstance(tr.data, ma.masked_array):
            print("  -> masked array: filled with zeros")
            tr.data = tr.data.filled(0)

        # trim trace to 86400 seconds
        tr = tr.trim(tr.stats.starttime, tr.stats.starttime+86400-0.00001, nearest_sample=False)

        # write to SDS
        tr.write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")

        if os.path.isfile(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}"):
            print(f"  -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")
        else:
            print(f"  -> failed to store stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")

def mlti_intervals_to_zero(dat: np.ndarray, times: np.ndarray, mlti_t1: np.ndarray, mlti_t2: np.ndarray, t_offset_sec: int = 120) -> np.ndarray:
    """
    Set MLTI intervals to zero

    Parameters
    ----------
    dat : np.ndarray
        Data to set MLTI intervals to zero
    times : np.ndarray
        Times of the data
    mlti_t1 : np.ndarray
        Start times of MLTI intervals
    mlti_t2 : np.ndarray
        End times of MLTI intervals
    t_offset_sec : int, optional
        Offset in seconds

    Returns
    -------
    np.ndarray
        Data with MLTI intervals set to zero
    """
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

def get_trace(seed: str, tbeg: UTCDateTime, Nexpected: int, sampling_rate: float) -> Trace:
    """
    Get a trace object

    Parameters
    ----------
    seed : str
        Seed string
    config : Dict
        Configuration dictionary

    Returns
    -------
    Trace
        Trace object
    """
    from numpy import zeros

    net, sta, loc, cha = seed.split('.')

    trr = Trace()
    trr.stats.starttime = tbeg
    trr.data = zeros(Nexpected)
    trr.stats.network = net
    trr.stats.station = sta
    trr.stats.location = loc
    trr.stats.channel = cha
    trr.stats.sampling_rate = sampling_rate

    return trr

def read_from_sds(path_to_archive: str, seed: str, 
             tbeg: Union[str, UTCDateTime], 
             tend: Union[str, UTCDateTime], 
             data_format: str = "MSEED") -> Stream:
    """
    Read SDS data

    Parameters
    ----------
    path_to_archive : str
        Path to archive
    seed : str
        Seed string
    tbeg : str
        Start time
    tend : str
        End time
    data_format : str, optional
        Data format

    Returns
    -------
    Stream
        Stream object
    """
    import os
    from obspy.core import UTCDateTime, Stream
    from obspy.clients.filesystem.sds import Client

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if not os.path.exists(path_to_archive):
        print(f" -> {path_to_archive} does not exist!")
        return Stream()

    # separate seed id
    net, sta, loc, cha = seed.split(".")

    # define SDS client
    client = Client(path_to_archive, sds_type='D', format=data_format)

    # read waveforms
    try:
        st = client.get_waveforms(net, sta, loc, cha, tbeg, tend, merge=-1)
    except:
        print(f" -> failed to obtain waveforms!")
        st = Stream()

    return st


def main(config):

    if config['verbose']:
        print(f"processing: {config['tbeg'].date} {config['tend'].date}...")

    # load inventory
    try:
        romy_inv = obs.read_inventory(config['path_to_inventory']+config['inventory_file'])
    except:
        print(f" -> failed to load inventory from {config['path_to_inventory']}!")
        return

    # load ROMY data
    st0 = Stream()
    for seed in config['seeds']:
        if config['verbose']:
            print(f" -> loading data for {seed}...")
        st0 += read_from_sds(config['path_to_sds'], seed, config['tbeg'], config['tend'])

    # remove sensitivity
    if config['remove_sensitivity']:
        st0 = st0.remove_sensitivity(romy_inv)

    if len(st0) < 3:
        print(f" -> not enough data found! Only {len(st0)} traces found!")
        return

    # if sizes are not the same, print the station and channel
    sizes = [len(tr.data) for tr in st0]
    trim_required = False
    for i, size in enumerate(sizes):
        if size != sizes[0]:
            print(f" -> data length is not correct for {st0[i].stats.station}.{st0[i].stats.channel}: {size} != {sizes[0]}")
            trim_required = True
    if trim_required:
        st0 = st0._trim_common_channels()

    # check if merging is required
    if len(st0) > 3:
        print(f" -> merging required!")
        st0 = st0.merge(fill_value="interpolate")

    # trim to defined interval
    st0 = st0.trim(config['tbeg'], config['tend'])

    # remove trend
    st0 = st0.detrend("linear")

    # rotate streams
    components = [tr.stats.channel[-1] for tr in st0]

    if config['verbose']:
        print(f" -> rotating to ZNE...")
    st0 = rotate_romy_ZNE(st0, romy_inv, use_components=components, keep_z=config['keep_z'])

    # update components after rotation
    components_zne = [tr.stats.channel[-1] for tr in st0]

    # assume to be identical for all traces
    Nsamples = st0[0].stats.npts
    sampling_rate = st0[0].stats.sampling_rate

    # apply deglitching
    if config['apply_deglitching']:
        if config['verbose']:
            print(f" -> applying deglitching...")
        for tr in st0:
            st0 = deglitch_trace(tr)

    if config['apply_mlti_mask']:
        if config['verbose']:
            print(f" -> preparing and applying MLTI masks...")
        
        # prepare MLTI masks for components
        mlti_masks = {}
        for component in components:

            # load MLTI logs
            try:
                mlti = load_mlti(config['tbeg'], config['tend'], component, config['path_to_mlti_logs'])
            except:
                print(f" -> failed to load MLTI logs for {component}!")
                continue
    
            # compute intervals for MLTI
            if mlti is not None:
                mlti_t1, mlti_t2 = get_mlti_intervals(mlti.time_utc)
            else:
                mlti_t1, mlti_t2 = np.array([]), np.array([])

            # create MLTI trace
            tr_mlti = get_trace("BW.ROMY.30.MLT",
                                config['tbeg'],
                                Nsamples, 
                                sampling_rate
                                )
            
            mlti_masks[component] = mlti_intervals_to_zero(tr_mlti.data,
                                                            tr_mlti.times(reftime=config['tbeg'], type="utcdatetime"),
                                                            mlti_t1,
                                                            mlti_t2,
                                                            t_offset_sec=60
                                                            )

        # merge mlti_masks for components UVW
        mlti_masks_uvw = np.zeros(Nsamples)
        for component in components:
            if component != "Z" and config['keep_z']:
                continue
            mlti_masks_uvw += mlti_masks[component]

        # remove periods with value 2 due to summation
        mlti_masks_uvw = np.where(mlti_masks_uvw >= 1, 1, mlti_masks_uvw)

        # # make mlti trace for horizontal rings > rotation makes all affected by MLTI on single rings
        # Hexists = False
        # for component in components:
        #     if component != "Z" and not Hexists:
        #         print(component)
        #         tr_mltiH = tr_mlti[component].copy()
        #         Hexists = True
        # for component in components:
        #     if config['keep_z'] and component == "Z":
        #         continue
        #     # combine mlti periods
        #     tr_mltiH += tr_mlti[component]

        # # remove periods with value 2 due to summation
        # tr_mltiH.data = np.where(tr_mltiH.data >= 1, 1, tr_mltiH.data)

        # apply MLTI mask
        for component, tr in zip(components, st0):
            if component == "Z" and not config['keep_z']:
                tr.data = ma.masked_array(tr.data, mask=mlti_masks[component])
            else:
                # create masked array
                masked_data = ma.masked_array(tr.data, mask=mlti_masks_uvw)
                # avoid all masked data
                if masked_data.mask.all():
                    tr.data = np.full_like(tr.data, np.nan)
                else:
                    tr.data = masked_data

    if config['apply_lxx_mask']:
        if config['verbose']:
            print(f" -> preparing and applying LXX mask...")

        # load maintenance file
        try:
            lxx = load_lxx(config['tbeg'], config['tend'], config['path_to_sds'])
        except:
            print(f" -> failed to load LXX logs!")

        # prepare maintenance LXX mask
        if lxx is not None:
            lxx_t1, lxx_t2 = get_mlti_intervals(lxx.datetime)
        else:
            lxx_t1, lxx_t2 = np.array([]), np.array([])

        # create LXX trace
        tr_lxx = get_trace("BW.ROMY.30.LXX", 
                           config['tbeg'],
                           Nsamples, 
                           sampling_rate
                           )
        
        tr_lxx.data = mlti_intervals_to_zero(tr_lxx.data,
                                            tr_lxx.times(reftime=config['tbeg'], type="utcdatetime"),
                                            lxx_t1,
                                            lxx_t2,
                                            t_offset_sec=60
                                            )

        # apply LXX mask
        for component, tr in zip(components, st0):
            if component == "Z" and not config['keep_z']:
                tr.data = ma.masked_array(tr.data, mask=tr_lxx.data)
            else:
                # create masked array
                masked_data = ma.masked_array(tr.data, mask=tr_lxx.data)
                # avoid all masked data
                if masked_data.mask.all():
                    tr.data = np.full_like(tr.data, np.nan)
                else:
                    tr.data = masked_data

    # write output to sds
    if config['verbose']:
        print(f" -> writing output to sds: {config['path_to_sds_out']}")
    
    write_stream_to_sds(st0, config['loc_out'], config['path_to_sds_out'])


if __name__ == "__main__":


    # Create argument parser
    parser = argparse.ArgumentParser(description='Process ROMY data for a specific date or date range.')
    parser.add_argument('-d', '--date', type=str, help='Date in format YYYY-MM-DD or YYYYMMDD, or date range as YYYY-MM-DD,YYYY-MM-DD')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no date argument is provided, show help and exit
    if args.date is None:
        parser.print_help()
        print("\nError: Date argument is required.")
        sys.exit(1)
    else:
        config['tbeg'] = UTCDateTime(args.date)
        config['tend'] = config['tbeg'] + 86400

    main(config)

# End of File
