#!/usr/bin/env python3
# coding: utf-8
"""
Compute Sagnac Frequency

This script processes ROMY ring data to compute Sagnac frequencies
using parallel processing for efficiency.
"""

import os
import gc
import sys
import shutil
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from pprint import pprint


from obspy import UTCDateTime, Stream
from obspy.clients.filesystem.sds import Client
from astropy.time import Time
from scipy.signal import hilbert

# Configuration
archive_path = "/import/freenas-ffb-01-data/"

config = {}

# Parse command line arguments
if len(sys.argv) < 2:
    raise ValueError("Usage: python script.py <ring> [date]")
    
config['ring'] = sys.argv[1]

# Specify seed IDs for [sagnac, monobeam1, monobeam2]
config['seeds'] = [
    f"BW.DROMY..FJ{config['ring']}",
    f"BW.DROMY..F1{config['ring']}",
    f"BW.DROMY..F2{config['ring']}"
]
if config['ring'] == "Z":
    config['seeds'] = [
        f"BW.DROMY..FJ{config['ring']}",
        "BW.DROMY..F1V",
        "BW.DROMY..F2V"
    ]

config['n_cpu'] = 3

# Parse date or use default
if len(sys.argv) >= 3:
    config['tbeg'] = UTCDateTime(sys.argv[2]).date
else:
    config['tbeg'] = (UTCDateTime.now() - 40000).date

config['tend'] = config['tbeg']

# Output configuration
config['outfile_appendix'] = ""
config['path_to_sds'] = os.path.join(archive_path, "romy_archive")
config['outpath_data'] = os.path.join(
    archive_path, "romy_autodata", str(config['tbeg'].year), f"R{config['ring']}"
)
config['outpath_logs'] = os.path.join(
    archive_path, "romy_autodata", str(config['tbeg'].year), "logfiles"
)
config['logfile'] = f"{config['ring']}_{config['tbeg'].year}_autodata.log"

# Processing parameters
config['conversion'] = 0.59604645e-6  # Convert from V to count [0.59604645ug from obsidian]
config['repository'] = "archive"
config['method'] = "hilbert"

# Sagnac frequency configuration
rings = {"Z": 553, "U": 302, "V": 448, "W": 448}
config['f_expected'] = rings[config['ring']]  # Expected Sagnac frequency in Hz
config['f_band'] = 3  # Frequency band around Sagnac frequency (+- Hz)

# Time window configuration
config['t_steps'] = 60  # Time steps in seconds
config['t_overlap'] = 180  # Time overlap in seconds
config['time_interval'] = 3600  # Time interval for data chunks in seconds
config['NN'] = int(config['time_interval'] / config['t_steps'])


def read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED"):
    """
    Read waveform data from SDS archive.
    
    Parameters:
    -----------
    path_to_archive : str
        Path to SDS archive
    seed : str
        SEED ID (format: NET.STA.LOC.CHA)
    tbeg : UTCDateTime or str
        Start time
    tend : UTCDateTime or str
        End time
    data_format : str
        Data format (default: "MSEED")
    
    Returns:
    --------
    Stream
        ObsPy Stream object with waveform data
    """
    tbeg = UTCDateTime(tbeg)
    tend = UTCDateTime(tend)
    
    if not os.path.exists(path_to_archive):
        logging.warning(f"Archive path does not exist: {path_to_archive}")
        return Stream()
    
    # Parse SEED ID
    try:
        net, sta, loc, cha = seed.split(".")
    except ValueError:
        logging.error(f"Invalid SEED ID format: {seed}")
        return Stream()
    
    # Create SDS client and read waveforms
    try:
        client = Client(path_to_archive, sds_type='D', format=data_format)
        st = client.get_waveforms(net, sta, loc, cha, tbeg, tend, merge=-1)
        return st
    except Exception as e:
        logging.error(f"Failed to obtain waveforms for {seed}: {e}")
        return Stream()


def hilbert_frequency_estimator(config, st, fs):
    """
    Estimate frequency using Hilbert transform method.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    st : Stream
        ObsPy Stream object
    fs : float
        Sampling rate
    
    Returns:
    --------
    tuple
        (t_mid, f_mean, amplitude_mean, f_std)
    """
    st0 = st.copy()
    
    f_lower = config['f_expected'] - config['f_band']
    f_upper = config['f_expected'] + config['f_band']
    
    # Preprocess signal
    st0.detrend("linear")
    st0.taper(0.1)
    st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=8, zerophase=True)
    
    # Estimate instantaneous frequency using Hilbert transform
    signal = st0[0].data
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    
    # Cut first and last 5% (corrupted edges)
    dd = int(0.05 * len(instantaneous_frequency))
    insta_f_cut = instantaneous_frequency[dd:-dd]
    
    # Calculate mid time
    t = st0[0].times()
    t_mid = t[len(t) // 2]
    
    # Calculate statistics
    f_mean = np.mean(insta_f_cut)
    f_std = np.std(insta_f_cut)
    amplitude_mean = np.mean(amplitude_envelope)
    
    return t_mid, f_mean, amplitude_mean, f_std


def compute_ac_dc_contrast(st):
    """
    Compute AC, DC, and contrast from stream data.
    
    Parameters:
    -----------
    st : Stream
        ObsPy Stream object
    
    Returns:
    --------
    tuple
        (ac, dc, contrast)
    """
    st0 = st.copy()
    dat = st0[0].data
    
    percentiles = np.percentile(dat, [2.5, 97.5])
    ac = percentiles[1] - percentiles[0]
    dc = np.nanmean(dat)
    contrast = (percentiles[1] - percentiles[0]) / (percentiles[1] + percentiles[0])
    
    return ac, dc, contrast


def compute_phase(config, st0):
    """
    Compute phase shift between two monobeam channels using cross-correlation.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    st0 : Stream
        ObsPy Stream object containing monobeam data
    
    Returns:
    --------
    tuple
        (time_shift, correlation_value)
    """
    from obspy.signal.cross_correlation import correlate, xcorr_max
    
    st_tmp = st0.copy()
    df_new = st_tmp[0].stats.sampling_rate * 10
    
    mono1 = config['seeds'][1].split(".")[3]
    mono2 = config['seeds'][2].split(".")[3]
    
    f1 = st_tmp.select(channel=mono1).detrend("demean").normalize().interpolate(df_new)[0]
    f2 = st_tmp.select(channel=mono2).detrend("demean").normalize().interpolate(df_new)[0]
    
    cc = correlate(f1.data, f2.data, int(len(f1.data) / 2))
    shift, value = xcorr_max(cc)
    
    time_shift = shift / df_new
    return time_shift, value


def compute(config, st0, starttime, method="hilbert"):
    """
    Compute frequency and signal parameters for time windows.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    st0 : Stream
        ObsPy Stream object
    starttime : UTCDateTime
        Start time for processing
    method : str
        Method to use (default: "hilbert")
    
    Returns:
    --------
    tuple
        (tt0, tt1, tt2, ff, hh, pp, ac, dc, con)
    """
    NN = config['NN']
    
    # Initialize arrays
    tt0 = np.zeros(NN)
    tt1 = np.zeros(NN, dtype=object)  # Will hold strings
    tt2 = np.zeros(NN)
    ff = np.zeros(NN)
    hh = np.zeros(NN)
    pp = np.zeros(NN)
    ac = np.zeros(NN)
    dc = np.zeros(NN)
    con = np.zeros(NN)
    
    n1 = 0
    n2 = config['t_steps']
    
    for ii in range(NN):
        try:
            # Cut stream to chunks with overlap
            t_start = starttime + n1 - config['t_overlap'] / 2
            t_end = starttime + n1 + config['t_steps'] + config['t_overlap'] / 2
            st_tmp = st0.copy().trim(t_start, t_end)
            
            if len(st_tmp) == 0:
                raise ValueError("Empty stream after trimming")
            
            # Get sampling rate
            df = st_tmp[0].stats.sampling_rate
            
            # Compute frequency based on method
            if method == "hilbert":
                f_tmp, f_max, p_max, h_tmp = hilbert_frequency_estimator(config, st_tmp, df)
            else:
                logging.warning(f"Unknown method: {method}")
                continue
            
            # Get time arrays
            times_utc = st_tmp[0].times("utcdatetime")
            times_sec = st_tmp[0].times()
            
            # Compute AC, DC, and contrast
            ac0, dc0, con0 = compute_ac_dc_contrast(st_tmp)
            
            # Store values
            mid_idx = len(times_sec) // 2
            tt0[ii] = times_sec[mid_idx]
            tt1[ii] = str(times_utc[mid_idx])
            tt2[ii] = utc_to_mjd(times_utc[mid_idx])
            ff[ii] = f_max
            pp[ii] = p_max
            hh[ii] = h_tmp
            ac[ii] = ac0
            dc[ii] = dc0
            con[ii] = con0
            
        except Exception as e:
            logging.warning(f"Computing failed for window {ii}: {e}")
            tt0[ii] = np.nan
            tt1[ii] = ""
            tt2[ii] = np.nan
            ff[ii] = np.nan
            pp[ii] = np.nan
            hh[ii] = np.nan
            ac[ii] = np.nan
            dc[ii] = np.nan
            con[ii] = np.nan
        
        n1 += config['t_steps']
        n2 += config['t_steps']
    
    return tt0, tt1, tt2, ff, hh, pp, ac, dc, con


def utc_to_mjd(datetime_input):
    """
    Convert UTC datetime to Modified Julian Date.
    
    Parameters:
    -----------
    datetime_input : UTCDateTime, str, or list
        Input datetime(s)
    
    Returns:
    --------
    float or list
        Modified Julian Date(s)
    """
    if isinstance(datetime_input, list):
        return [Time(str(UTCDateTime(dt)), format='isot', scale='utc').mjd 
                for dt in datetime_input]
    else:
        return Time(str(UTCDateTime(datetime_input)), format='isot', scale='utc').mjd


def join_pickle_files(config):
    """
    Join all pickle files in tmp directory into a single DataFrame.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    DataFrame
        Combined DataFrame sorted by times_mjd
    """
    tmp_dir = os.path.join(config['outpath_data'], "tmp")
    
    if not os.path.exists(tmp_dir):
        logging.warning(f"Temporary directory does not exist: {tmp_dir}")
        return pd.DataFrame()
    
    files = sorted([f for f in os.listdir(tmp_dir) if f.endswith(".pkl")])
    
    if not files:
        logging.warning("No pickle files found in tmp directory")
        return pd.DataFrame()
    
    dataframes = []
    for filex in files:
        try:
            df = pd.read_pickle(os.path.join(tmp_dir, filex))
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Failed to read {filex}: {e}")
    
    if not dataframes:
        return pd.DataFrame()
    
    data = pd.concat(dataframes, ignore_index=True)
    data.sort_values(by="times_mjd", inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    return data


def main(times):
    """
    Main processing function for parallel execution.
    
    Parameters:
    -----------
    times : tuple
        (counter, tbeg, tend)
    """
    jj, tbeg, tend = times
    
    # Format counter as 3-digit string
    jj = str(jj).rjust(3, "0")
    
    # Convert times
    t1 = UTCDateTime(tbeg)
    t2 = UTCDateTime(tend)
    
    # Initialize arrays for all channels (fixes "referenced before assignment" error)
    NN = config['NN']
    fj = np.zeros(NN)
    f1 = np.zeros(NN)
    f2 = np.zeros(NN)
    pz = np.zeros(NN)
    p1 = np.zeros(NN)
    p2 = np.zeros(NN)
    ac_z = np.zeros(NN)
    ac_1 = np.zeros(NN)
    ac_2 = np.zeros(NN)
    dc_z = np.zeros(NN)
    dc_1 = np.zeros(NN)
    dc_2 = np.zeros(NN)
    con_z = np.zeros(NN)
    con_1 = np.zeros(NN)
    con_2 = np.zeros(NN)
    tt_utc = np.zeros(NN, dtype=object)
    tt_sec = np.zeros(NN)
    tt_mjd = np.zeros(NN)
    
    # Process each seed
    for seed in config['seeds']:
        loading_data_status = False
        
        try:
            # Load data
            st = read_sds(
                config['path_to_sds'],
                seed,
                t1 - 2 * config['t_overlap'],
                t2 + 2 * config['t_overlap'],
                data_format='MSEED'
            )
            
            if len(st) == 0:
                raise ValueError("Empty stream returned")
            
            # Convert from V to count
            st[0].data = st[0].data * config['conversion']
            loading_data_status = True
            
        except Exception as e:
            logging.error(f"Failed to load data for {seed}: {e}")
            loading_data_status = False
        
        # Compute signal values
        if loading_data_status:
            try:
                tt_sec, tt_utc, tt_mjd, ff, hh, pp, ac, dc, con = compute(
                    config, st, t1, method=config['method']
                )
            except Exception as e:
                logging.error(f"Computation failed for {seed}: {e}")
                # Use zeros as fallback
                tt_sec = np.zeros(NN)
                tt_utc = np.zeros(NN, dtype=object)
                tt_mjd = np.zeros(NN)
                ff = np.zeros(NN)
                hh = np.zeros(NN)
                pp = np.zeros(NN)
                ac = np.zeros(NN)
                dc = np.zeros(NN)
                con = np.zeros(NN)
        else:
            # Use zeros if loading failed
            ff = np.zeros(NN)
            hh = np.zeros(NN)
            pp = np.zeros(NN)
            ac = np.zeros(NN)
            dc = np.zeros(NN)
            con = np.zeros(NN)
        
        # Assign to appropriate channel variables
        channel = seed.split(".")[3]
        seed_channels = [s.split(".")[3] for s in config['seeds']]
        
        if channel == seed_channels[0]:  # Sagnac (FJ)
            fj = ff.copy()
            pz = pp.copy()
            ac_z = ac.copy()
            dc_z = dc.copy()
            con_z = con.copy()
        elif channel == seed_channels[1]:  # Monobeam 1 (F1)
            f1 = ff.copy()
            p1 = pp.copy()
            ac_1 = ac.copy()
            dc_1 = dc.copy()
            con_1 = con.copy()
        elif channel == seed_channels[2]:  # Monobeam 2 (F2)
            f2 = ff.copy()
            p2 = pp.copy()
            ac_2 = ac.copy()
            dc_2 = dc.copy()
            con_2 = con.copy()
    
    # Create DataFrame
    df = pd.DataFrame({
        'times_utc': tt_utc,
        'times_utc_sec': tt_sec,
        'times_mjd': tt_mjd,
        'fj': fj,
        'f1': f1,
        'f2': f2,
        'pz': pz,
        'p1': p1,
        'p2': p2,
        'ac_z': ac_z,
        'ac_1': ac_1,
        'ac_2': ac_2,
        'dc_z': dc_z,
        'dc_1': dc_1,
        'dc_2': dc_2,
        'contrast_z': con_z,
        'contrast_1': con_1,
        'contrast_2': con_2,
    })
    
    # Clean up memory
    try:
        del st, ff, hh, pp, ac, dc, con
        gc.collect()
    except NameError:
        pass
    
    # Write DataFrame to pickle file
    output_file = os.path.join(
        config['outpath_data'],
        "tmp",
        f"tmp_{tbeg}_{tend}_{jj}.pkl"
    )
    df.to_pickle(output_file, protocol=4)


if __name__ == '__main__':
    # Print configuration
    pprint(config)
    
    # Create output directories
    os.makedirs(config['outpath_data'], exist_ok=True)
    os.makedirs(config['outpath_logs'], exist_ok=True)
    
    # Setup logging
    logfile_path = os.path.join(config['outpath_logs'], config['logfile'])
    if os.path.isfile(logfile_path):
        os.remove(logfile_path)
    
    logging.basicConfig(
        filename=logfile_path,
        format="%(asctime)s,%(levelname)s,%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S",
        level=logging.DEBUG
    )
    logging.info("__________ START ______________")
    logging.info(f"Processing ring {config['ring']} for date {config['tbeg']}")
    logging.info(f"Start time: {UTCDateTime.now()}")
    
    # Create/clean tmp directory
    tmp_dir = os.path.join(config['outpath_data'], "tmp")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Generate time windows for processing
    list_of_times = []
    for j, date in enumerate(pd.date_range(config['tbeg'], config['tend'])):
        date = UTCDateTime(date.date())
        hh = 0
        counter = 0
        
        while hh < 86400:
            tbeg = date - config['t_overlap'] + hh
            tend = date + config['t_overlap'] + hh + config['time_interval']
            list_of_times.append((counter, tbeg, tend))
            hh += config['time_interval']
            counter += 1
    
    # Launch parallel processes
    logging.info(f"Processing {len(list_of_times)} time windows with {config['n_cpu']} CPUs")
    
    import multiprocessing as mp
    with mp.Pool(processes=config['n_cpu']) as pool:
        pool.map(main, list_of_times)
    
    # Join pickle files
    logging.info("Joining pickle files to master file...")
    df_all = join_pickle_files(config)
    
    # Store master DataFrame
    if not df_all.empty:
        date_str = str(config['tbeg']).replace("-", "")
        output_file = os.path.join(
            config['outpath_data'],
            f"FJ{config['ring']}_{date_str}{config['outfile_appendix']}.pkl"
        )
        df_all.to_pickle(output_file)
        logging.info(f"Master file written: {output_file}")
        logging.info(f"Total records: {len(df_all)}")
    else:
        logging.warning("No data to write - DataFrame is empty")
    
    logging.info(f"Done at {UTCDateTime.now()}")
