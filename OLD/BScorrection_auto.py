"""
 Run backscatter quantity computation and correction automatically

"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, date
from pandas import DataFrame, read_pickle, date_range, concat, read_csv
from obspy import UTCDateTime, read
from scipy.signal import hilbert


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


# ______________________________________
# Configurations

config = {}

config['tbeg'] = UTCDateTime(sys.argv[1])
config['tend'] = config['tbeg']+86400

config['ring'] = sys.argv[2]

config['seeds'] = [f"BW.DROMY..FJ{config['ring']}", "BW.DROMY..F1V", "BW.DROMY..F2V"]

config['interval'] = 60

# interval buffer (before and after) in seconds
config['ddt'] = 30

# frequency band (minus and plus)
config['fband'] = 2 # 10

# define nominal sagnac frequency of rings
config['ring_sagnac'] = {"U":303.05, "V":447.5, "W":447.5, "Z":553.5}
config['nominal_sagnac'] = config['ring_sagnac'][config['ring']]

# path to Sagnac data
config['path_to_autodata'] = archive_path+f"romy_autodata/"

config['path_to_data'] = data_path+"sagnac_frequency/data/"

config['path_to_figs'] = data_path+"sagnac_frequency/figures/"

config['path_to_sds'] = archive_path+"romy_archive/"


# ______________________________________
# methods

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

def __load_romy_raw_data(seed, tbeg, tend, path_to_sds):

    from obspy import Stream, UTCDateTime

    print(f" -> loading {seed}...")

    try:
        st00 = __read_sds(path_to_sds, seed, tbeg,tend, data_format='MSEED')
    except:
        print(f" -> failed for {seed}")

    st0 = st00.sort()

    for tr in st0:
        tr.data = tr.data*0.59604645e-6 # V / count  [0.59604645ug  from obsidian]

    return st0

def __get_time_intervals(tbeg, tend, interval_seconds, interval_overlap):

    from obspy import UTCDateTime

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    times = []
    t1, t2 = tbeg, tbeg + interval_seconds
    while t2 <= tend:
        times.append((t1, t2))
        t1 = t1 + interval_seconds - interval_overlap
        t2 = t2 + interval_seconds - interval_overlap

    return times

def __hilbert_frequency_estimator(st, nominal_sagnac, fband=10, cut=0):

    from scipy.signal import hilbert
    import numpy as np

    st0 = st.copy()

    ## extract sampling rate
    df = st0[0].stats.sampling_rate

    ## define frequency band around Sagnac Frequency
    f_lower = nominal_sagnac - fband
    f_upper = nominal_sagnac + fband

    ## bandpass with butterworth around Sagnac Frequency
    st0.detrend("linear")
    st0.taper(0.01)
    st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=4, zerophase=True)


    ## estimate instantaneous frequency with hilbert
    signal = st0[0].data

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * df)

    ## cut first and last 5% (corrupted data)
    # dd = int(0.05*len(instantaneous_frequency))
    dd = int(cut*df)
    insta_f_cut = instantaneous_frequency[dd:-dd]

    ## get times
    t = st0[0].times()
    t_mid = t[int((len(t))/2)] + config['ddt']

    ## averaging of frequencies
    # insta_f_cut_avg = np.mean(insta_f_cut)
    insta_f_cut_avg = np.median(insta_f_cut)

    ## standard error
    insta_f_cut_std = np.std(insta_f_cut)

    return t_mid, insta_f_cut_avg, np.mean(amplitude_envelope), insta_f_cut_std

def __backscatter_correction(m01, m02, phase0, w_obs, fs0, cm_filter_factor=1.033):

    ## Correct for bias
    m1 = m01 * ( 1 + m01**2 / 4 )
    m2 = m02 * ( 1 + m02**2 / 4 )

    ## angular correction for phase
    phase = phase0 + 0.5 * m1 * m2 * np.sin( phase0 )

    ## compute squares of common-mode modulations
    m2c = ( m1**2 + m2**2 + 2*m1*m2*np.cos( phase ) ) / 4

    ## compute squares of differential-mode modulations
    m2d = ( m1**2 + m2**2 - 2*m1*m2*np.cos( phase ) ) / 4  ## different angle!

    ## correct m2c for gain saturation of a HeNe laser
    # m2c = m2c * ( 1 + ( beta + theta )**2 * fL**2 * I0**2 / ws**2 )
    m2c = m2c * cm_filter_factor

    ## compute backscatter correction factor
    M = m2c - m2d + 0.25 * m1**2 * m2**2 * np.sin(phase)**2

    ## correction term
    term = ( 4 + M ) / ( 4 - M )

    ## backscatter correction
    correction = -1 * ( term -1 ) * fs0
    # w_corrected = np.array(w_obs) + correction

    # apply backscatter correction
    w_corrected = np.array(w_obs) * term

    return w_corrected, correction, term

def __get_fft_values(signal_in, dt, f_sagn, window=None):

    from numpy import argmax, sqrt, where, argmin, gradient, mean
    from scipy.fft import fft, fftfreq, fftshift
    from scipy import signal
    from numpy import angle, imag, unwrap

    ## determine length of the input time series
    n = int(len(signal_in))

    signal_in = fftshift(signal_in)

    ## calculate spectrum (with or without window function applied to time series)
    if window:
        win = signal.get_window(window, n);
        spectrum = fft(signal_in * win, norm="forward")

    else:
        spectrum = fft(signal_in, norm="forward")

    ## calculate frequency array
    frequencies = fftfreq(n, d=dt)

    ## correct amplitudes of spectrum
    magnitude_corrected = abs(spectrum) *2 /n

    ## none corrected magnitudes
    magnitude = abs(spectrum)

    ## phase spectrum
    phase = angle(spectrum, deg=False)

    freq = frequencies[0:n//2]
    spec = magnitude[0:n//2]
    pha = phase[0:n//2]

    ## specify f-band around Sagnac frequency
    fl = f_sagn - 2
    fu = f_sagn + 2

    ## get index of Sagnac peak
    idx_fs = where(spec == max(spec[(freq > fl) & (freq < fu)]))[0][0]

    ## estimate Sagnac frequency
    f_sagn_est = freq[idx_fs]

    ## estimate AC value at Sagnac peak
    AC_est = spec[idx_fs] * 2

    ## estimate DC value at ff = 0
    DC_est = spec[0]

    ## estimate phase at Sagnac peak
    phase_est = pha[idx_fs] ## select phase of Sagnac

    return f_sagn_est, AC_est, DC_est, phase_est

def __merge_backscatter_data(tbeg, tend, ring, path_to_data):

    import shutil
    from obspy import UTCDateTime
    from datetime import date
    from pandas import read_pickle, concat, DataFrame, date_range

    t1 = date.fromisoformat(str(UTCDateTime(tbeg).date))
    t2 = date.fromisoformat(str((UTCDateTime(tend)-86400).date))

    df = DataFrame()
    for dat in date_range(t1, t2):

        print(str(dat)[:11])

        dat_str = str(dat)[:10].replace("-", "")
        file = f"FJ{ring}_{dat_str}_backscatter.pkl"

        if not os.path.isfile(path_to_data+file):
            process = True
        else:
            print(f" -> alread exists!")
            if config['overwrite']:
                process = True
            else:
                process = False

        if process:

            _path = data_path+"sagnac_frequency/data/"

            out = DataFrame()
            for m in range(24):
                hour = str(m).rjust(2, '0')+":00:00"
                filename = f"FJ{ring}_{dat_str}_{hour}_backscatter.pkl"
                try:
                    _df = read_pickle(_path+filename)
                    out = concat([out, _df])
                except:
                    print(f" -> failed: {_path}{filename}")
                    continue

            if not out.empty:
                print(f" -> write to: {_path}backscatter/FJ{ring}_{dat_str}_backscatter.pkl")
                out.to_pickle(f"{_path}backscatter/FJ{ring}_{dat_str}_backscatter.pkl")

                # move file to tmp files
                try:
                    shutil.move(_path+filename, _path+f"tmp_backscatter_R{ring}/"+filename)
                except:
                    print(f" -> failed to move file {filename}")
            else:
                continue

        try:
            df0 = read_pickle(path_to_data+file)
            df = concat([df, df0])
        except:
            print(f"error for {file}")

    df.reset_index(inplace=True)

# _____________________________________________________________________________________

def main(config):

    # hourly data because much data for memory
    hours = __get_time_intervals(config['tbeg'], config['tend'], interval_seconds=3600, interval_overlap=0)

    for _tbeg, _tend in hours:

        # get time intervals for iteration
        times = __get_time_intervals(_tbeg, _tend, interval_seconds=config['interval'], interval_overlap=0)

        # prepare output arrays
        NN = len(times)
        fs, ac, dc, ph, st = np.ones(NN)*np.nan, np.ones(NN)*np.nan, np.ones(NN)*np.nan, np.ones(NN)*np.nan, np.ones(NN)*np.nan

        ph_wrap = np.ones(NN)*np.nan

        # load data
        try:
            sagn = __load_romy_raw_data(config['seeds'][0], _tbeg-2*config['ddt'], _tend+2*config['ddt'], config['path_to_sds'])
            mon1 = __load_romy_raw_data(config['seeds'][1], _tbeg-2*config['ddt'], _tend+2*config['ddt'], config['path_to_sds'])
            mon2 = __load_romy_raw_data(config['seeds'][2], _tbeg-2*config['ddt'], _tend+2*config['ddt'], config['path_to_sds'])
        except:
            continue

        # check for correct size
        for _st in [sagn, mon1, mon2]:
            if len(_st) > 1:
                _st.merge()


        # prepare output dataframe
        out_df = DataFrame()
        out_df['time1'] = list(zip(*times))[0]
        out_df['time2'] = list(zip(*times))[1]

        for _k, _st in enumerate([sagn, mon1, mon2]):

            print(" -> processing ", _k, "...")

            for _n, (t1, t2) in enumerate(times):

                try:
                    # print(t1,t2)

                    # _dat = _st.copy().trim(t1, t2)
                    _dat = _st.copy().trim(t1-config['ddt'], t2+config['ddt'])

                    # estimate AC and DC values in frequency domain
                    fs[_n], ac[_n], dc[_n], ph[_n] = __get_fft_values(_dat[0].data,
                                                                    _dat[0].stats.delta,
                                                                    config['nominal_sagnac']
                                                                    )

                    # estimate instantaneous frequency average via hilbert
                    t, fs[_n], _, st[_n] = __hilbert_frequency_estimator(_dat,
                                                                        config['nominal_sagnac'],
                                                                        fband=config['fband'],
                                                                        cut=config['ddt']
                                                                        )
                    # t = (t2 - t2)/2 + t1

                    # estimate DC and AC based on time series (time domain)
                    # dc[_n] = np.mean(_dat)
                    # ac[_n] = np.percentile(_dat[0].data, 99.9) - np.percentile(_dat[0].data, 100-99.9)

                except:
                    print(f" -> procssing failed!")
                
                ph_wrap = ph
                ph = np.unwrap(ph)

                # fill output dataframe
                if _k == 0:
                    out_df['fj_fs'], out_df['fj_ac'], out_df['fj_dc'], out_df['fj_ph'], out_df['fj_st'] = fs, ac, dc, ph, st
                    out_df['fj_phw'] = ph_wrap
                elif _k == 1:
                    out_df['f1_fs'], out_df['f1_ac'], out_df['f1_dc'], out_df['f1_ph'], out_df['f1_st'] = fs, ac, dc, ph, st
                    out_df['f1_phw'] = ph_wrap
                elif _k == 2:
                    out_df['f2_fs'], out_df['f2_ac'], out_df['f2_dc'], out_df['f2_ph'], out_df['f2_st'] = fs, ac, dc, ph, st
                    out_df['f2_phw'] = ph_wrap

        # prepare values for backscatter correction
        m01 = out_df.f1_ac / out_df.f1_dc
        m02 = out_df.f2_ac / out_df.f2_dc
        phase0 = out_df.f1_ph - out_df.f2_ph
        w_obs = out_df.fj_fs

        out_df['w_s'], out_df['bscorrection'], out_df['term'] = __backscatter_correction(m01, m02,
                                                                                        phase0,
                                                                                        w_obs,
                                                                                        config['nominal_sagnac'],
                                                                                        cm_filter_factor=1.033
                                                                                        )

        # store data
        date_str = f"{_tbeg.year}{str(_tbeg.month).rjust(2,'0')}{str(_tbeg.day).rjust(2,'0')}"
        time_str = f"{str(_tbeg.time).split('.')[0]}"
        out_df.to_pickle(config['path_to_data']+f"FJ{config['ring']}_{date_str}_{time_str}_backscatter.pkl")
        print(f" -> writing: {config['path_to_data']}FJ{config['ring']}_{date_str}_{time_str}_backscatter.pkl")

    # load hourly data files to form one data frame
    __merge_backscatter_data(_tbeg, _tend, config['ring'], config['path_to_data'])


# ________ MAIN  ________
if __name__ == "__main__":

    main(config)

# End of File
