"""

Run backscatter quantity computation and correction automatically

"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
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

# extract date
# config['tbeg'] = UTCDateTime(sys.argv[1])
# config['tend'] = config['tbeg'] + 86400

config['tbeg'] = UTCDateTime("2024-10-23 00:00")
config['tend'] = UTCDateTime("2024-10-24 00:00")

# project name
config['project'] = "_testrun"

# select if progress bar is shown
config['show_progress'] = True

# select if info is printed
config['verbose'] = True

# store configurations as file
config['store_config'] = True

# extract ring
# config['ring'] = sys.argv[2]
config['ring'] = "Z"

# select frequency estimation mode
config['mode'] = "hilbert" # "hilbert" | "sine"

# specify seed codes
config['seeds'] = [f"BW.DROMY..FJ{config['ring']}", "BW.DROMY..F1V", "BW.DROMY..F2V"]

# specify interval of data to load (in seconds)
config['time_interval'] = 3600

# specify time interval in seconds
config['interval'] = 10

# set if amplitdes are corrected with envelope
config['correct_amplitudes'] = True

# set prewhitening factor (to avoid division by zero)
config['prewhitening'] = 0.001

# interval buffer (before and after) in seconds
config['ddt'] = 1800

# frequency band (minus and plus)
config['fband'] = 1 # 10

# specify cm filter value for backscatter correction
config['cm_value'] = 1.033

# define nominal sagnac frequency of rings
config['ring_sagnac'] = {"U":303.05, "V":447.5, "W":447.5, "Z":553.5}
config['nominal_sagnac'] = config['ring_sagnac'][config['ring']]

# specify path to Sagnac data
config['path_to_data'] = data_path+"sagnac_frequency/data/compare_methods/"
#config['path_to_data'] = data_path+"sagnac_frequency/data/"

# specify path to output figures
config['path_to_figs'] = data_path+"sagnac_frequency/figures/"

# specify path to sds data archive
config['path_to_sds'] = archive_path+"romy_archive/"

# specify amount of cpu kernels to use for parallel processing
config['n_cpu'] = 3

# ______________
# automatic settings

# set a label
if config['correct_amplitudes']:
    config['alabel'] = "_CA"
else:
    config['alabel'] = ""



# ______________________________________

class sagnacdemod:

    def __init__(self, config=None, output_sampling_rate=1, ddt=100, ring="Z"):

        from obspy import Stream

        if config is not None:
            for k in config.keys():
                self.k = config[k]

        # specify output sampling rate
        self.output_sampling_rate = output_sampling_rate

        # specify ring
        self.ring = ring

        # ROMY conversion for Obsidian
        self.conversion = 0.59604645e-6 # V / count  [0.59604645ug  from obsidian]

        # specify nominal sagnac frequencies to expect
        self.nominal_sagnac = {"U":303.05, "V":447.5, "W":447.5, "Z":553.5}

        # time buffer
        self.ddt = ddt

        # assign instantaneous frequency stream
        self.fstream = Stream()

    def save_to_pickle(self, obj, path, name):

        import os
        import pickle

        ofile = open(path+name+".pkl", 'wb')
        pickle.dump(obj, ofile)

        if os.path.isfile(path+name+".pkl"):
            print(f"\n -> created:  {path}{name}.pkl")

    def load_sagnac_data(self, seed, tbeg, tend, path_to_sds, output=True, verbose=False):

        import os
        from obspy import Stream, UTCDateTime
        from obspy.clients.filesystem.sds import Client

        self.seed = seed
        self.tbeg = UTCDateTime(tbeg)
        self.tend = UTCDateTime(tend)
        self.path_to_sds = path_to_sds

        if verbose:
            print(f" -> loading {seed}...")

        if not os.path.exists(self.path_to_sds):
            print(f" -> {self.path_to_sds} does not exist!")
            return
        
        # separate seed id
        self.net, self.sta, self.loc, self.cha = seed.split(".")

        # define SDS client
        client = Client(self.path_to_sds, sds_type='D', format='MSEED')

        # read waveforms
        try:
            st0 = client.get_waveforms(self.net, self.sta, self.loc, self.cha, 
                                       self.tbeg-self.ddt, self.tend+self.ddt, 
                                       merge=-1
                                       )
    
        except:
            print(f" -> failed for {seed}")
            return
        
        st0 = st0.sort()

        for tr in st0:
            tr.data = tr.data * self.conversion

        self.st0 = st0

        if output:
            return st0

    def hilbert_estimator(self, fband=10, acorrect=False, prewhiten=None):

        import numpy as np
        from scipy.signal import hilbert

        # assign values
        self.amplitude_correction = acorrect
        self.fband = fband
        self.prewhiten = prewhiten

        # get copy of data stream
        # _st = self.st0.copy()

        # extract sampling rate
        self.df0 = self.st0[0].stats.sampling_rate

        # define frequency band around Sagnac Frequency
        f_lower = self.nominal_sagnac[self.ring] - fband
        f_upper = self.nominal_sagnac[self.ring] + fband

        # bandpass with butterworth around Sagnac Frequency
        self.st0 = self.st0.detrend("linear")
        self.st0 = self.st0.taper(0.01, type="cosine")
        self.st0 = self.st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=4, zerophase=True)

        # estimate instantaneous frequency with hilbert
        signal = self.st0[0].data

        # compute analytic signal
        analytic_signal = hilbert(signal)

        # compute envelope
        envelope = np.abs(analytic_signal)

        # correct for amplitude variations
        if self.amplitude_correction:
            if self.prewhiten is None:
                self.prewhiten = 0.001

            signal = signal / (envelope + self.prewhiten)

            # recompute analytic signal
            analytic_signal = hilbert(signal)

        # estimate instantaneous phase (as radian -> deg=False)
        insta_phase = np.unwrap(np.angle(analytic_signal, deg=False))

        # estimate instantaneous frequeny
        # insta_f = (np.diff(insta_phase) / (2.0*np.pi) * df)
        insta_f = (np.gradient(insta_phase) / (2.0*np.pi) * self.df0)

        # cut time buffer
        ncut = int(self.ddt*self.df0*0.5)
        insta_f_cut = insta_f[ncut:-ncut]

        # get times
        times = self.st0[0].times()
        times_cut = times[ncut:-ncut]

        # get utc starttime
        self.st0 = self.st0.trim(self.tbeg, self.tend)
        self.starttime = self.st0[0].stats.starttime

        # assgin data
        self.insta_f = insta_f_cut
        self.insta_p = insta_phase
        self.times = times_cut

        # remove stream to release memory
        self.st0 = None

    def get_stream(self, df, output=True):

        from obspy import Stream, Trace

        if df >= 50:
            c = "H"
        elif df < 50 and df > 1:
            c = "B"
        else:
            c = "L"

        # create stream object
        tr = Trace()
        tr.stats.network = "BW"
        tr.stats.station = "ROMY"
        tr.stats.location = self.loc
        tr.stats.channel = c+self.cha[1:]

        tr.data = self.insta_f
        tr.stats.sampling_rate = self.df0
        tr.stats.starttime = self.starttime

        tr = tr.detrend("demean")
        tr = tr.taper(0.01, type="cosine")
        tr = tr.filter("lowpass", freq=500, corners=4, zerophase=True)
        tr = tr.resample(1000, no_filter=True)

        tr = tr.detrend("demean")
        tr = tr.filter("lowpass", freq=300, corners=4, zerophase=True)
        tr = tr.resample(600, no_filter=True)

        tr = tr.detrend("demean")
        tr = tr.filter("lowpass", freq=df/2, corners=4, zerophase=True)
        tr = tr.resample(df, no_filter=True)

        tr = tr.trim(self.tbeg, self.tend)

        self.fstream += tr

        self.fstream.merge()

        if output:
            return self.fstream

    def write_stream_to_sds(self, path_to_out_sds):

        import os

        ## check if output path exists
        if not os.path.exists(path_to_out_sds):
            print(f" -> {path_to_out_sds} does not exist!")
            return

        self.fstream = self.fstream.merge()

        for tr in self.fstream:
            nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
            yy, jj = tr.stats.starttime.year, tr.stats.starttime.julday

            if not os.path.exists(path_to_out_sds+f"{yy}/"):
                os.mkdir(path_to_out_sds+f"{yy}/")
                print(f"creating: {path_to_out_sds}{yy}/")
            if not os.path.exists(path_to_out_sds+f"{yy}/{nn}/"):
                os.mkdir(path_to_out_sds+f"{yy}/{nn}/")
                print(f"creating: {path_to_out_sds}{yy}/{nn}/")
            if not os.path.exists(path_to_out_sds+f"{yy}/{nn}/{ss}/"):
                os.mkdir(path_to_out_sds+f"{yy}/{nn}/{ss}/")
                print(f"creating: {path_to_out_sds}{yy}/{nn}/{ss}/")
            if not os.path.exists(path_to_out_sds+f"{yy}/{nn}/{ss}/{cc}.D"):
                os.mkdir(path_to_out_sds+f"{yy}/{nn}/{ss}/{cc}.D")
                print(f"creating: {path_to_out_sds}{yy}/{nn}/{ss}/{cc}.D")

        for tr in self.fstream:
            nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
            yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

            try:
                st_tmp = self.fstream.copy()
                st_tmp = st_tmp.select(network=nn, station=ss, location=ll, channel=cc)
                st_tmp.write(path_to_out_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")
            except:
                print(f" -> failed to write: {cc}")
            finally:
                print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")

    @staticmethod
    def get_time_intervals(tbeg, tend, interval_seconds, interval_overlap=0, output=True):

        from obspy import UTCDateTime

        tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

        times = []
        t1, t2 = tbeg, tbeg + interval_seconds
        while t2 <= tend:
            times.append((t1, t2))
            t1 = t1 + interval_seconds - interval_overlap
            t2 = t2 + interval_seconds - interval_overlap

        if output:
            return times


# _____________________________________________________________________________________

def main(config):


    sagnac = sagnacdemod(output_sampling_rate=1.0,
                         ddt=config.get('ddt'),
                         ring=config.get('ring')
                         )


    intervals = sagnac.get_time_intervals(config.get('tbeg'),config.get('tend'), config.get('time_interval'))

    for t1, t2 in tqdm(intervals):

        sagnac.load_sagnac_data(config['seeds'][0],
                                t1,
                                t2,
                                config.get('path_to_sds'),
                                verbose=False,
                                )


        sagnac.hilbert_estimator(fband=config.get('fband'),
                                 acorrect=False,
                                 prewhiten=None,
                                 )

        sagnac.get_stream(df=20)


    opath = "/import/kilauea-data/sagnac_frequency/data/demod/"
    sagnac.write_stream_to_sds(opath)

    sagnac.fstream.plot()


# ________ MAIN  ________
if __name__ == "__main__":

    main(config)

# End of File
