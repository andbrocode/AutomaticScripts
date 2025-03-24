from typing import Optional, Union, List, Tuple
from obspy import Stream, UTCDateTime
from obspy.core.trace import Trace

class sagnacdemod:
    """
    Class for demodulating Sagnac signals from ring laser data.
    
    This class provides methods for processing ring laser data to extract 
    instantaneous frequency variations using Hilbert transform techniques.
    """

    def __init__(self, config=None, output_sampling_rate: float = 1, ddt: int = 100, mode: str = "seismic", 
                 nominal_sagnacf: float = 0, loc: str = "", ring: str = "Z", adaptive_scaling: bool = False)   :
        """
        Initialize the Sagnac demodulator.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with processing parameters
        output_sampling_rate : float, default=1
            Sampling rate of output data in Hz
        ddt : int, default=100
            Time buffer in seconds for data loading
        mode : str, default="seismic"
            Processing mode ('seismic' or 'geodetic')
        nominal_sagnacf : float, optional
            Expected nominal Sagnac frequency in Hz
        loc : str, default=""
            Location code for data selection
        ring : str, default="Z"
            Ring identifier (Z, U, V, or W)
        adaptive_scaling : bool, default=False
            Whether to use adaptive amplitude scaling
        """
        
        import numpy as np
        from obspy import Stream

        if config is not None:
            for k in config.keys():
                self.k = config[k]

        # specify output sampling rate
        self.output_sampling_rate = output_sampling_rate

        # ROMY conversion for Obsidian
        self.conversion = 0.59604645e-6 # V / count  [0.59604645ug  from obsidian]

        # specify nominal sagnac frequencies to expect
        if nominal_sagnacf == 0:
            print(" -> no nominal sagnac frequency specified!")
        else:
            self.nominal_sagnac = nominal_sagnacf

        # time buffer
        self.ddt = ddt

        # assign instantaneous frequency stream
        self.fstream = Stream()

        # location code for dataset discrimiation
        self.oloc = loc

        self.onet = "BW"
        self.osta = "ROMY"

        # select which ring
        self.ring = ring

        # earth rotation rate
        self.omegaE = 2*np.pi/86400

        # set adaptive scaling
        self.adaptive_scaling = adaptive_scaling

        # select mode (seismic | geodetic)
        self.mode = mode

    def load_sagnac_data(self, seed: str, tbeg: Union[str, UTCDateTime], tend: Union[str, UTCDateTime], path_to_sds: str, output=True, verbose=False):
        """
        Load Sagnac data from SDS archive.

        Parameters
        ----------
        seed : str
            SEED identifier (NET.STA.LOC.CHA)
        tbeg : str or UTCDateTime
            Start time
        tend : str or UTCDateTime
            End time
        path_to_sds : str
            Path to SDS archive
        output : bool, default=True
            Return loaded stream if True
        verbose : bool, default=False
            Print additional information if True

        Returns
        -------
        obspy.Stream or None
            Loaded data stream if output=True
        """
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
        self.net, self.sta, _loc, self.cha = seed.split(".")

        # define SDS client
        client = Client(self.path_to_sds, sds_type='D', format='MSEED')

        # read waveforms
        try:
            st0 = client.get_waveforms(self.net, self.sta, _loc, self.cha,
                                       self.tbeg-self.ddt, self.tend+self.ddt,
                                       merge=-1
                                       )

        except:
            print(f" -> failed for {seed}")
            return

        st0 = st0.sort()

        self.sampling_rate = st0[0].stats.sampling_rate

        for tr in st0:
           tr.data = tr.data * self.conversion

        self.st0 = st0.copy()

        if output:
            return st0

    def hilbert_estimator(self, fband: float = 10, acorrect: bool = False, prewhiten: Optional[float] = None, method: int = 1):
        """
        Estimate instantaneous frequency using Hilbert transform.

        Parameters
        ----------
        fband : float, default=10
            Frequency band width around nominal Sagnac frequency
        acorrect : bool, default=False
            Apply amplitude correction if True
        prewhiten : float, optional
            Prewhitening factor for amplitude correction
        method : int, default=1
            Method for frequency estimation:
            1: Phase gradient method
            2: Complex signal method

        Notes
        -----
        The method applies bandpass filtering around the nominal Sagnac frequency
        before computing the instantaneous frequency using the Hilbert transform.
        """
        import numpy as np
        from scipy.signal import hilbert

        # assign values
        self.amplitude_correction = acorrect
        self.fband = fband
        self.prewhiten = prewhiten

        # get copy of data stream
        # _st = self.st0.copy()

        # define frequency band around Sagnac Frequency
        f_lower = self.nominal_sagnac - fband
        f_upper = self.nominal_sagnac + fband

        # extract sampling rate
        if isinstance(self.st0, Stream) and len(self.st0) > 0:
            self.df0 = self.sampling_rate

            # bandpass with butterworth around Sagnac Frequency
            self.st0 = self.st0.detrend("linear")
            self.st0 = self.st0.taper(0.01, type="cosine")
            self.st0 = self.st0.filter("bandpass", freqmin=f_lower, freqmax=f_upper, corners=4, zerophase=True)

            # estimate instantaneous frequency with hilbert
            signal = self.st0[0].data
        else:
            print(" -> no data stream provided!")
            return

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

        # compute instantaneous phase
        insta_phase = np.unwrap(np.angle(analytic_signal, deg=False))

        # method 1
        # !!! scaling is off by a factor of about 0.72 !!!
        if method == 1:
            insta_f = (np.gradient(insta_phase) / (2.0*np.pi) * self.df0)

            # was a test to adjust the scaling .. no effect
            # insta_f = (np.gradient(insta_phase) / (2.0*np.pi*np.sqrt(signal**2 + np.real(hilbert(signal))**2)))

        # method 2
        # !!! does not work for some reason at the moment !!!
        elif method == 2:
            h = hilbert(signal)
            grad_h = np.gradient(np.real(h), axis=0)
            grad_s = np.gradient(signal, axis=0)
            insta_f = (grad_h * signal - grad_s * h) / (2.0*np.pi * np.sqrt(signal**2 + np.real(h)**2))

        # cut time buffer
        tcut = self.ddt*0.5
        ncut = int(tcut*self.df0)
        insta_f_cut = insta_f[ncut:-ncut]

        # get times
        times = self.st0[0].times()
        times_cut = times[ncut:-ncut]

        # get utc starttime
        self.st0 = self.st0.trim(self.tbeg-tcut, self.tend+tcut)
        self.starttime = self.st0[0].stats.starttime

        # assgin data
        self.insta_f = insta_f_cut
        self.insta_p = insta_phase
        self.times = times_cut

        # remove stream to release memory
        self.st0 = None

    def get_stream(self, df: float, output: bool = True):

        import numpy as np
        from obspy import Stream, Trace

        if df >= 50:
            c = "H"
        elif df < 50 and df > 1:
            c = "B"
        else:
            c = "L"

        # Nyquist frequency
        fnqst = 0.5*df

        # create stream object
        tr = Trace()
        tr.stats.network = self.onet
        tr.stats.station = self.osta
        tr.stats.location = self.oloc
        tr.stats.channel = c+self.cha[1:]

        tr.data = self.insta_f
        tr.stats.sampling_rate = self.df0
        tr.stats.starttime = self.starttime

        # get scalefactor
        sf = self.get_scalefactor(self.ring)

        if self.adaptive_scaling:
            sf_estimate = np.median(tr.data) / self.omegaE

        if self.mode == "seismic":

            # scaling
            if self.adaptive_scaling:
                # tr.data = (tr.data - np.median(tr.data)) / np.median(tr.data) * self.omegaE
                tr.data = tr.data / sf_estimate - self.omegaE
            else:
                # tr.data = (tr.data - self.nominal_sagnac) / self.nominal_sagnac * self.omegaE
                tr.data = tr.data / sf - self.omegaE

            # decimate to desired sampling rate
            tr = tr.detrend("demean")
            tr = tr.filter("lowpass", freq=500, corners=4, zerophase=True)
            tr = tr.resample(1000, no_filter=True)

            tr = tr.detrend("demean")
            tr = tr.filter("lowpass", freq=300, corners=4, zerophase=True)
            tr = tr.resample(600, no_filter=True)

            tr = tr.detrend("demean")
            tr = tr.filter("lowpass", freq=fnqst, corners=4, zerophase=True)
            tr = tr.resample(df, no_filter=True)

            # adjust time interval
            tr = tr.trim(self.tbeg, self.tend, nearest_sample=False)

            # remove last sample to avoid overlaps in stream
            tr.data = tr.data[:-1]

        elif self.mode == "geodetic":

            # adjust time interval
            tr = tr.trim(self.tbeg, self.tend, nearest_sample=False)
    
            # remove last sample to avoid overlaps in stream
            tr.data = tr.data[:-1]

            print(tr)

            # sampling time 
            dt = 1/df

            # total samples of trace
            NN = tr.stats.npts

            # sample increment
            dN = int(self.df0 * dt)

            # indies
            N1 = np.arange(0, NN, dN).astype(int)
            N2 = N1 + dN - 1

            # averaging
            avg = np.zeros(len(N1))
            for k, (n1, n2) in enumerate(zip(N1, N2)):
                avg[k] = (np.median(tr.data[n1:n2]))

            tr.data = avg
            tr.stats.starttime = tr.stats.starttime + dt/2
            tr.stats.sampling_rate = df

            print(tr)

        # add trace to stream
        self.fstream += tr

        # merge same traces
        self.fstream.merge()

        if output:
            return self.fstream

    def write_stream_to_sds(self, path_to_out_sds: str):
        """
        Write processed stream to SDS archive.

        Parameters
        ----------
        path_to_out_sds : str
            Path to output SDS archive directory

        Notes
        -----
        Creates necessary directory structure and writes data in MSEED format
        following SDS conventions.
        """
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
                print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")
            except:
                print(f" -> failed to write: {cc}")

    @staticmethod
    def get_scalefactor(ring: str = "Z") -> float:
        """
        Calculate scale factor for ring laser measurements.

        Parameters
        ----------
        ring : str, default="Z"
            Ring identifier (Z, U, V, W)

        Returns
        -------
        float
            Scale factor for converting optical measurements to rotation rate

        Notes
        -----
        Accounts for ring geometry, wavelength, and Earth rotation effects.
        Uses ROMY site coordinates and ring-specific parameters.
        """
        from numpy import pi, sqrt, arccos, deg2rad, arcsin, cos, sin, array, zeros

        # angle in horizontal plane
        h_rot = {"Z":0, "U":0, "V":60, "W":60}

        # angle from vertical
        v_rot = {"Z":0, "U":109.5, "V":70.5, "W":70.5}
        # v_rot = {"Z":-90, "U":19.5, "V":-19.5, "W":-19.5}

        # side length
        L = {"Z":11.2, "U":12, "V":12, "W":12}

        # wavelength
        lamda = 632.8e-9

        # Scale factor
        S = (sqrt(3)*L[ring])/(3*lamda)

        # ROMY latitude
        lat = deg2rad(48.162941)
        lon = deg2rad(11.275501)

        # nominal Earth rotation
        omegaE = 2*pi/86400 * array([0, 0, 1])

        # matrix 1
        D = array([[-sin(lat)*cos(lon), -sin(lon), cos(lat)*cos(lon)],
                [sin(lat)*sin(lon), cos(lon), cos(lat)*sin(lon)],
                [cos(lat), 0, sin(lat)]
                ])

        # tilt
        da = deg2rad(0)
        dz = deg2rad(0)

        # tilt matrix
        R = array([[1, -da, -dz], [da,  1, 0], [dz, 0, 1]])

        pv = deg2rad(v_rot[ring])
        ph = deg2rad(h_rot[ring])

        # normal vector of ring
        nx = array([[sin(pv)*cos(ph)], [sin(pv)*sin(ph)], [cos(pv)]])

        one = array([0, 0, 1])

        out = S * ( one @ ( D @ (R @ nx) ) )[0]

        return out

    @staticmethod
    def get_time_intervals(tbeg: Union[str, UTCDateTime], tend: Union[str, UTCDateTime],
                           interval_seconds: float, interval_overlap: float = 0,
                           output: bool = True):
        """
        Generate time intervals for data processing.

        Parameters
        ----------
        tbeg : str or UTCDateTime
            Start time
        tend : str or UTCDateTime
            End time
        interval_seconds : float
            Length of each interval in seconds
        interval_overlap : float, default=0
            Overlap between intervals in seconds
        output : bool, default=True
            Return intervals if True

        Returns
        -------
        list of tuple
            List of (start, end) time pairs for each interval
        """
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

    @staticmethod
    def save_to_pickle(obj: object, path: str, name: str):
        """
        Save object to pickle file.

        Parameters
        ----------
        obj : object
            Object to save
        path : str
            Directory path for output file
        name : str
            Base name for output file (without extension)

        Notes
        -----
        Creates a .pkl file and prints confirmation message.
        """
        import os
        import pickle

        ofile = open(path+name+".pkl", 'wb')
        pickle.dump(obj, ofile)

        if os.path.isfile(path+name+".pkl"):
            print(f"\n -> created:  {path}{name}.pkl")