import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from matplotlib.figure import Figure
from obspy import UTCDateTime, Stream
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta

class spectra:
    """
    A class for computing and analyzing spectral estimates of seismic data.
    
    This class provides methods for reading data, computing various types of spectra,
    and visualizing the results.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the spectra class.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with processing parameters
        """
        self.specta = {}
        self.mode = None
        self.verbose = True


        # If config is provided, store and validate it
        if config is not None:
            self.set_config(config)

    def set_config(self, config: dict) -> None:
        """
        Set and validate configuration parameters.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary that may contain:
            - 'tbeg': Start time (str or UTCDateTime)
            - 'tend': End time (str or UTCDateTime)
            - 'seeds': SEED identifier (list of str)
            - 'path_to_sds': Path to SDS archive (str)
            - 'path_to_data_out': Output path for data (str)
            - 'path_to_figures_out': Output path for figures (str)
            - 'tinterval': Time interval in seconds (int)
            - 'toverlap': Time overlap in seconds (int)
            - 'method': Spectral method ('welch'|'multitaper'|'fft')
            - 'verbose': Print detailed information (bool)
        """
        required_keys = [
            'tbeg', 
            'tend', 
            'seeds',
            'path_to_sds',
            'path_to_data_out',
            'path_to_figures_out',
            'tinterval',
            'toverlap',
            'method'
        ]
        
        # Check for required keys
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate paths
        paths = ['path_to_sds', 'path_to_data_out', 'path_to_figures_out']
        for path_key in paths:
            path = config[path_key]
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                    print(f"Created directory: {path}")
                except Exception as e:
                    raise ValueError(f"Could not create {path_key}: {path}\nError: {e}")
        
        # Validate method
        valid_methods = ['welch', 'multitaper', 'fft']
        if config['method'] not in valid_methods:
            raise ValueError(f"Invalid method: {config['method']}. Must be one of {valid_methods}")
        
        # Validate numerical parameters
        if config['tinterval'] <= 0:
            raise ValueError("tinterval must be positive")
        if config['toverlap'] < 0:
            raise ValueError("toverlap must be non-negative")
        if config['toverlap'] >= config['tinterval']:
            raise ValueError("toverlap must be less than tinterval")
        
        # Store configuration
        self.config = config
        
        # Set verbose mode
        self.verbose = config.get('verbose', False)
        
        if self.verbose:
            print("Configuration set successfully")

    @staticmethod
    def read_from_sds(path_to_archive: str, seed: str, 
                      tbeg: Union[str, UTCDateTime], tend: Union[str, UTCDateTime], 
                      data_format: str="MSEED", merge: bool=False) -> Stream:
        """
        Read waveform data from a SeisComP Data Structure (SDS) archive.
        
        Parameters:
        -----------
        path_to_archive : str
            Path to the SDS archive root directory
        seed : str
            SEED identifier in format "NET.STA.LOC.CHA"
        tbeg : str or UTCDateTime
            Start time of data to read
        tend : str or UTCDateTime
            End time of data to read
        data_format : str, optional
            Format of data files (default: "MSEED")
        merge : bool, optional
            Merge traces if True (default: False)
            
        Returns:
        --------
        obspy.Stream
            Stream containing the requested waveform data
        """

        import os
        from obspy.core import UTCDateTime, Stream
        from obspy.clients.filesystem.sds import Client
        from numpy import ma
        tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

        if not os.path.exists(path_to_archive):
            print(f" -> {path_to_archive} does not exist!")
            return Stream()

        ## separate seed id
        net, sta, loc, cha = seed.split('.')

        ## define SDS client
        client = Client(path_to_archive, sds_type='D', format=data_format)

        ## read waveforms
        try:
            st = client.get_waveforms(net, sta, loc, cha, tbeg-10, tend+10, merge=-1)
        except:
            print(f" -> failed to obtain waveforms!")
            st = Stream()

        # for tr in st:
        #     if isinstance(tr.data, ma.masked_array):
        #         print(f" -> {tr.id} has masked values! Filling masked values...")
        #         tr.data = np.nan_to_num(tr.data, 0)
        #     if np.isnan(tr.data).any():
        #         print(f" -> {tr.id} has NaN values! Splitting trace...")
        #         st.split()

        print(f" -> detrend stream...")
        st = st.detrend("linear")
        st = st.detrend("demean")

        if merge:
            print(f" -> merging stream...")
            # st = st.merge(fill_value="interpolate")
            st = st.merge(fill_value=0)

        st = st.trim(tbeg, tend, nearest_sample=False)

        return st

    def add_trace(self, trace):
        """
        Add a seismic trace to the spectra object.
        
        Parameters:
        -----------
        trace : obspy.Trace
            Seismic trace to analyze
            
        Notes:
        ------
        The trace is detrended (demean) before being stored.
        """

        # trace = trace.detrend("linear")
        # trace = trace.detrend("demean")

        self.tr = trace
        self.dt = trace.stats.delta
        self.tr_id = trace.get_id()

        self.tbeg = trace.stats.starttime
        self.tend = trace.stats.endtime

    def save_collection(self, path: str) -> None:
        """
        Save spectral collection and parameters to a daily pickle file.
        
        Parameters:
        -----------
        path : str, optional
            Root directory path where to save the pickle file
            If None, saves to current directory
            
        Notes:
        ------
        Creates subdirectories: path/NETWORK/STATION/CHANNEL/
        Creates file: YYYYMMDD_NET_STA_LOC_CHA_spectra.pkl
        """
        import os
        import pickle
        from datetime import datetime
        
        # Prepare output directory
        if path is None:
            path = os.getcwd()
        
        # Create subdirectory structure based on trace ID
        net = self.tr.stats.network
        sta = self.tr.stats.station
        cha = self.tr.stats.channel
        year = str(self.tr.stats.starttime.year)

        save_dir = os.path.join(path, year, net, sta, cha)
        os.makedirs(save_dir, exist_ok=True)
        
        # Create output dictionary
        out_dict = {
            'collection': self.collection,
            'parameters': {
                'method': self.method,
                'sampling_rate': 1/self.dt,
                't_interval': self.t_interval,
                't_overlap': self.t_overlap,
                'fmin': self.fmin,
                'fmax': self.fmax
            },
            'trace_stats': {
                'network': net,
                'station': sta,
                'channel': cha,
                'starttime': self.tr.stats.starttime,
                'endtime': self.tr.stats.endtime
            }
        }
        
        # Create filename
        date_str = self.tr.stats.starttime.strftime('%Y%m%d')
        fname = f"{date_str}_{self.tr.id.replace('.', '_')}_spectra.pkl"
        fpath = os.path.join(save_dir, fname)
        
        # Save to pickle file
        with open(fpath, 'wb') as f:
            pickle.dump(out_dict, f)
        
        print(f"Saved to: {fpath}")

    def get_time_intervals(self):
        """
        Prepare time intervals for spectral analysis.
        
        Uses the object's t_interval and t_overlap attributes to create a list
        of time window tuples. Results are stored in self.time_intervals.
        """

        from obspy import UTCDateTime

        times = []
        t1, t2 = self.tbeg, self.tbeg + self.t_interval

        while t2 <= self.tend+1:
            times.append((t1, t2))
            t1 = t1 + self.t_interval - self.t_overlap
            t2 = t2 + self.t_interval - self.t_overlap

        self.time_intervals = times

    def get_fft(self, arr: np.ndarray, dt: float, window: str="hann"):
        """
        Calculate Fast Fourier Transform of a time series.
        
        Parameters:
        -----------
        arr : numpy.ndarray, optional
            Input time series array (default: uses self.tr.data)
        dt : float, optional
            Sampling interval in seconds (default: uses self.dt)
        window : str, optional
            Window function to apply (default: "hann")
            
        Returns:
        --------
        tuple
            (frequencies, magnitude, phase)
            - frequencies: array of frequency points in Hz
            - magnitude: FFT magnitude spectrum
            - phase: phase spectrum in radians
        """

        from scipy.fft import fft, fftfreq, fftshift
        from scipy import signal
        from numpy import angle, imag, array

        if arr is None:
            arr = array(self.tr.data)

        if dt is None:
            dt = float(self.dt)

        # determine length of the input time series
        n = int(len(arr))

        # calculate spectrum (with or without window function applied to time series)
        if window:
            win = signal.get_window(window, n);
            spectrum = array(fft(arr * win))

        else:
            spectrum = array(fft(arr))

        # calculate frequency array
        frequencies = fftfreq(n, d=dt)

        # correct amplitudes of spectrum
        magnitude = abs(spectrum) * 2.0 / n

        # compute phase spectrum
        phase = angle(spectrum, deg=False)

        # return the positive frequencies
        return frequencies[0:n//2], magnitude[0:n//2], phase[0:n//2]

    def get_welch_psd(self, arr: np.ndarray, dt: float, twin_sec: float=60):
        """
        Compute power spectral density using Welch's method.
        
        Parameters:
        -----------
        arr : numpy.ndarray, optional
            Input time series array (default: uses self.tr.data)
        dt : float, optional
            Sampling interval in seconds (default: uses self.dt)
        twin_sec : float, optional
            Window length in seconds (default: 60)
            
        Returns:
        --------
        tuple
            (frequencies, psd)
            - frequencies: frequency array in Hz
            - psd: power spectral density estimate
        """

        from scipy.signal import welch
        from scipy.signal.windows import hann
        from numpy import array
        if arr is None:
            arr = array(self.tr.data)

        if dt is None:
            dt = float(self.dt)

        # Calculate window size in samples
        if twin_sec is None:
            nblock = len(arr)
        else:
            nblock = int(twin_sec / dt)
        
        # Ensure nblock is not larger than array length
        nblock = min(nblock, len(arr))
        
        # Calculate overlap in samples
        overlap = int(nblock * 0.5)
        
        # Ensure overlap is less than nblock
        if overlap >= nblock:
            overlap = nblock - 1
        
        # Apply Hann window
        win = "hann"
        
        # Calculate PSD
        ff, Pxx = welch(arr,
                        fs=1/dt,
                        window=win,
                        nperseg=nblock,
                        noverlap=overlap,
                        scaling="density",
                        return_onesided=True)
        
        return ff, Pxx

    def get_multitaper_psd(self, arr: np.ndarray, dt: float, n_win: int=5, time_bandwidth: float=4.0) -> tuple:
        """
        Compute power spectral density using multitaper method.
        
        Parameters:
        -----------
        arr : numpy.ndarray
            Input time series
        dt : float
            Sampling interval
        n_win : int, optional
            Number of windows (default: 5)
        time_bandwidth : float, optional
            Time-bandwidth product (default: 4.0)
            
        Returns:
        --------
        tuple
            Frequencies and PSD values
        """
        import multitaper as mt
        from numpy import array

        if arr is None:
            arr = array(self.tr.data)

        if dt is None:
            dt = float(self.dt)

        # Compute multitaper spectrum
        out_psd = mt.MTSpec(arr, nw=int(time_bandwidth), kspec=n_win, dt=dt, iadapt=2)
        
        # Get frequencies and PSD
        p = out_psd.rspec()
        
        # Reshape arrays
        if p is not None:
            f = array(p[0]).reshape(array(p[0]).size)
            psd = array(p[1]).reshape(array(p[1]).size)
        else:
            f = array([])
            psd = array([])
        
        return f, psd

    def get_collection(self, tinterval: int, toverlap: int, method: str="welch", twin_sec: int=3600):
        '''
        Get time intervals between starttime and endtime
        '''

        from gc import collect
        from numpy import nanmin, nanmax

        self.t_interval = tinterval
        self.t_overlap = toverlap

        self.get_time_intervals()

        self.method = method.lower()

        out = {}
        out['freq'] = []
        out['spec'] = []
        out['phas'] = []
        out['time'] = []
        out['time_label'] = []
        out['trace_zeros'] = []

        for _t1, _t2 in self.time_intervals:

            _tr = self.tr.copy().trim(_t1, _t2)

            # check for zero value sequences
            zeros = np.isclose(_tr.data, 0, atol=1e-30)
            # detect zero sequences
            zero_seq_lengths = np.diff(np.where(np.concatenate(([zeros[0]], 
                                                                 zeros[1:] != zeros[:-1], 
                                                                 [True])))[0])
            max_zero_seq = np.max(zero_seq_lengths) if len(zero_seq_lengths) > 0 else 0

            split = False
            if max_zero_seq > 30:
                # flag trace with zeros
                out['trace_zeros'].append(True)
                # splite trace for detrending
                split=True
                _tr.split()
            else:
                out['trace_zeros'].append(False)

            _tr = _tr.detrend("linear")
            _tr = _tr.detrend("demean")

            if split:
                _tr.merge()

            # compute FFT spectrum
            if self.method == "fft":
                f, s, p = self.get_fft(_tr.data, _tr.stats.delta, window="hann")

            # compute psd with Welch method
            elif self.method == "welch":
                if twin_sec is None:
                    f, s = self.get_welch_psd(arr=_tr.data, dt=_tr.stats.delta, twin_sec=tinterval)
                else:
                    f, s = self.get_welch_psd(arr=_tr.data, dt=_tr.stats.delta, twin_sec=twin_sec)

                p = f * 0  # dummy phases

            # compute psd with multitaper method
            elif self.method == "multitaper":
                f, s = self.get_multitaper_psd(arr=_tr.data, dt=_tr.stats.delta, n_win=10)
                p = f * 0  # dummy phases

            out['time_label'].append(f"{_tr.stats.starttime.time} - {_tr.stats.endtime.time}")
            out['freq'].append(f)
            out['spec'].append(s)
            out['phas'].append(p)
            out['time'].append(_tr.stats.starttime+0.5*(_tr.stats.endtime-_tr.stats.starttime))

            del _tr
            collect()

        # self.fmin = nanmin(f)
        self.fmin = 1/tinterval
        self.fmax = 0.5*1/self.dt
        self.collection = out

    def get_octave_bands(self, fmin, fmax, fraction_of_octave=1):
        '''
        Computing octave bands
        '''

        from acoustics.octave import Octave
        from numpy import array

        # avoid fmin = zero
        if fmin == 0:
            # print(f" -> set fmin to 1e-10 instead of 0")
            fmin = 1e-10

        f_lower, f_upper, f_centers = [], [], []

        _octaves = Octave(fraction=fraction_of_octave,
                          interval=None,
                          fmin=fmin,
                          fmax=fmax,
                          unique=False,
                          reference=1000.0
                         )

        f_centers = array(_octaves.center)
        f_lower = array(_octaves.lower)
        f_upper = array(_octaves.upper)

        return f_lower, f_upper, f_centers

    def get_fband_average(self, fraction_of_octave: int=1, average: str="mean"):
        """
        Compute averages for frequency octave bands.
        
        Parameters:
        -----------
        fraction_of_octave : int, optional
            Number of bands per octave (default: 1)
        average : str, optional
            Type of average to compute ("mean" or "median", default: "mean")
            
        Notes:
        ------
        Results are stored in self.collection dictionary with keys:
        - freq: center frequencies
        - spec: averaged spectra
        - time_label: time window labels
        - time: time points
        """

        import matplotlib.pyplot as plt
        from numpy import nanmean, nanmedian, array

        # get octave bands
        f_center, f_upper, f_lower = self.get_octave_bands(self.fmin,
                                                           self.fmax,
                                                           fraction_of_octave=fraction_of_octave,
                                                          )

        out = {}
        out['freq'] = []
        out['spec'] = []
        out['phas'] = []

        for freq, spec, phas in zip(self.collection['freq'], self.collection['spec'], self.collection['phas']):

            # get frequency indices
            fl_idx, fu_idx = [], []

            for _k, (fl, fu) in enumerate(zip(f_lower, f_upper)):
                if _k <= len(f_center):

                    for _i, _f in enumerate(freq):
                        if _f >= fl:
                            fl_idx.append(int(_i))
                            break

                    for _i, _f in enumerate(freq):
                        if _f >= fu:
                            fu_idx.append(int(_i))
                            break

            # compute mean per band
            psd_average, pha_average, fc, fu, fl = [], [], [], [], []
            for _n, (ifl, ifu) in enumerate(zip(fl_idx, fu_idx)):
                if ifl != ifu:
                    if average == "mean":
                        psd_average.append(nanmean(spec[ifl:ifu]))
                        pha_average.append(nanmean(phas[ifl:ifu]))
                    elif average == "median":
                        psd_average.append(nanmedian(spec[ifl:ifu]))
                        pha_average.append(nanmedian(phas[ifl:ifu]))

                    fc.append(f_center[_n])
                    fu.append(f_upper[_n])
                    fl.append(f_lower[_n])

            out['phas'].append(pha_average)
            out['freq'].append(array(fc))
            out['spec'].append(array(psd_average))
            out['time_label'] = self.collection['time_label']
            out['time'] = self.collection['time']
            out['trace_zeros'] = self.collection['trace_zeros']

        self.collection = out

    def plot_collection(self, mode=None, out=False):
        """
        Plot collection of spectral estimates.
        
        Parameters:
        -----------
        mode : str, optional
            Plot mode:
            - None: plot all individual spectra
            - "avg": plot only averaged spectra
            - "all": plot both individual and averaged spectra
        out : bool, optional
            Return figure object if True
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if out=True, else None
        """

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 7))

        if mode is None:
            for t, f, s in zip(self.collection['time_label'], self.collection['freq'], self.collection['spec']):
                plt.plot(f, s, label=t)
            
        if self.method in ['welch', 'multitaper']:
            plt.ylabel("Power Spectal Density")
        elif self.method in ['fft']:
            plt.ylabel("Amplitude Spectrum")

        plt.yscale("log")
        plt.xscale("log")

        plt.xlim(self.fmin, self.fmax)

        plt.xlabel("Frequency (Hz)")

        if self.mode == "fft":
            plt.ylabel("Spectrum")
        elif self.mode == "psd":
            plt.ylabel("PSD")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.grid(which="both", ls="--", color="grey", alpha=0.5, zorder=0)

        plt.show();
        if out:
            return fig

    def plot_spectrum(self, method: str="welch", window: str="hann", n_win: int=5, time_bandwidth: float=4.0, 
                     twin_sec: float=60, out: bool=False) -> Optional[Figure]:
        """
        Compute and plot a single spectrum using specified method
        
        Parameters:
        -----------
        method : str
            Spectral method to use: 'fft', 'welch', or 'multitaper'
        window : str
            Window function for FFT (default: 'hann')
        n_win : int
            Number of windows for multitaper method (default: 5)
        time_bandwidth : float
            Time-bandwidth product for multitaper method (default: 4.0)
        twin_sec : float
            Window length in seconds for Welch method (default: 60)
        out : bool
            Return figure if True
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if out=True, else None
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if method is None and self.mode is not None:
            method = self.mode
        else:
            print("method is not provided.")
        
        # Compute spectrum based on method
        if method == "fft":
            f, s, p = self.get_fft(self.tr.data, self.dt, window=window)
        elif method == "welch":
            f, s = self.get_welch_psd(self.tr.data, self.dt, twin_sec=twin_sec)
        elif method == "multitaper":
            f, s = self.get_multitaper_psd(self.tr.data, self.dt, n_win=n_win, time_bandwidth=time_bandwidth)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'fft', 'welch', or 'multitaper'")

        # Create figure
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Plot spectrum
        ax.plot(f, s, 'k-', lw=1, alpha=0.8)
        
        # Set scales and labels
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (Hz)')
        
        if method == "fft":
            ax.set_ylabel('Amplitude')
            title = f'FFT Spectrum (window: {window})'
        elif method == "welch":
            ax.set_ylabel('PSD')
            title = f'Welch PSD (window length: {twin_sec}s)'
        elif method == "multitaper":
            ax.set_ylabel('PSD')
            title = f'Multitaper PSD (n_win: {n_win}, time-bandwidth: {time_bandwidth})'
        
        # Add trace info to title
        title += f'\n{self.tr.id} ({self.tr.stats.starttime} - {self.tr.stats.endtime})'
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, which='both', ls='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        if out:
            return fig

    def plot_collection_with_trace(self, out=False):
        """
        Plot both time series and spectral estimates.
        
        Parameters:
        -----------
        mode : str, optional
            Plot mode:
            - None: plot all individual spectra
            - "avg": plot only averaged spectra
            - "all": plot both individual and averaged spectra
        out : bool, optional
            Return figure object if True
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if out=True, else None
            
        Notes:
        ------
        Creates a figure with two subplots:
        - Top: Time series data
        - Bottom: Spectral estimates
        """

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec

        tscale, tunit = 1/3600, "hour"

        Ncol, Nrow = 1, 5

        font = 12

        fig = plt.figure(figsize=(12, 7))

        gs1 = GridSpec(Nrow, Ncol, figure=fig, hspace=0.1, wspace=0.2)

        ax1 = fig.add_subplot(gs1[0, :])
        ax2 = fig.add_subplot(gs1[1:, :])

        for n, (_t1, _t2) in enumerate(self.time_intervals):
            _tr = self.tr.copy().trim(_t1, _t2)

            _times = np.arange(0, _t2-_t1+_tr.stats.delta, _tr.stats.delta) + n * (_t2-_t1)

            ax1.plot(_times*tscale, _tr)

        ax1.set_ylabel("Amplitude")

        for t, f, s in zip(self.collection['time_label'], self.collection['freq'], self.collection['spec']):
            ax2.plot(f, s, label=t)

        ax2.grid(which="both", ls="--", color="grey", alpha=0.5, zorder=0)
        ax2.set_yscale("log")
        ax2.set_xscale("log")

        ax2.set_xlim(self.fmin, self.fmax)

        ax2.set_xlabel("Frequency (Hz)")

        if self.mode == "fft":
            ax2.set_ylabel("Amplitude Spectrum")
        elif self.mode == "psd":
            ax2.set_ylabel("PSD")

        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show();
        if out:
            return fig

    def load_collection(self, path: str, seed_id: str, 
                       start_date: Union[str, datetime], 
                       end_date: Union[str, datetime, None] = None,
                       freq_range: Optional[Tuple[float, float]] = None,
                       n_freqs: int = 1000) -> bool:
        """
        Load spectral collection from files between start and end dates.
        
        Parameters
        ----------
        path : str
            Path to data directory
        seed_id : str
            SEED identifier (NET.STA.LOC.CHA)
        start_date : str or datetime
            Start date in format YYYY-MM-DD
        end_date : str or datetime, optional
            End date in format YYYY-MM-DD. If None, only load start_date.
        freq_range : tuple of float, optional
            (fmin, fmax) for generating frequencies if needed
        n_freqs : int, optional
            Number of frequency points if generating new frequency array
        """
        import os
        import pickle
        from datetime import datetime, timedelta
        from obspy import UTCDateTime
        
        try:
            # Convert dates to datetime if string
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            elif end_date is None:
                end_date = start_date
            
            # Create date range
            date_range = []
            current_date = start_date
            while current_date <= end_date:
                date_range.append(current_date)
                current_date += timedelta(days=1)
            
            # Create filepath base
            net, sta, loc, cha = seed_id.split('.')
            filepath_base = os.path.join(path, net, sta, cha)
            
            # Initialize collection
            combined_collection = {
                'time': [],
                'time_label': [],
                'spec': [],
                'freq': None,
                'trace_zeros': []
            }
            
            files_loaded = 0
            interval_hours = 1  # Default hourly intervals
            
            # Generate default frequency array if provided
            if freq_range is not None:
                default_freqs = np.logspace(np.log10(freq_range[0]), 
                                          np.log10(freq_range[1]), 
                                          n_freqs)
            
            # Load each day's data
            for date in date_range:
                filename = f"{date.strftime('%Y%m%d')}.pkl"
                full_path = os.path.join(filepath_base, filename)
                
                if os.path.exists(full_path):
                    with open(full_path, 'rb') as f:
                        daily_collection = pickle.load(f)
                    
                    # Check and fix frequencies if needed
                    daily_freqs = daily_collection.get('freq', daily_collection.get('frequencies', None))
                    if daily_freqs is None or np.all(np.isnan(daily_freqs)):
                        if freq_range is not None:
                            daily_freqs = default_freqs
                        else:
                            print(f"Warning: No valid frequencies in {filename} and no default provided")
                            continue
                    
                    # Set frequencies if not set
                    if combined_collection['freq'] is None:
                        combined_collection['freq'] = daily_freqs
                    
                    # Get and validate daily times
                    daily_times = daily_collection.get('time', [])
                    daily_specs = daily_collection.get('spec', daily_collection.get('spectra', []))
                    daily_trace_zeros = daily_collection.get('trace_zeros', [])

                    # Check if times are valid
                    if not daily_times or np.all(np.isnan(daily_times)):
                        # Generate expected times for this day
                        day_start = UTCDateTime(date)
                        daily_times = [day_start + hour * 3600 for hour in range(24)]
                        print(f"Warning: No valid times in {filename}, adding dummy times and specs")

                    if daily_times and daily_specs:
                        combined_collection['time'].extend(daily_times)
                        combined_collection['spec'].extend(daily_specs)
                        combined_collection['trace_zeros'].extend(daily_trace_zeros)
                        combined_collection['time_label'].extend([t.strftime('%Y-%m-%d %H:%M') 
                                                               if isinstance(t, (UTCDateTime, datetime))
                                                               else UTCDateTime(t).strftime('%Y-%m-%d %H:%M')
                                                               for t in daily_times])
                        files_loaded += 1
                else:
                    if self.verbose:
                        print(f"No file found for {date.strftime('%Y-%m-%d')}, adding dummy times")
                    
                    # Add dummy times and NaN spectra for the whole day
                    if combined_collection['freq'] is not None:
                        freq_len = len(combined_collection['freq'])
                        day_start = UTCDateTime(date)
                        
                        # Add hourly intervals
                        for hour in range(24):
                            time = day_start + hour * 3600
                            combined_collection['time'].append(time)
                            combined_collection['time_label'].append(time.strftime('%Y-%m-%d %H:%M'))
                            combined_collection['spec'].append(np.full(freq_len, np.nan))
                            combined_collection['trace_zeros'].append(False)
            if files_loaded == 0:
                if self.verbose:
                    print(f"No data files found between {start_date} and {end_date}")
                return False
            
            # Sort all arrays by time
            sorted_indices = np.argsort([t.datetime if isinstance(t, UTCDateTime) else t 
                                       for t in combined_collection['time']])
            for i in sorted_indices:
                if isinstance(combined_collection['time'][i], UTCDateTime):
                    combined_collection['time'][i] = combined_collection['time'][i].datetime
                else:
                    combined_collection['time'][i] = UTCDateTime(combined_collection['time'][i])
            combined_collection['time_label'] = [combined_collection['time_label'][i] for i in sorted_indices]
            combined_collection['spec'] = [combined_collection['spec'][i] for i in sorted_indices]
            combined_collection['trace_zeros'] = [combined_collection['trace_zeros'][i] for i in sorted_indices]

            # Store in object
            self.collection = combined_collection
            
            if self.verbose:
                print(f"Loaded {files_loaded} days of data")
                print(f"Total time points: {len(combined_collection['time'])}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading collection: {str(e)}")
            return False

    def classify_spectrum_quality(self, spectrum: np.ndarray, threshold: float = 1e-15, 
                         zero_seq_limit: int = 10, high_seq_limit: int = 50,
                         flat_seq_limit: int = 20, trace_zeros: bool = False) -> str:
        """
        Classify spectrum quality based on zero sequences, high values, and flat-line segments.
        
        Parameters
        ----------
        spectrum : np.ndarray
            Spectrum to analyze
        threshold : float
            Threshold for considering values as high
        zero_seq_limit : int
            Maximum allowed length of zero sequences
        high_seq_limit : int
            Maximum allowed length of high value sequences
        flat_seq_limit : int
            Maximum allowed length of flat-line segments
        
        Returns
        -------
        str
            Quality classification ('good' or 'bad')
        """
        
        # Convert to numpy array if needed
        spec = np.array(spectrum)
        
        # Find zero sequences
        zeros = np.isclose(spec, 0, atol=1e-30)
        zero_seq_lengths = np.diff(np.where(np.concatenate(([zeros[0]], 
                                                           zeros[1:] != zeros[:-1], 
                                                           [True])))[0])
        max_zero_seq = np.max(zero_seq_lengths) if len(zero_seq_lengths) > 0 else 0
        
        # Find high value sequences
        highs = spec > threshold
        high_seq_lengths = np.diff(np.where(np.concatenate(([highs[0]], 
                                                           highs[1:] != highs[:-1], 
                                                           [True])))[0])
        max_high_seq = np.max(high_seq_lengths) if len(high_seq_lengths) > 0 else 0
        
        # Check for flat-line segments
        has_flat_segments = False
        for i in range(0, len(spec) - flat_seq_limit):
            if np.all(spec[i:i+flat_seq_limit] == spec[i]):
                has_flat_segments = True
                break
        
        # Classify quality - simplified to good/bad
        if (max_zero_seq >= zero_seq_limit or 
            max_high_seq >= high_seq_limit or 
            trace_zeros or has_flat_segments):
            return 'bad'
        else:
            return 'good'

    def classify_collection_quality(self, threshold: float = 1e-15, 
                           zero_seq_limit: int = 10, high_seq_limit: int = 10,
                           flat_seq_limit: int = 20):
        """
        Classify quality for all spectra in the collection.
        
        Parameters
        ----------
        threshold : float
            Threshold for high value detection
        zero_seq_limit : int
            Maximum allowed length of zero sequences
        high_seq_limit : int
            Maximum allowed length of high value sequences
        flat_seq_limit : int
            Maximum allowed length of flat-line segments
        
        Updates the collection with quality flags for each spectrum.
        """
        if not hasattr(self, 'collection') or 'spec' not in self.collection:
            print("No collection data available")
            return
        
        if not 'trace_zeros' in self.collection.keys():
            print("No trace zeros available")
            self.collection['trace_zeros'] = [False] * len(self.collection['spec'])
        
        qualities = []
        for spectrum, trace_zeros in zip(self.collection['spec'], self.collection['trace_zeros']):
            quality = self.classify_spectrum_quality(
                spectrum, 
                threshold=threshold,
                zero_seq_limit=zero_seq_limit,
                high_seq_limit=high_seq_limit,
                flat_seq_limit=flat_seq_limit,
                trace_zeros=trace_zeros
            )
            qualities.append(quality)
        
        self.collection['quality'] = qualities
        
        # Print summary
        if self.verbose:
            n_good = sum(q == 'good' for q in qualities)
            print(f"Quality classification: {n_good} good, {len(qualities)-n_good} bad")

    def plot_spectrogram(self, cmap: Optional[str]=None, scale: str="log", 
                         fmin: float=0.01, fmax: float=10, vmin: Optional[float]=None, vmax: Optional[float]=None,
                         quality_filter: Optional[str]=None, out: bool=False) -> Optional[Figure]:
        """
        Plot spectrogram with optional quality filtering.
        
        Parameters
        ----------
        cmap : str
            Colormap name
        scale : str
            Scale type ('log' or 'linear')
        fmin, fmax : float
            Frequency range limits
        vmin, vmax : float, optional
            Color scale limits
        quality_filter : str, optional
            Only plot spectra with this quality ('good' or 'bad')
            Other spectra will be set to NaN but maintain time/frequency structure
        out : bool
            Return figure if True
        """
        from matplotlib.dates import HourLocator, DayLocator, DateFormatter, AutoDateLocator
        from matplotlib.colors import LogNorm
        import numpy as np
        import matplotlib.pyplot as plt

        if not hasattr(self, 'collection'):
            print("No collection available")
            return None
        
        if cmap is None:
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='white', alpha=1.0)

        # Get and validate data from collection
        specs_list = self.collection.get('spec', [])
        times = self.collection.get('time', [])
        freqs = self.collection.get('freq', [])
        
        if not (specs_list and times and freqs):
            raise ValueError("Missing required data in collection")
        
        # Handle case where freqs is a list of arrays
        if isinstance(freqs, list):
            freqs = freqs[0]
        
        # Convert specs_list to numpy array, ensuring consistent shapes
        freq_len = len(freqs)
        specs_array = np.zeros((len(specs_list), freq_len))
        for i, spec in enumerate(specs_list):
            if isinstance(spec, np.ndarray) and spec.size == freq_len:
                specs_array[i] = spec
            else:
                specs_array[i] = np.full(freq_len, np.nan)

        # Apply quality filter if specified
        n_filtered = 0
        if quality_filter is not None and 'quality' in self.collection:
            for i, quality in enumerate(self.collection['quality']):
                if quality != quality_filter:
                    specs_array[i] = np.full_like(specs_array[i], np.nan)
                    n_filtered += 1
                    if self.verbose:
                        print(f"Filtered spectrum at index {i} (quality != {quality_filter})")
        
        if n_filtered == len(specs_array):
            print("All spectra filtered out")
            return None
        
        # Filter frequencies based on fmin and fmax
        freq_array = np.array(freqs)
        freq_mask = (freq_array >= fmin) & (freq_array <= fmax)
        freq_array = freq_array[freq_mask]
        specs_array = specs_array[:, freq_mask]
        
        # Convert times consistently
        datetime_times = []
        for t in times:
            if isinstance(t, UTCDateTime):
                datetime_times.append(t.datetime)
            elif isinstance(t, (datetime, np.datetime64)):
                datetime_times.append(t)
            else:
                raise ValueError(f"Unexpected time type: {type(t)}")
        
        # Create time array
        time_array = np.array(datetime_times)
        
        # Create meshgrid for pcolormesh
        # Note: pcolormesh needs N+1 edges for N cells
        time_edges = np.zeros(len(time_array) + 1)
        time_edges[:-1] = np.array([date2num(t) for t in time_array])
        if len(time_array) > 1:
            time_edges[-1] = date2num(time_array[-1]) + (date2num(time_array[-1]) - date2num(time_array[-2]))
        
        # Create frequency edges (logarithmically spaced)
        freq_edges = np.zeros(len(freq_array) + 1)
        freq_edges[:-1] = freq_array
        freq_edges[-1] = freq_array[-1] * (freq_array[-1]/freq_array[-2])
        
        # Set up plot
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Configure colormap scaling
        if scale == 'log':
            if vmin is None:
                valid_data = specs_array[specs_array > 0]
                if len(valid_data) > 0:
                    vmin = float(np.nanpercentile(valid_data, 5))
                else:
                    vmin = 1e-10
            if vmax is None:
                vmax = float(np.nanpercentile(specs_array[~np.isnan(specs_array)], 95))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
        
        # Create spectrogram
        if norm is None:
            im = ax.pcolormesh(time_edges, freq_edges, specs_array.T,
                                cmap=cmap,
                                shading='flat',
                                rasterized=True)
        else:
            im = ax.pcolormesh(time_edges, freq_edges, specs_array.T,
                                cmap=cmap,
                                norm=norm,
                                shading='flat',
                                rasterized=True)
        
        # Configure axes
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (Hz)')
        
        # Determine time span and set appropriate locator
        time_span = max(datetime_times) - min(datetime_times)
        time_span_hours = time_span.total_seconds() / 3600
        
        if time_span_hours <= 24:
            ax.xaxis.set_major_locator(HourLocator(interval=2))
            ax.xaxis.set_minor_locator(HourLocator(interval=1))
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.set_xlabel('Time (HH:MM)')
        else:
            ax.xaxis.set_major_locator(AutoDateLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            ax.set_xlabel('')
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Make minor ticks visible but smaller
        ax.tick_params(which='minor', length=4, width=0.5)
        ax.tick_params(which='major', length=6, width=1)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        if hasattr(self, 'method'):
            if self.method in ['welch', 'multitaper']:
                cbar.set_label('Power Spectral Density')
            elif self.method in ['fft']:
                cbar.set_label('Amplitude Spectrum')
        
        # Set frequency limits
        if fmin is not None:
            ax.set_ylim(bottom=fmin)
        if fmax is not None:
            ax.set_ylim(top=fmax)
        
        # Add title
        title = f"{self.tr_id}"
        if hasattr(self, 'method'):
            title += f" | {self.method.upper()}"
        if len(datetime_times) > 24:
            title += f" | {min(datetime_times).strftime('%Y-%m-%d')} to {max(datetime_times).strftime('%Y-%m-%d')}"
        ax.set_title(title)
        
        # Add y limits
        if hasattr(self, 'fmin') and hasattr(self, 'fmax'):
            ax.set_ylim(self.fmin, self.fmax)
        if fmin is not None:
            ax.set_ylim(bottom=fmin)
        if fmax is not None:
            ax.set_ylim(top=fmax)
        
        plt.tight_layout()
        plt.show()
        
        if out:
            return fig

    @staticmethod
    def get_time_intervals_static(tbeg: Union[str, UTCDateTime], tend: Union[str, UTCDateTime], 
                                t_interval: float, t_overlap: float) -> List[Tuple[UTCDateTime, UTCDateTime]]:
        """
        Create list of time intervals between start and end time.
        
        Parameters:
        -----------
        tbeg : UTCDateTime
            Start time
        tend : UTCDateTime
            End time
        t_interval : float
            Length of each time window in seconds
        t_overlap : float
            Overlap between windows in seconds
            
        Returns:
        --------
        list
            List of tuples (start_time, end_time) for each window
        """
        from obspy import UTCDateTime

        tbeg = UTCDateTime(tbeg)
        tend = UTCDateTime(tend)

        times = []
        t1, t2 = tbeg, tbeg + t_interval

        while t2 <= tend:
            times.append((t1, t2))
            t1 = t1 + t_interval - t_overlap
            t2 = t2 + t_interval - t_overlap

        return times

    @staticmethod
    def load_collections(path: str, start_date: Union[str, datetime], end_date: Union[str, datetime],
                         seed: str, join: bool=False, fill_gaps: bool=False) -> Union[Dict[str, Dict], List[Dict]]:
        """
        Load multiple collections between dates.
        
        Parameters
        ----------
        path : str
            Path to data directory
        start_date : str or datetime
            Start date in format YYYY-MM-DD
        end_date : str or datetime
            End date in format YYYY-MM-DD
        seed : str
            SEED identifier (NET.STA.LOC.CHA)
        join : bool, optional
            Join collections if True
        fill_gaps : bool, optional
            Fill gaps with NaN values if True
        
        Returns
        -------
        Union[Dict[str, Dict], List[Dict]]
            Dictionary of collections keyed by seed ID if join=True,
            or list of collections if join=False
        """
        import pickle

        try:
            # Parse trace ID if provided
            if seed is not None:
                try:
                    network, station, location, channel = seed.split('.')
                except ValueError:
                    print(f"Invalid seed format: {seed}")
                    return {}
            else:
                location = ''  # Default empty location code
            
            # year = datetime.strptime(start_date.split()[0], '%Y')

            # # Construct data directory path
            # data_dir = os.path.join(str(path), str(year), str(network), str(station), str(channel))
            # if not os.path.exists(data_dir):
            #     print(f"Data directory not found: {data_dir}")
            #     return {}
            
            # Convert dates to datetime if needed
            if isinstance(start_date, str):
                try:
                    start_date = datetime.strptime(start_date.split()[0], '%Y-%m-%d')
                except ValueError:
                    print(f"Invalid start date format: {start_date}. Use 'YYYY-MM-DD'")
                    return {}
            
            if isinstance(end_date, str):
                try:
                    end_date = datetime.strptime(end_date.split()[0], '%Y-%m-%d')
                except ValueError:
                    print(f"Invalid end date format: {end_date}. Use 'YYYY-MM-DD'")
                    return {}
                
            # Get year as string
            year = str(start_date.year)
            
            # Construct data directory path with year
            data_dir = os.path.join(str(path), year, str(network), str(station), str(channel))
            if not os.path.exists(data_dir):
                print(f"Data directory not found: {data_dir}")
                return {}
            
            # First pass: find valid template
            template = None
            N_expected = None
            if fill_gaps:
                current_date = start_date
                while current_date <= end_date and template is None:
                    date_str = current_date.strftime('%Y%m%d')
                    fname = f"{date_str}_{network}_{station}_{location}_{channel}_spectra.pkl"
                    fpath = os.path.join(data_dir, fname)
                    
                    try:
                        with open(fpath, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Validate data structure and content
                        if ('collection' in data and 
                            'spec' in data['collection'] and 
                            'freq' in data['collection'] and
                            len(data['collection']['spec']) > 0):
                            
                            # Check if at least one spectrum has valid data
                            specs = np.array(data['collection']['spec'])
                            if not np.any(np.isnan(specs)):
                                template = data
                                N_expected = len(data['collection']['time'])
                                print(f"Found valid template in {fname}")
                                
                    except Exception as e:
                        print(f"Error reading {fname}: {str(e)}")
                    
                    current_date += timedelta(days=1)
                
                if template is None:
                    print("Warning: No valid template found. Gaps will not be filled.")
                    fill_gaps = False
            
            # Second pass: load all data
            collections = []
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y%m%d')
                fname = f"{date_str}_{network}_{station}_{location}_{channel}_spectra.pkl"
                fpath = os.path.join(data_dir, fname)
                
                try:
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Validate and fix time array if needed
                    if ('collection' in data and 
                        'time' in data['collection'] and 
                        'spec' in data['collection']):
                        
                        n_specs = len(data['collection']['spec'])
                        n_times = len(data['collection']['time'])
                        
                        if n_specs != n_times or N_expected is not None and n_specs != N_expected:
                            print(f"Warning: mismatch in {fname} ({n_times} times, {n_specs} spectra, {n_times} != {N_expected} expected)")
                            
                            # Generate expected times based on parameters
                            t_interval = data['parameters']['t_interval']
                            t_overlap = data['parameters'].get('t_overlap', 0)
                            day_start = UTCDateTime(current_date)
                            
                            intervals = spectra.get_time_intervals_static(
                                day_start, day_start + 86400,
                                t_interval, t_overlap
                            )
                            
                            # Replace time arrays with correct times
                            data['collection']['time'] = [t1 for t1, _ in intervals]
                            data['collection']['time_label'] = [f"{t1.time} - {t2.time}" for t1, t2 in intervals]
                            
                            # Pad or trim spec array if needed
                            freq_len = len(data['collection']['freq'])
                            if n_specs < len(intervals):
                                # Pad with NaN
                                pad_length = len(intervals) - n_specs
                                data['collection']['spec'].extend([np.full(freq_len, np.nan) for _ in range(pad_length)])
                            elif n_specs > len(intervals):
                                # Trim excess
                                data['collection']['spec'] = data['collection']['spec'][:len(intervals)]
                    
                    collections.append(data)
                    print(f"Loaded: {fname}")
                    
                except Exception as e:
                    print(f"No data for: {current_date.date()} ( {fname} )")
                    
                    # Create dummy data if fill_gaps is True and we have a valid template
                    if fill_gaps and template is not None:
                        dummy = {
                            'collection': {
                                'time': [],
                                'time_label': [],
                                'freq': template['collection']['freq'],
                                'spec': [],
                                'phas': [],
                                'quality': []  # Add quality array
                            },
                            'parameters': template['parameters'],
                            'trace_stats': {
                                'network': network,
                                'station': station,
                                'channel': channel,
                                'starttime': UTCDateTime(current_date),
                                'endtime': UTCDateTime(current_date) + 86400
                            }
                        }
                        
                        # Create time points for the day
                        t_interval = template['parameters']['t_interval']
                        t_overlap = template['parameters'].get('t_overlap', 0)
                        day_start = UTCDateTime(current_date)
                        
                        # Get time intervals for the day
                        intervals = spectra.get_time_intervals_static(
                            day_start, day_start + 86400,
                            t_interval, t_overlap
                        )
                        
                        # Fill with NaN data and mark as bad quality
                        freq_len = len(template['collection']['freq'])
                        for t1, t2 in intervals:
                            dummy['collection']['time'].append(t1)
                            dummy['collection']['time_label'].append(f"{t1.time} - {t2.time}")
                            dummy['collection']['spec'].append(np.full(freq_len, np.nan))
                            dummy['collection']['phas'].append(np.full(freq_len, np.nan))
                            dummy['collection']['quality'].append('bad')  # Mark dummy data as bad quality
                        collections.append(dummy)
                
                current_date += timedelta(days=1)
            
            if not collections:
                return {}  # Return empty dict instead of empty list
            
            if not join:
                # Convert list to dictionary
                return {seed: collection for collection in collections}
            
            # Join collections if requested
            try:
                # Get first collection for reference
                first_collection = collections[0]
                
                # Initialize joined collection
                joined = {
                    'collection': {
                        'time': [],
                        'time_label': [],
                        'freq': first_collection['collection']['freq'],
                        'spec': [],
                        'phas': [],
                        'quality': []
                    },
                    'parameters': first_collection['parameters'],
                    'trace_stats': first_collection['trace_stats'].copy()
                }
                
                # Update time range
                joined['trace_stats']['starttime'] = min(c['trace_stats']['starttime'] for c in collections)
                joined['trace_stats']['endtime'] = max(c['trace_stats']['endtime'] for c in collections)
                
                # Combine collections
                try:
                    for c in collections:
                        try:
                            joined['collection']['time'].extend(c['collection']['time'])
                            joined['collection']['time_label'].extend(c['collection']['time_label'])
                            joined['collection']['spec'].extend(c['collection']['spec'])
                            if 'phas' in c['collection']:
                                joined['collection']['phas'].extend(c['collection']['phas'])
                            joined['collection']['quality'].extend(c['collection']['quality'])
                        except Exception as e:
                            continue
                except Exception as e:
                    print(f"Error joining collections: {str(e)}")
                    return {seed: collection for collection in collections}
                
                # try:
                #     # Sort by time
                #     time_order = np.argsort([t.datetime for t in joined['collection']['time']])
                #     joined['collection']['time'] = [joined['collection']['time'][i] for i in time_order]
                #     joined['collection']['time_label'] = [joined['collection']['time_label'][i] for i in time_order]
                #     joined['collection']['spec'] = [joined['collection']['spec'][i] for i in time_order]
                #     if joined['collection']['phas']:
                #         joined['collection']['phas'] = [joined['collection']['phas'][i] for i in time_order]
                # except Exception as e:
                #     print(f"Error sorting collections: {str(e)}")
                #     return {seed: collection for collection in collections}
                
                # Return dictionary with seed ID as key
                return {seed: joined}
                
            except Exception as e:
                print(f"Error joining collections: {str(e)}")
                # Return unjoined collections as dictionary
                return {seed: collection for collection in collections}
                
        except Exception as e:
            print(f"Error loading collections: {str(e)}")
            return {}

    def plot_spectra_and_helicorder(self, fmin: Optional[float]=None, fmax: Optional[float]=None,
                             cmap: str='viridis', alpha: float=0.7, data_unit: Optional[str]=None,
                             quality_filter: Optional[str]=None,
                             out: bool=False, savefig: Optional[str]=None, show: bool=True) -> Optional[Figure]:
        """
        Plot spectra and corresponding time series segments side by side.
        
        Parameters:
        -----------
        fmin : float, optional
            Minimum frequency to plot
        fmax : float, optional
            Maximum frequency to plot
        cmap : str, optional
            Colormap for traces and spectra (default: 'viridis')
        alpha : float, optional
            Transparency for plots (default: 0.7)
        data_unit : str, optional
            Unit of data for axis labels
        quality_filter : str, optional
            Only plot spectra with this quality ('good' or 'bad')
            Other spectra will be set to NaN but maintain time/frequency structure
        out : bool, optional
            Return figure if True
        savefig : str, optional
            Path to save figure
        show : bool, optional
            Whether to display the figure
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if out=True, else None
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import Normalize, ListedColormap
        from matplotlib.cm import ScalarMappable, get_cmap
        from datetime import datetime, timedelta

        # Check if collection exists
        if not hasattr(self, 'collection'):
            raise AttributeError("No collection data available. Run get_collection() first.")
        
        # Create figure with two subplots side by side and space for colorbar
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(1, 3, width_ratios=[1, 0.02, 1], figure=fig)
        ax_spec = fig.add_subplot(gs[0])  # Spectra
        ax_heli = fig.add_subplot(gs[2])  # Helicorder
        ax_cbar = fig.add_subplot(gs[1])  # Colorbar
        
        # Get colormap and make it discrete
        n_intervals = len(self.collection['time'])
        color_norm = Normalize(vmin=-0.5, vmax=n_intervals-0.5)
        
        # Create discrete colormap
        base_cmap = get_cmap(cmap)
        colors = base_cmap(np.linspace(0, 1, n_intervals))
        discrete_cmap = ListedColormap(list(colors))

        sm = ScalarMappable(norm=color_norm, cmap=discrete_cmap)
        
        # First pass: detect gaps and apply quality filter
        gap_intervals = []  # Store which intervals have gaps
        quality_mask = []   # Store quality mask for each interval
        
        # Create quality mask if filter is specified
        if quality_filter is not None and 'quality' in self.collection:
            quality_mask = [q == quality_filter for q in self.collection['quality']]
        else:
            quality_mask = [True] * len(self.time_intervals)
        
        for i, ((t1, t2), is_good_quality) in enumerate(zip(self.time_intervals, quality_mask)):
            # Get data for this interval
            tr_slice = self.tr.slice(t1, t2)
            data = tr_slice.data
            
            # Improved gap detection
            has_gaps = False
            if len(data) > 0:  # Ensure we have data
                # Method 1: Check for zero sequences
                if np.any(data == 0.0):
                    # Find sequences of zeros
                    zero_mask = data == 0
                    zero_starts = np.where(np.diff(np.concatenate(([False], zero_mask))))[0]
                    zero_ends = np.where(np.diff(np.concatenate((zero_mask, [False]))))[0]
                    
                    # Make sure arrays have same length before subtraction
                    min_len = min(len(zero_starts), len(zero_ends))
                    if min_len > 0:
                        zero_lengths = zero_ends[:min_len] - zero_starts[:min_len]
                        if np.any(zero_lengths > 10):
                            has_gaps = True
                            if self.verbose:
                                print(f"Trace {i}: Found gap with {max(zero_lengths)} zeros")

                # Method 2: Check for flat-line segments
                if not has_gaps:
                    n_size = 20
                    for j in range(0, len(data) - n_size, n_size):
                        if np.all(data[j:j+n_size] == data[j]):
                            has_gaps = True
                            if self.verbose:
                                print(f"Trace {i}: Found flat-line segment")
                            break
            
            # Skip based on quality
            if not has_gaps and not is_good_quality:
                has_gaps = True
            
            if has_gaps:
                gap_intervals.append(i)
        
        # Plot spectra
        for i, (t, f, s) in enumerate(zip(self.collection['time_label'], 
                                        self.collection['freq'], 
                                        self.collection['spec'])):
            if i in gap_intervals:
                color = 'grey'
            else:
                color = discrete_cmap(i / (n_intervals-1))
            ax_spec.plot(f, s, color=color, alpha=alpha, lw=1)
        
        # Configure spectra axis
        ax_spec.set_xscale('log')
        ax_spec.set_yscale('log')
        ax_spec.set_xlabel('Frequency (Hz)')
        if self.method in ['welch', 'multitaper']:
            if data_unit is not None:
                hz = r'$^2$/Hz'
                ax_spec.set_ylabel(f'Power Spectral Density (({data_unit}){hz})')
            else:
                ax_spec.set_ylabel('Power Spectral Density')
        else:
            if data_unit is not None:
                hz = r'$\sqrt{Hz}$'
                ax_spec.set_ylabel(f'Amplitude Spectrum ({data_unit}/{hz})')
            else:
                ax_spec.set_ylabel('Amplitude Spectrum')

        ax_spec.grid(True, which='both', ls='--', alpha=0.3)
        
        # Set frequency limits
        if fmin is not None:
            ax_spec.set_xlim(left=fmin)
        if fmax is not None:
            ax_spec.set_xlim(right=fmax)
        
        # Plot helicorder
        max_amp = 0
        yticks = []
        yticklabels = []
        
        # Get the day's start and end time
        # start_times = [t[0] for t in self.time_intervals]
        # start_times_labels = [t.strftime('%H:%M') for t in start_times]

        # # Plot each hour's trace
        # for i, hour_start in enumerate(start_times):
        #     hour_end = UTCDateTime(hour_start + timedelta(hours=1))
            
        # Find if we have data for this hour
        data_found = False
        for j, (t1, t2) in enumerate(self.time_intervals):
            # Check if this interval overlaps with the current hour
            if t1 <= self.tend and t2 >= self.tbeg:
                # Get data for this interval
                tr_slice = self.tr.slice(max(t1, self.tbeg), min(t2, self.tend))

                if len(tr_slice.data) > 0:
                    # Calculate time in minutes for x-axis
                    start_minute = (tr_slice.stats.starttime - t1) / 60  # Convert seconds to minutes
                    times = start_minute + np.arange(len(tr_slice.data)) * self.dt / 60  # Convert to minutes
                    
                    # Normalize slice data
                    data = tr_slice.data
                    if np.any(np.abs(data) > 0):
                        norm_data = data / np.abs(data).max()
                        
                        if j in gap_intervals:
                            color = 'grey'
                        else:
                            color = discrete_cmap(j / (n_intervals-1))
                            
                        ax_heli.plot(times, norm_data - j, color=color, alpha=alpha, lw=1)
                        data_found = True
                        
                        max_amp = max(max_amp, np.abs(norm_data).max())
        
            # If no data found for this hour, plot dummy trace
            if not data_found:
                dummy_times = np.linspace(0, 60, 100)
                dummy_data = np.full_like(dummy_times, np.nan)
                ax_heli.plot(dummy_times, dummy_data - i, color='lightgray', alpha=0.3, lw=1, linestyle='--')
            
            # Store tick position and label
            yticks.append(-j)
            yticklabels.append(t1.strftime('%H:%M'))
        
        # Configure helicorder axis
        ax_heli.set_xlabel('Time (minutes)')
        ax_heli.set_xlim(0, 60)  # Set x-axis from 0 to 60 minutes
        ax_heli.set_yticks(yticks)
        ax_heli.set_yticklabels(yticklabels)
        ax_heli.grid(True, which='both', ls='--', alpha=0.3)
        
        # Add colorbar with discrete colors but no labels
        cbar = plt.colorbar(sm, cax=ax_cbar)
        cbar.set_ticks([])  # Remove ticks
        cbar.set_label('')  # Remove label
        
        # Adjust spacing between plots
        plt.subplots_adjust(wspace=0.2)  # Adjust this value to control spacing
        
        # Add title
        title = f"{self.tr_id}"
        if hasattr(self, 'tbeg'):
            title += f" | {self.tbeg.strftime('%Y-%m-%d')}"
        if hasattr(self, 'method'):
            title += f" | {self.method.upper()}"
        if quality_filter is not None:
            n_good = sum(quality_mask)
            title += f" | {quality_filter} quality ({n_good}/{len(quality_mask)} intervals)"
        if len(self.time_intervals) > 24:
            title += f" | {min(self.time_intervals).strftime('%Y-%m-%d')} to {max(self.time_intervals).strftime('%Y-%m-%d')}"
        plt.suptitle(title)
                
        # Save figure if path is provided
        if savefig is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(savefig), exist_ok=True)
            fig.savefig(savefig, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Figure saved to: {savefig}")
        
        if show:
            plt.tight_layout()
            plt.show()
            
        if out:
            return fig
        else:
            plt.close(fig)
            return None

    @staticmethod
    def remove_response(st: Stream, inv_file: str, output: str = "ACC", 
                       pre_filt: Optional[Tuple[float, float, float, float]] = None,
                       water_level: float = 60.0) -> Stream:
        """
        Remove instrument response or sensitivity from a stream using inventory file.
        
        Parameters
        ----------
        st : obspy.Stream
            Input stream object
        inv_file : str
            Path to inventory file (StationXML, RESP, etc.)
        output : str, optional
            Output units. One of:
            - "DISP" for displacement
            - "VEL" for velocity
            - "ACC" for acceleration
            Default is "VEL"
        pre_filt : tuple, optional
            Pre-filter frequencies (f1, f2, f3, f4) in Hz.
            Specify corner frequencies of cosine taper which is applied
            before instrument correction.
        water_level : float, optional
            Water level in dB for deconvolution. Default is 60.0.
        
        Returns
        -------
        obspy.Stream
            Stream with response removed
        
        Notes
        -----
        If inventory contains only sensitivity, a simple sensitivity correction
        will be applied. Otherwise, full response removal will be performed.
        """
        from obspy import read_inventory
        
        # Create copy of stream to avoid modifying original
        st_out = st.copy()
        
        net, sta, loc, cha = st_out[0].get_id().split(".")

        try:
            # Read inventory file
            inv = read_inventory(inv_file)
            
            # Try full response removal first
            try:
                if "J" in cha:
                    print("Removing sensitivity for J-channel")
                    st_out.remove_sensitivity(inventory=inv)
                else:
                    st_out.remove_response(
                        inventory=inv,
                        output=output,
                        pre_filt=pre_filt,
                        water_level=water_level
                    )
                print("Full response removal applied successfully")
                
            except Exception as e:
                # If full response fails, try sensitivity correction
                print(f"Full response removal failed: {str(e)}")
                print("Attempting sensitivity correction...")
                
                try:
                    st_out.remove_sensitivity(inventory=inv)
                    print("Sensitivity correction applied successfully")
                    
                except Exception as e:
                    print(f"Sensitivity correction failed: {str(e)}")
                    raise
                
        except Exception as e:
            print(f"Error reading inventory file: {str(e)}")
            raise
        
        return st_out

    def get_collection_statistics(self, percentiles=[5, 25, 50, 75, 95], 
                                fmin=None, fmax=None, quality_filter=None):
        """
        Calculate statistics (median, mean, percentiles) for the spectral collection.
        
        Parameters:
        -----------
        percentiles : list, optional
            List of percentiles to calculate (default: [5, 25, 50, 75, 95])
        fmin : float, optional
            Minimum frequency to include in statistics
        fmax : float, optional
            Maximum frequency to include in statistics
        quality_filter : str or None, optional
            Filter spectra by quality ('good', 'fair', 'poor', or None for all)
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'frequencies': frequency array
            - 'mean': mean spectrum
            - 'median': median spectrum (same as 50th percentile)
            - 'percentiles': dictionary of percentile spectra
            - 'count': number of spectra used in calculations
        """
        import numpy as np
        
        if not hasattr(self, 'collection') or not self.collection:
            print("No collection available")
            return None
        
        # Get frequencies and initialize arrays
        frequencies = self.collection['freq']
        
        # Apply frequency limits if specified
        if fmin is not None or fmax is not None:
            fmin = fmin if fmin is not None else frequencies[0]
            fmax = fmax if fmax is not None else frequencies[-1]
            freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
            frequencies = frequencies[freq_mask]
        else:
            freq_mask = slice(None)  # Use all frequencies
        
        # Get timestamps and initialize data array
        timestamps = sorted(self.collection['time'])
        
        # Filter by quality if specified
        data = []
        for spec, quality in zip(self.collection['spec'], self.collection['quality']):
            if quality == 'good':
                data.append(spec)
        data = np.array(data)

        # Calculate statistics
        mean_spectrum = np.mean(data, axis=0)
        median_spectrum = np.median(data, axis=0)
        percentile_spectra = {p: np.percentile(data, p, axis=0) for p in percentiles}
    
        # Create result dictionary
        result = {
            'frequencies': frequencies,
            'mean': mean_spectrum,
            'median': median_spectrum,
            'percentiles': percentile_spectra,
            'count': len(timestamps)
        }
        
        return result

    def plot_collection_statistics(self, stats=None, percentiles=[5, 25, 75, 95], 
                                fmin=None, fmax=None, quality_filter=None,
                                figsize=(10, 6), title=None, out=False,
                                show_mean=True, show_median=True, 
                                show_percentiles=True, grid=True,
                                xlim=None, ylim=None, xscale='log', yscale='log',
                                color_mean='red', color_median='blue', 
                                color_percentiles='skyblue', alpha_percentiles=0.3,
                                linewidth_mean=2, linewidth_median=2):
        """
        Plot statistics (median, mean, percentiles) for the spectral collection.
        
        Parameters:
        -----------
        stats : dict, optional
            Statistics dictionary from get_collection_statistics()
            If None, will be calculated using the other parameters
        percentiles : list, optional
            List of percentiles to plot (default: [5, 25, 75, 95])
        fmin : float, optional
            Minimum frequency to include
        fmax : float, optional
            Maximum frequency to include
        quality_filter : str or None, optional
            Filter spectra by quality ('good', 'fair', 'poor', or None for all)
        figsize : tuple, optional
            Figure size (width, height) in inches
        title : str, optional
            Figure title
        out : bool, optional
            If True, return the figure object instead of displaying it
        show_mean : bool, optional
            Whether to show the mean spectrum
        show_median : bool, optional
            Whether to show the median spectrum
        show_percentiles : bool, optional
            Whether to show percentile ranges
        grid : bool, optional
            Whether to show grid lines
        xlim : tuple, optional
            X-axis limits (min, max)
        ylim : tuple, optional
            Y-axis limits (min, max)
        xscale : str, optional
            X-axis scale ('log' or 'linear')
        yscale : str, optional
            Y-axis scale ('log' or 'linear')
        color_mean : str, optional
            Color for mean spectrum line
        color_median : str, optional
            Color for median spectrum line
        color_percentiles : str, optional
            Color for percentile ranges
        alpha_percentiles : float, optional
            Alpha (transparency) for percentile ranges
        linewidth_mean : float, optional
            Line width for mean spectrum
        linewidth_median : float, optional
            Line width for median spectrum
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if out=True, otherwise None
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate statistics if not provided
        if stats is None:
            stats = self.get_collection_statistics(
                percentiles=percentiles + [50] if 50 not in percentiles else percentiles,
                fmin=fmin,
                fmax=fmax,
                quality_filter=quality_filter
            )
        
        if stats is None:
            return None if out else plt.figure()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data
        frequencies = stats['frequencies'][0]
        mean_spectrum = stats['mean']
        median_spectrum = stats['median']
        percentile_spectra = stats['percentiles']
        
        # Plot percentiles as filled areas
        if show_percentiles:
            # Sort percentiles for proper filling
            sorted_percentiles = sorted(percentiles)
            
            # Plot pairs of percentiles as filled areas
            for i in range(len(sorted_percentiles) // 2):
                lower = sorted_percentiles[i]
                upper = sorted_percentiles[-(i+1)]
                ax.fill_between(
                    frequencies,
                    percentile_spectra[lower],
                    percentile_spectra[upper],
                    color=color_percentiles,
                    alpha=alpha_percentiles,
                    label=f"{lower}th-{upper}th percentile"
                )
        
        # Plot median
        if show_median:
            ax.plot(frequencies, median_spectrum, color=color_median, 
                linewidth=linewidth_median, label="Median")
        
        # Plot mean
        if show_mean:
            ax.plot(frequencies, mean_spectrum, color=color_mean, 
                linewidth=linewidth_mean, label="Mean")
        
        # Set scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        
        # Set limits if provided
        if xlim is not None:
            ax.set_xlim(xlim)
        elif fmin is not None and fmax is not None:
            ax.set_xlim(fmin, fmax)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Add grid
        if grid:
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        
        if title is None:
            title = f"Spectral Statistics for {self.tr_id}"
            if quality_filter:
                title += f" (Quality: {quality_filter})"
        
        ax.set_title(title)
        
        # Add count information
        ax.text(0.02, 0.02, f"n = {stats['count']} spectra", 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7))
        
        # Add legend
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        
        if out:
            return fig
        else:
            plt.show()
            return None

    def get_spectrum_at_time(self, timestamp, interpolate=True):
        """
        Get the spectrum at a specific timestamp.
        
        Parameters:
        -----------
        timestamp : str or UTCDateTime
            Timestamp to retrieve spectrum for
        interpolate : bool, optional
            If True and exact timestamp not found, interpolate between nearest timestamps
            
        Returns:
        --------
        tuple
            (frequencies, psd) arrays or None if not found
        """
        from obspy import UTCDateTime
        import numpy as np
        
        if not hasattr(self, 'collection') or not self.collection:
            print("No collection available")
            return None
        
        # Convert timestamp to string format used in collection
        if isinstance(timestamp, UTCDateTime):
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
        else:
            timestamp_str = str(timestamp)
        
        # Check if exact timestamp exists
        if timestamp_str in self.collection['spectra']:
            return (
                self.collection['frequencies'],
                self.collection['spectra'][timestamp_str]['psd']
            )
        
        # If interpolation not requested, return None
        if not interpolate:
            print(f"Timestamp {timestamp_str} not found in collection")
            return None
        
        # Convert all timestamps to UTCDateTime for comparison
        timestamps = sorted(self.collection['spectra'].keys())
        utc_timestamps = [UTCDateTime(ts) for ts in timestamps]
        target_utc = UTCDateTime(timestamp_str)
        
        # Find nearest timestamps
        time_diffs = [abs(ts - target_utc) for ts in utc_timestamps]
        if not time_diffs:
            print("No timestamps available for interpolation")
            return None
        
        # Find two closest timestamps
        sorted_indices = np.argsort(time_diffs)
        if len(sorted_indices) == 1:
            # Only one timestamp available, return it
            nearest_idx = sorted_indices[0]
            return (
                self.collection['frequencies'],
                self.collection['spectra'][timestamps[nearest_idx]]['psd']
            )
        
        # Get two closest timestamps
        idx1, idx2 = sorted_indices[0], sorted_indices[1]
        ts1, ts2 = utc_timestamps[idx1], utc_timestamps[idx2]
        
        # Ensure ts1 < ts2
        if ts1 > ts2:
            ts1, ts2 = ts2, ts1
            idx1, idx2 = idx2, idx1
        
        # Check if target is between the two timestamps
        if not (ts1 <= target_utc <= ts2):
            # Target is outside range, use nearest
            nearest_idx = idx1 if time_diffs[idx1] < time_diffs[idx2] else idx2
            return (
                self.collection['frequencies'],
                self.collection['spectra'][timestamps[nearest_idx]]['psd']
            )
        
        # Interpolate between the two spectra
        weight2 = (target_utc - ts1) / (ts2 - ts1)
        weight1 = 1 - weight2
        
        psd1 = self.collection['spectra'][timestamps[idx1]]['psd']
        psd2 = self.collection['spectra'][timestamps[idx2]]['psd']
        
        interpolated_psd = weight1 * psd1 + weight2 * psd2
        
        return (self.collection['frequencies'], interpolated_psd)
