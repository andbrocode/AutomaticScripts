class sagnacdemod:

    def __init__(self, config=None, output_sampling_rate=1, ddt=100, mode="seismic", nominal_sagnacf=None, loc="", ring="Z", adaptive_scaling=False):
        
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
        f_lower = self.nominal_sagnac - fband
        f_upper = self.nominal_sagnac + fband

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

    def get_stream(self, df, output=True):

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
                print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")
            except:
                print(f" -> failed to write: {cc}")

    @staticmethod
    def get_scalefactor(ring="Z"):

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

    @staticmethod
    def save_to_pickle(obj, path, name):

        import os
        import pickle

        ofile = open(path+name+".pkl", 'wb')
        pickle.dump(obj, ofile)

        if os.path.isfile(path+name+".pkl"):
            print(f"\n -> created:  {path}{name}.pkl")