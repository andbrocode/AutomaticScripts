"""

Run backscatter quantity computation and correction automatically

"""

import os
import sys
import numpy as np
# import multiprocessing as mp
# import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime, date
from pandas import DataFrame, read_pickle, date_range, concat, read_csv
from obspy import UTCDateTime, read
from scipy.signal import hilbert

from sagnacdemod import sagnacdemod

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

# extract and assign arguments
config['tbeg'] = UTCDateTime(sys.argv[1])
config['tend'] = config['tbeg'] + 86400

# config['tbeg'] = UTCDateTime("2024-11-04 00:00")
# config['tend'] = UTCDateTime("2024-11-05 00:00")

# project name
# config['project'] = ""

# select if progress bar is shown
config['show_progress'] = True

# select if info is printed
config['verbose'] = False

# store configurations as file
config['store_config'] = False

# specify path to store config file
config['path_to_config'] = "./"

# extract ring
config['ring'] = sys.argv[2]
# config['ring'] = "Z"

# set location code (to discrimiate datasets)
config['loc'] = "60"

# select frequency estimation mode
config['mode'] = "hilbert" # "hilbert" | "sine"

# specify seed codes
# config['seeds'] = [f"BW.DROMY..FJ{config['ring']}", "BW.DROMY..F1V", "BW.DROMY..F2V"]
config['seed'] = f"BW.DROMY..FJ{config['ring']}"

# specify interval of data to load (in seconds)
config['time_interval'] = 3600

# specify output sampling rate
config['output_sps'] = 20

# set adaptive scaling
config['adaptive_scaling'] = False

# set if amplitdes are corrected with envelope
config['correct_amplitudes'] = False

# set prewhitening factor (to avoid division by zero)
config['prewhitening'] = 0.001

# interval buffer (before and after) in seconds
config['ddt'] = 1800

# frequency band (minus and plus)
config['fband'] = 10 # 10

# specify cm filter value for backscatter correction
# config['cm_value'] = 1.033

# define nominal sagnac frequency of rings
config['ring_sagnac'] = {"U":303.05, "V":447.5, "W":447.5, "Z":553.5}
config['nominal_sagnac'] = config['ring_sagnac'][config['ring']]

# specify path to Sagnac data
#config['path_to_out_data'] = data_path+"sagnac_frequency/data/demod/"
config['path_to_out_data'] = archive_path+"temp_archive/"

# specify path to output figures
config['path_to_figs'] = data_path+"sagnac_frequency/figures/"

# specify path to sds data archive
config['path_to_sds'] = archive_path+"romy_archive/"

# specify amount of cpu kernels to use for parallel processing
# config['n_cpu'] = 3

# set seismic mode for high-frequency data demodulation
# or geodetic mode for averaging over the selected time period (e.g. 60s averages for sps = 1/60)
config['mode'] = "seismic" # seismic | geodetic



# _____________________________________________________________________________________

def main(config):


    sagnac = sagnacdemod(output_sampling_rate=20,
                         ddt=config.get('ddt'),
                         nominal_sagnacf=config.get('nominal_sagnac'),
                         loc=config.get('loc'),
                         ring=config.get('ring'),
                         adaptive_scaling=config.get('adaptive_scaling'),
                         mode=config.get('mode')
                         )

    if config['store_config']:
        sagnac.save_to_pickle(config, config['path_to_config'], f"{config['project']}_config.pkl")

    # get time intervals for data processing (memory limited)
    intervals = sagnac.get_time_intervals(config.get('tbeg'),config.get('tend'), config.get('time_interval'))

    # loop over time intervals
    if config['show_progress']:

        for t1, t2 in tqdm(intervals):

            sagnac.load_sagnac_data(config['seed'],
                                    t1,
                                    t2,
                                    config.get('path_to_sds'),
                                    verbose=config.get('verbose'),
                                    )


            sagnac.hilbert_estimator(fband=config.get('fband'),
                                    acorrect=config.get('correct_amplitudes'),
                                    prewhiten=config.get('prewhitening'),
                                    )

            sagnac.get_stream(df=config.get('output_sps'))

    else:

        for t1, t2 in intervals:

            sagnac.load_sagnac_data(config['seed'],
                                    t1,
                                    t2,
                                    config.get('path_to_sds'),
                                    verbose=config.get('verbose'),
                                    )


            sagnac.hilbert_estimator(fband=config.get('fband'),
                                    acorrect=config.get('correct_amplitudes'),
                                    prewhiten=config.get('prewhitening'),
                                    )

            sagnac.get_stream(df=config.get('output_sps'))

    print(sagnac.fstream)

    sagnac.write_stream_to_sds(config.get('path_to_out_data'))

    #sagnac.fstream.plot()


# ________ MAIN  ________
if __name__ == "__main__":

    main(config)

# End of File
