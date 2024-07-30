#!/usr/bin/env python
# coding: utf-8

# # ROMY Status File

# In[1]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy as obs
import matplotlib.colors

from pandas import DataFrame
from andbro__save_to_pickle import __save_to_pickle


# In[2]:

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


# In[3]:

from functions.load_beat import __load_beat
from functions.load_mlti import __load_mlti
from functions.get_mlti_intervals import __get_mlti_intervals

# ## Configurations

# In[4]:


config = {}


config['tbeg'] = obs.UTCDateTime(sys.argv[2])
config['tend'] = obs.UTCDateTime(sys.argv[2])+86400

# config['tbeg'] = obs.UTCDateTime("2023-09-21")
# config['tend'] = obs.UTCDateTime("2023-09-22")

config['ring'] = sys.argv[1]
# config['ring'] = "U"

config['path_to_autodata'] = archive_path+f"romy_autodata/"

config['path_to_figures'] = archive_path+f"romy_plots/{config['tbeg'].year}/R{config['ring']}/status/"

config['path_to_output'] = archive_path+f"temp_archive/{config['tbeg'].year}/BW/R{config['ring']}/"

config['fsagnac_rings'] = {"U":303, "V":447.5, "W":447.5, "Z":553.5}
config['fsagnac_nominal'] = config['fsagnac_rings'][config['ring']]

config['DC_threshold'] = 0.1

config['AC_threshold'] = 0.15

config['delta_fsagnac'] = 1.0


# ### Load MLTI Logs

# In[5]:

try:
    mlti_log = __load_mlti(config['tbeg'], config['tend'], config['ring'], archive_path)
except:
    print(f"no MLTI log: {config['tbeg']}")
    quit()

try:
    mlti_t1, mlti_t2 = __get_mlti_intervals(mlti_log.time_utc, time_delta=100)
except:
    print("mlti intervals failed!")
    mlti_t1, mlti_t2 = np.array([]), np.array([])


# ### Load Beat Data

# In[6]:

try:
    beat = __load_beat(config['tbeg'], config['tend'], config['ring'], config['path_to_autodata'])
except:
    print(f" -> failed to load data: {config['tbeg']}")
    quit()

if len(beat) == 0:
    print(f" -> no beat file: {config['tbeg']}")
    quit()

# ### Define Variables

# In[7]:


status = DataFrame()

status['times_utc'] = beat.times_utc
status['times_utc_sec'] = beat.times_utc_sec

N = status.shape[0]

quality = np.ones(N)
fsagnac = np.ones(N)
mlti = np.ones(N)
dc_threshold = np.ones(N)
ac_threshold = np.ones(N)


# ## Determine Status

# In[8]:


idx_mlti = 0

for idx in range(beat.shape[0]):

    _time = obs.UTCDateTime(status.times_utc.iloc[idx])

    ## check if time conincides with MLTI
    # print(_time, mlti_t1[idx_mlti], mlti_t2[idx_mlti])
    if len(mlti_t1) > 0 and len(mlti_t2) > 0:
        if _time >= mlti_t1[idx_mlti] and _time <= mlti_t2[idx_mlti]:
            quality[idx] = 0
            mlti[idx] = 0

        ## update mlti interval
        if _time > mlti_t2[idx_mlti] and idx_mlti < len(mlti_t1)-1:
            idx_mlti += 1

    if beat.fj.iloc[idx] < config['fsagnac_nominal'] - config['delta_fsagnac'] or beat.fj.iloc[idx] > config['fsagnac_nominal'] + config['delta_fsagnac']:
        quality[idx] = 0
        fsagnac[idx] = 0

    if beat.dc_z.iloc[idx] < config['DC_threshold']:
        quality[idx] = 0
        dc_threshold[idx] = 0

    if beat.ac_z.iloc[idx] < config['AC_threshold']:
        quality[idx] = 0
        ac_threshold[idx] = 0


status['quality'] = quality
status['fsagnac'] = fsagnac
status['mlti'] = mlti
status['ac_threshold'] = ac_threshold
status['dc_threshold'] = dc_threshold


## store output to file
# print(f"-> store: {config['path_to_output']}R{config['ring']}_{config['tbeg'].date}_status.pkl")
__save_to_pickle(status, config['path_to_output'],f"R{config['ring']}_{config['tbeg'].date}_status")


# ### Plotting

# In[9]:


arr = np.ones((3, status['quality'].size))

arr[0] *= status['quality']
arr[1] *= status['fsagnac']
arr[2] *= status['mlti']


# In[10]:

try:
    names = ["quality", "fsagnac", "mlti", "ac_threshold", "dc_threshold"]
    bars = np.ones(len(names))-0.5

    arr = np.ones((len(names), status['quality'].size))

    for _n, name in enumerate(names):
        arr[_n] *= status[name]


    cmap = matplotlib.colors.ListedColormap(['darkred', 'green'])

    fig = plt.figure(figsize=(15, 4))

    c = plt.pcolormesh(np.arange(0, arr.shape[1]), names, arr, cmap=cmap, rasterized=True, alpha=0.8)

    for _k, bar in enumerate(bars):
        plt.axhline(bar+_k, color="k", alpha=0.5)

    plt.xlabel("Time (min)")


    plt.title(f"Quality Status of R{config['ring']} on {config['tbeg'].date}")

    # plt.show();

    print(f" -> stored: {config['path_to_figures']}R{config['ring']}_{config['tbeg'].date}_status.png")
    fig.savefig(config['path_to_figures']+f"R{config['ring']}_{config['tbeg'].date}_status.png", format="png", dpi=100, bbox_inches='tight')

except:
    print(" -> failed to plot: {config['tbeg']")



## End of File
