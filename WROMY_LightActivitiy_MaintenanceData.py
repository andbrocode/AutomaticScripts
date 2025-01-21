#!/bin/python3

# Generate Data for Maintenance Activitiy based on WROMY LightSensors

# Use the light sensors of WROMY to identify maintenance activity

import os
import sys
import pandas as pd
import numpy as np

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


## Configurations

config = {}


if len(sys.argv) > 1:
    config['date'] = sys.argv[1]
    print(config['date'])

config['year'] = config['date'][:4]

config['threshold'] = 5 ## light threshold to classify as <light on> / <light off>

config['path_to_LX_data'] = archive_path+"romy_archive/"

config['path_to_outdata'] = archive_path+f"romy_archive/{config['year']}/BW/WROMY/"

config['path_to_outlog'] = archive_path+f"romy_autodata/{config['year']}/logfiles/"

config['path_to_figs'] = archive_path+f"romy_plots/{config['year']}/logs/"


def __read_LX_files(date, path_to_files, threshold=5):

    ## format date
    fdate = str(date)[:10].replace("-", "")

    year = fdate[:4]

    ## interate for all sensors of WROMY
    counter = 0
    for sensor in [1,4,5,6,7,8,9]:

        ## assemble file name
        filename = f'WS{sensor}_{fdate}.txt'

        ## modify data path
        datapath = f'{path_to_files}{year}/BW/WROMY/LX{sensor}.D/'

        ## read files
        if os.path.isfile(datapath+filename):
            df = pd.read_csv(datapath+filename, names=['Date', 'Time', 'State'])
            counter += 1
        else:
            #print(f" -> {datapath+filename} does not exists!")
            continue

        df['State'] = [1 if _val > threshold else 0 for _val in df['State']]

        ## rename column properly by sensor number
        df.rename(columns={"State":f"WS{sensor}"}, inplace=True)

        ## prepend zeros for times in column Time and convert to string
        df['Time'] = [str(_t).rjust(6, "0") for _t in df.Time]

        ## drop nan
        # df = df.dropna(subset=['Date'], inplace=True)

        ## convert Date to string
        df['Date'] = df['Date'].fillna(-1).astype('Int64').astype(str).replace('-1', np.nan)

        ## creat new datetime column
        df['datetime'] = pd.to_datetime(df['Date'] + 'T' + df['Time'], format="%Y%m%dT%H%M%S")

        ## set datetime column as index
        df.set_index("datetime", inplace=True)

        ## resample to one minute rows and use maximum of values (=conserving ones)
        df = df.resample('1min').max()

        ## drop columns no longer needed
        df.drop(columns=["Date", "Time"], inplace=True)

        ## merge dataframes after first one
        if counter == 1:
            df0 = df
        else:
            df0 = pd.merge(left=df0, right=df, how="left", left_on=["datetime"], right_on=["datetime"])

    df0.reset_index(inplace=True)

    return df0


def main():

    ## load dataframe
    df = __read_LX_files(config['date'], config['path_to_LX_data'], threshold=config['threshold'])

    ## add column with sum of all sensors
    df['sum_all'] = df.sum(axis=1, numeric_only=True).astype(int)

    ## write to daily files
    ## test if directory already exists
    if not os.path.isdir(config['path_to_outdata']+"LXX.D/"):
        os.mkdir(config['path_to_outdata']+"LXX.D/")

    ## format date
    fdate = config['date'].replace("-", "")

    ## write to daily pickle file
    df.to_pickle(config['path_to_outdata']+"LXX.D/"+f"{fdate}.pkl")

    ## add to overall maintenance log
    df = df[df.sum_all != 0]

    ## check if file already exists
    if not os.path.isfile(config['path_to_outlog']+"LXX_maintenance.log"):
        df.to_csv(config['path_to_outlog']+"LXX_maintenance.log", mode='w', index=False, header=False)
    else:
        df.to_csv(config['path_to_outlog']+"LXX_maintenance.log", mode='a', index=False, header=False)


if __name__ == "__main__":
    main()

## End of File
