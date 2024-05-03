#!/bin/python3

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calplot
import matplotlib.colors

import warnings
warnings.filterwarnings('ignore')

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'
elif os.uname().nodename == 'lin-ffb-01':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/bay200/'


year1 = int(input("Enter year1 [2022]: ")) or 2022
year2 = int(input("Enter year2 [2024]: ")) or 2024


years = range(year1, year2+1)

ring = input("Enter ring ( [RZ], RU, RV, RW ): ") or "RZ"

print(f" 1  status:  /temp_archive/<year>/BW/{ring}/")
print(f" 2  sagnac:  /romy_autodata/<year>/{ring}/")

data = input("Select data files: ")


# create dates for all years
days_of_year = pd.date_range(f"{years[0]}-01-01",  f"{years[-1]}-12-31", freq='D')

# create empty file series
file_exists = pd.Series(-1*np.ones(len(days_of_year)), index=days_of_year)

# loop over all years and dates
for year in years:

    # select path
    if data == "1" or data == "status":
        path = archive_path+f"temp_archive/{year}/BW/{ring}/"
    elif data == "2" or data == "sagnac":
        path = archive_path+f"romy_autodata/{year}/{ring}/"

    # modify path
    if path[-1] != "/":
        path = path+"/"

 
    for doy in days_of_year:

        doy = str(doy).split(" ")[0]

        _doy = doy

        # if os.path.isfile(path+f"FJU_{_doy}.pkl"):
        if len(glob.glob(path+f'*{_doy}*')) == 1:
            file_exists.loc[doy] = 1
        
        _doy = doy.replace("-", "")

        if len(glob.glob(path+f'*{_doy}*')) == 1:
            file_exists.loc[doy] = 1


cmap = matplotlib.colors.ListedColormap(['darkred', 'green'])

calplot.calplot(file_exists, cmap=cmap, edgecolor="k", colorbar=False, yearlabel_kws={'fontname':'sans-serif'})

plt.show();

# End of File