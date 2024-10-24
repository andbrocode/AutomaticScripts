#!/bin/python3
#
# converts csv files of PROMY sensor to mseed day files in SDS file structure
#
# Andreas Brotzer | 2023-09-18
#
# updated by AB 2023-10-31
# updated by AB 2023-11-02

## _______________________________
## libraries

import os
import sys

from obspy import Stream, Trace, UTCDateTime
from pandas import read_csv, DataFrame, merge
from datetime import datetime
from numpy import array, nan

## _______________________________
## configurations

# pi_num = os.uname()[1][-2:]
# pi_num = "04"

# hand over pi number [1-9]
pi_num = sys.argv[1]

## date
if len(sys.argv) > 1:
    dat = UTCDateTime(str(sys.argv[2]))
else:
    dat = UTCDateTime.now() - 86400 ## date for yesterday

year = str(dat.year)
doy = str(dat.julday).rjust(3,"0")

## seed before
net0,sta0,loc0,cha0 = "BW", "WROMY", "", f"WS{pi_num[-1]}"

## seed after
net1,sta1,loc1,cha1 = "BW", "WROMY", f"0{pi_num[-1]}", ""


# path_to_data = f"/home/pi/PROMY/data/PS{pi_num[-1]}.D/{year}/"
path_to_data = f"/import/freenas-ffb-01-data/romy_archive/{year}/{net0}/{sta0}/WS{pi_num[-1]}.D/"

# path_to_sds = "/home/pi/PROMY/data/mseed/"
path_to_sds = "/import/freenas-ffb-01-data/temp_archive/"

## time delta
dt = 1

## _______________________________
## methods

def __write_stream_to_sds(st, path_to_sds):

    import os

    ## check if output path exists
    if not os.path.exists(path_to_sds):
        print(f" -> {path_to_sds} does not exist!")
        return

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, tr.stats.starttime.julday

        if not os.path.exists(path_to_sds+f"{yy}/"):
            os.mkdir(path_to_sds+f"{yy}/")
            print(f"creating: {path_to_sds}{yy}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/")
            print(f"creating: {path_to_sds}{yy}/{nn}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/")
            print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/")
        if not os.path.exists(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D"):
            os.mkdir(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D")
            print(f"creating: {path_to_sds}{yy}/{nn}/{ss}/{cc}.D")

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

        try:
            st_tmp = st.copy()
            st_tmp.select(network=nn, station=ss, location=ll, channel=cc).write(path_to_sds+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")
        except:
            print(f" -> failed to write: {cc}")
        finally:
            print(f" -> stored stream as: {yy}/{nn}/{ss}/{cc}.D/{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}")

## _______________________________
## main

st0 = Stream()

file=f"{net0}.{sta0}.{cha0}.D.{year}.{doy}"

if not os.path.isfile(path_to_data+file):
    print(f" -> file: {file} does not exist!")
    quit()
else:
    df = read_csv(path_to_data+file)

## check data sample size
if df.shape[0] != 86400:
    print(f"-> {file} - Error: size not 86400 but {df.shape[0]}")
    #continue

## get current date from file name
day_num = file[-3:]
date = datetime.strptime(year + "-" + day_num, "%Y-%j").strftime("%Y%m%d")

## create data frame for all times of day
df_nan = DataFrame()
df_nan['datetime_UTC'] = array([str(UTCDateTime(date)+_t).split('.')[0].replace('-','').replace(':','') for _t in range(0, 86400, dt)])

## add datetime_UTC column
df['datetime_UTC'] = array([f"{d}T{str(t).rjust(6,'0')}" for d, t in zip(df["Date"], df["Time (UTC)"])])

## remove doubles
df.drop_duplicates(subset="datetime_UTC", inplace=True)
df.reset_index(inplace=True,)

## replace all -9999 error code values with NaN
df.replace(-9999, nan, inplace=True)

## merge dataframes to replace missing datetimes in data with nan and make sure dataframe is full
df = merge(df, df_nan, on="datetime_UTC", how="right")


tr1 = Trace()
tr1.stats.starttime = UTCDateTime(df.datetime_UTC[0])
tr1.stats.delta = dt
tr1.stats.network = net1
tr1.stats.station = sta1
tr1.stats.location = loc1
tr1.stats.channel = "LKI"
tr1.data = df['Temperature (°C)'].to_numpy()

tr2 = Trace()
tr2.stats.starttime = UTCDateTime(df.datetime_UTC[0])
tr2.stats.delta = dt
tr2.stats.network = net1
tr2.stats.station = sta1
tr2.stats.location = loc1
tr2.stats.channel = "LDI"
tr2.data = df['Pressure (hPa)'].to_numpy()

tr3 = Trace()
tr3.stats.starttime = UTCDateTime(df.datetime_UTC[0])
tr3.stats.delta = dt
tr3.stats.network = net1
tr3.stats.station = sta1
tr3.stats.location = loc1
tr3.stats.channel = "LII"
tr3.data = df['rel. Humidity (%)'].to_numpy()

st0 += tr1
st0 += tr2
st0 += tr3


st0 = st0.merge()


__write_stream_to_sds(st0, path_to_sds)


## _______________________________
## End of File
