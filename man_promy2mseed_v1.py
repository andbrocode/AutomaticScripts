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
from obspy import Stream, Trace, UTCDateTime
from pandas import read_csv

## _______________________________
## configurations

net, sta, loc = "BW", "PROMY", ""

#path_to_data = "/import/kilauea-data/LNM2/infrasound/PROMY/data/WSX.D/2023/"
path_to_data = "/import/freenas-ffb-01-data/temp_archive/2023/BW/PROMY/WS3.D/"

path_to_sds = "/import/freenas-ffb-01-data/temp_archive/"

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

files = os.listdir(path_to_data)
files.sort()

for file in files:
    st0 = Stream()

    df = read_csv(path_to_data+file)

    ## check data sample size
    if df.shape[0] != 86400:
        print(f"-> {file} - Error: size not 86400 but {df.shape[0]}")
        continue

    tr1 = Trace()
    tr1.stats.starttime = UTCDateTime(df.datetime_UTC[0])
    tr1.stats.delta = 1
    tr1.stats.network = net
    tr1.stats.station = sta
    tr1.stats.location = loc
    tr1.stats.channel = "LKI"
    tr1.data = df.temperature_degC.to_numpy()

    tr2 = Trace()
    tr2.stats.starttime = UTCDateTime(df.datetime_UTC[0])
    tr2.stats.delta = 1
    tr2.stats.network = net
    tr2.stats.station = sta
    tr2.stats.location = loc
    tr2.stats.channel = "LDI"
    tr2.data = df.pressure_Pa.to_numpy()

    st0 += tr1
    st0 += tr2

    st0 = st0.merge()

    __write_stream_to_sds(st0, path_to_sds)


## _______________________________
## End of File
