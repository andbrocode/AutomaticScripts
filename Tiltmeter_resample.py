#!/usr/bin/env python
# coding: utf-8

# ## Tiltmeter - resample and write data

# ### Import Libraries


import os
import json
import pprint

from obspy import UTCDateTime, read, Stream
from andbro__readYaml import __readYaml
from pandas import date_range
from tqdm import tqdm
from obspy.clients.filesystem.sds import Client

# ### Setup

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

# ### Define Functions


def __conversion(st, conf):

    def convertTemp(trace):
        Tvolt = trace.data * conf.get('gainTemp')
        coeff = conf.get('calcTempCoefficients')
        return coeff[0] + coeff[1]*Tvolt + coeff[2]*Tvolt**2 + coeff[3]*Tvolt**3

    def convertTilt(trace, conversion, sensitivity):
        return trace.data * conversion * sensitivity

    for tr in st:
        if tr.stats.channel[-1] == 'T':
            tr.data = convertTemp(tr)
        elif tr.stats.channel[-1] == 'N':
            tr.data = convertTilt(tr, conf['convPTN'], conf['gainTilt'])
        elif tr.stats.channel[-1] == 'E':
            tr.data = convertTilt(tr, conf['convPTE'], conf['gainTilt'])
        else:
            print("no match")

    print(f"  -> converted data of {st[0].stats.station}")
    return st


def __write_SDS(st, config):

    ## check if output_path and output_file is set in config
    if not "output_path" in config.keys():
        print(" -> missing config key: output_path")
        return

    ## check if output path exists
    if not os.path.exists(config['output_path']):
        print(f" -> {config['output_path']} does not exist!")
        return

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, tr.stats.starttime.julday

        if not os.path.exists(config['output_path']+f"{yy}/"):
            os.mkdir(config['output_path']+f"{yy}/")
            print(f"creating: {config['output_path']}{yy}/")
        if not os.path.exists(config['output_path']+f"{yy}/{nn}/"):
            os.mkdir(config['output_path']+f"{yy}/{nn}/")
            print(f"creating: {config['output_path']}{yy}/{nn}/")
        if not os.path.exists(config['output_path']+f"{yy}/{nn}/{ss}/"):
            os.mkdir(config['output_path']+f"{yy}/{nn}/{ss}/")
            print(f"creating: {config['output_path']}{yy}/{nn}/{ss}/")
        if not os.path.exists(config['output_path']+f"{yy}/{nn}/{ss}/{cc}.D"):
            os.mkdir(config['output_path']+f"{yy}/{nn}/{ss}/{cc}.D")
            print(f"creating: {config['output_path']}{yy}/{nn}/{ss}/{cc}.D")

    for tr in st:
        nn, ss, ll, cc = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        yy, jj = tr.stats.starttime.year, str(tr.stats.starttime.julday).rjust(3,"0")

        try:
            st_tmp = st.copy()
            st_tmp.select(channel=cc).write(config['output_path']+f"{yy}/{nn}/{ss}/{cc}.D/"+f"{nn}.{ss}.{ll}.{cc}.D.{yy}.{jj}", format="MSEED")
        except:
            print(f" -> failed to write: {cc}")


def __load_local(config, seed_id, tdate):

    st0 = Stream()

    doy  = str(tdate.julday).rjust(3,"0")
    year = str(tdate.year)
    net, sta, loc, cha = seed_id.split(".")

    try:
#                 st0 = read(config['datapath']+year+"/"+net+"/"+sta+"/"+cha[:-1]+c+".D/"+filename)
        st0 = Client(config['datapath']).get_waveforms(net, sta, loc, cha, 
                                                       tdate-config['temporal_puffer'], 
                                                       tdate+86400+config['temporal_puffer']
                                                       )

        config['output_file'] = f"{net}.{sta}.{loc}.{cha}.D.{tdate.year}.{str(tdate.julday).rjust(3,'0')}"

    except:
        print(f" -> failed to load {tdate}!")
        return st0

    if len(st0) > 3:
        st0.merge(fill_value='interpolate')
        print(f" -> merged {tdate}!")

    return st0


# ### Configurations
config = {}


## define new sampling rate
config['resample_rate'] = float(input("Enter new sampling rate [1/600]: "))
#config['resample_rate'] = 1/600

## define seed code of instrument
config['seed_id'] = input("Enter seed id [BW.DROMY..LA*]: ")
#config['seed_id'] = "BW.DROMY..LA*"



## specify path to data
config['datapath'] = archive_path+f"romy_archive/"

## define time window
config['tbeg'] = UTCDateTime(str(input("Enter starttime: ")))
config['tend'] = UTCDateTime(str(input("Enter endtime:   ")))
#config['tbeg'] = UTCDateTime("2023-12-01 00:00")
#config['tend'] = UTCDateTime("2023-12-31 00:00")

## add some time before and after one day to avoid filter effects
config['temporal_puffer'] = 6*3600 # seconds

## specify path for output data
config['output_path'] = data_path+f"TiltmeterDataBackup/Tilt_downsampled/"


# ### Tiltmeter Conversion

def main():

    #print(json.dumps(config, indent=4, sort_keys=True))

    # ### Load Tiltmeter Data
    for date in tqdm(date_range(str(config['tbeg']), str(config['tend']))):

        tdate = UTCDateTime(date)

        ## load data from archive
        st = __load_local(config, config['seed_id'], tdate)

        if len(st) == 0:
            print(f" -> empty stream: {tdate}")
            continue

        ## convert data to tilt units
        # if config['seed_id'].split(".") == "TROMY":
        #     st = __conversion(st, confPT)
        # elif config['seed_id'].split(".") == "ROMYT":
        #     st = __conversion(st, confTRII['PT'])

        ## taper
        st = st.taper(0.1)

        ## resample data
        st = st.resample(config['resample_rate'], no_filter=True)

        ## trim data to actual date
        st = st.trim(tdate, tdate+86400, nearest_sample=False)

        ## write data to mseed-files
        __write_SDS(st, config)


    print(f" -> stored data to: {config['output_path']}")

    print("\n Done!")

if __name__ == "__main__":
    main()

# ## End Of File
