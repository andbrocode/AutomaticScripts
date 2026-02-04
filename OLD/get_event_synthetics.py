#!/bin/python3

'''
get synthetics for earthquake from syngine database (very basic!)
'''

from obspy.clients.syngine import Client

client = Client()

syn_model = "iasp91_2s" # "ak135f_2s" | "prem_a_2s" | "iasp91_2s"

# get event data at: http://ds.iris.edu/spud/momenttensor
lat_event = 47.29 # in degrees
lon_event = 151.15 # in degrees
time_event = "2024-12-27 12:47:36.7"
dep_event = 10 # in km
event_id = "C202412271247A"

# station coordinates
lat_georg = 50.728310247878376
lon_georg = 7.0890113662166385

# obtain sythetics
synx = client.get_waveforms(model=syn_model,
                            receiverlatitude=lat_georg,
                            receiverlongitude=lat_georg,
                            networkcode="XX",
                            stationcode="GEORG",
                            locationcode=None,
                            eventid=f"GCMT:{event_id}",
                            sourcelatitude=lat_event,
                            sourcelongitude=lon_event,
                            sourcedepthinmeters=dep_event*1000,
                            origintime=time_event,
                            starttime=None,
                            endtime=None,
                            dt=1/20,
                           )

# plot seismograms
synx.plot();

# End of File