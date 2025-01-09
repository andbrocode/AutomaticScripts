#!/bin/python3

'''
get arrivals times for seismic phases of earthquake from 1D Earth model
'''

from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

# event information to set
lat_event = 47.29 # in degrees
lon_event = 151.15 # in degrees
dep_event = 10 # in km

# location of station
lat_georg = 50.728310
lon_georg = 7.089011

# get distance to event
distance_in_degree = locations2degrees(lat_georg, lon_georg, lat_event, lon_event)

# define 1D earth model
model = TauPyModel(model="iasp91")

# compute arrival times
arrivals = model.get_travel_times(source_depth_in_km=dep_event,
                                  distance_in_degree=distance_in_degree,
                                  phase_list=["P", "S"]
                                 )

print(arrivals)

# compute ray paths
paths = model.get_ray_paths(source_depth_in_km=dep_event,
                            distance_in_degree=distance_in_degree,
                            phase_list=["P", "S"]
                            )

# show ray paths
paths.plot_rays()