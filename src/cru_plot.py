#!/usr/bin/env python3

import os, re
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def main():

    # connect to netcdf file
    cru_rain = nc.Dataset(FILEPATH)
    # echo attributes to screen
    print(cru_rain.variables)

# --------------------------------------------------------------------------------

    # extract coordinate information
    lat = cru_rain.variables["lat"][:].flatten
    lon = cru_rain.variables["lon"][:].flatten

    lon_0 = -cru_rain.variables['lon'][:][0]
    lat_0 = cru_rain.variables['lat'][:][0]

    # extract time information
    time_sec = cru_rain.variables['time']
    time = time_sec[:].flatten
    sec_orig = re.search(r'\d+.*', str(time_sec.units)).group(0)
    print(sec_orig)

    # we need to summarise the 100+ year information to an aggregate map
    rain_var = cru_rain.variables["pre"]
    rain = rain_var[:]
    rain_units = rain_var.units
    # this just determines the average of days NOT YEARS
    #rain_100 = np.average(rain, axis=0)
    rain_100 = rain[0, :, :]

# --------------------------------------------------------------------------------

    # now create a global map canvas to plot on

    globe_map = Basemap(llcrnrlon=lonWest, llcrnrlat=latSouth,
                        urcrnrlon=lonEast, urcrnrlat=latNorth, \
                        resolution='i', projection='cyl', \
                        lat_0=latNorth, lon_0=lon_0)

    # draw spatial extras to denote land and sea
    globe_map.drawmapboundary(fill_color='dimgray')
    globe_map.drawcoastlines(color='black', linewidth=0.5)
    globe_map.fillcontinents(color='lightgray', lake_color='dimgray', zorder=0)

    globe_map.drawparallels(np.arange(-90, 90, 20), color='grey', labels=[1, 0, 0, 0])
    globe_map.drawmeridians(np.arange(0, 360, 30), color='grey', labels=[0, 0, 0, 1])

    levels = np.linspace(0, 600, 50)

    ny = rain_100.shape[0]; nx = rain_100.shape[1]
    lons, lats = globe_map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = globe_map(lons, lats) # compute map proj coordinates.

    cs = globe_map.contourf(x, y, rain_100, levels, cmap=get_cmap(MAPCOLOR, 100))

    cbar = globe_map.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label(rain_units)

    plt.show()

    return 1

if __name__ == "__main__":

    FILENAME = "cru_ts3.23.1901.2014.pre.dat.nc"
    FILEPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/CRU/" + FILENAME)

    latNorth = 90
    latSouth = -90
    lonWest = -180
    lonEast = 180

    MAPCOLOR = 'jet'

    main()
