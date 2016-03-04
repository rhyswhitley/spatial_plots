#!/usr/bin/env python3

import os, re
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cPickle as pickle

#from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LogNorm

from shapely.geometry import shape, Point, Polygon, MultiPoint, MultiPolygon


# --------------------------------------------------------------------------------
def main():

    get_file = lambda x: "cru_ts3.23.1901.2014.{0}.100mean.pkl".format(x)

    tair_data = pickle.load(open(FILEPATH + get_file('tmp'), 'rb'))
    rain_data = pickle.load(open(FILEPATH + get_file('pre'), 'rb'))

    fig = plt.figure(figsize=(12, 9), frameon=False)
    fig.add_axes([0, 0, 1.0, 1.0])
    n_plots = 2
    grid = gridspec.GridSpec(n_plots, 1)
    subaxs = [plt.subplot(grid[i]) for i in range(n_plots)]

    map1 = make_map(subaxs[0], tair_data, logData=False)
    map2 = make_map(subaxs[1], rain_data, logData=True)

    plt.show()

def make_map(ax_0, cru_data, logData=False):
    # extract coordinate information
    lat = cru_data["lat"]
    lon = cru_data["lon"]
    data_var = cru_data['value']
    data_units = cru_data['meta']['units']

# --------------------------------------------------------------------------------

    # now create a global map canvas to plot on
    globe_map = Basemap(llcrnrlon=min(lon), llcrnrlat=min(lat),
                        urcrnrlon=max(lon), urcrnrlat=max(lat), \
                        resolution='i', projection='cyl', \
                        lat_0=lat[0], lon_0=lon[0], ax=ax_0)

    # draw spatial extras to denote land and sea
    globe_map.drawmapboundary(fill_color='dimgray')
    globe_map.drawcoastlines(color='black', linewidth=0.5)
    globe_map.fillcontinents(color='lightgray', lake_color='dimgray', zorder=0)

    globe_map.drawparallels(np.arange(-90, 90, 20), color='grey', labels=[1, 0, 0, 0])
    globe_map.drawmeridians(np.arange(0, 360, 30), color='grey', labels=[0, 0, 0, 1])


    ny = data_var.shape[0]; nx = data_var.shape[1]
    lons, lats = globe_map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = globe_map(lons, lats) # compute map proj coordinates.

    from matplotlib.ticker import LogFormatter
    l_f = LogFormatter(10, labelOnlyBase=False)

    levels = np.linspace(np.min([np.min(data_var), 1e-4]), np.max(data_var), 50)
    if logData is True:
        cs = globe_map.contourf(x, y, data_var, levels, cmap=get_cmap(MAPCOLOR, 100), norm=LogNorm())
        log_levels = np.logspace(1e-4, 6000, 50)
        cbar = globe_map.colorbar(cs, location='bottom', pad="10%", ticks=log_levels, format=l_f)
    else:
        cs = globe_map.contourf(x, y, data_var, levels, cmap=get_cmap(MAPCOLOR, 100))
        cbar = globe_map.colorbar(cs, location='bottom', pad="10%")

    cbar.set_label(data_units)

    #plt.show()

    return globe_map

if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/CRU/CRU_TS3/")

    MAPCOLOR = 'jet'

    main()
