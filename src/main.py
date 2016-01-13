#!/usr/bin/env python3

from mpl_toolkits.basemap import Basemap
from pylab import cm, meshgrid
from matplotlib.mlab import griddata

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import datetime, time
import shapefile
import os, re

__author__ = 'Rhys Whitley'
__email__ = 'rhys.whitley@gmail.com'
__created__ = datetime.datetime(2015, 12, 14)
__modified__ = time.strftime("%c")
__version__ = '0.1'

def plot_map():

    temp = shapefile.Reader(os.path.splitext(SHAPEPATH)[0])

    import_data(doSum=True)
    return 1

    fig = plt.figure(figsize=(12, 9), frameon=False)
    fig.add_axes([0, 0, 1.0, 1.0])

    n_plots = 2
    grid = gridspec.GridSpec(n_plots, 1)
    subaxs = [plt.subplot(grid[i]) for i in range(n_plots)]

    subaxs[0].set_title("Temperature")
    paint_map(subaxs[0])
    subaxs[1].set_title("Rainfall")
    paint_map(subaxs[1])

    plt.show()

def import_data(fname="anuclim_5km_mat.csv", doSum=False):

    f_hd = ['lon', 'lat'] + ['m' + str(i) for i in range(1, 13)]

    data_matrix = pd.read_csv(DATAPATH + fname, header=None, names=f_hd)
    lons = np.around(np.array(data_matrix.ix[:, 'lon']), 3)
    lats = np.around(np.array(data_matrix.ix[:, 'lat']), 3)
    yval = filter(lambda x: re.search('^m[0-9]', x), f_hd)
    if doSum is False:
        clim = np.array(data_matrix.ix[:, yval].mean(axis=1))
    else:
        clim = np.array(data_matrix.ix[:, yval].sum(axis=1))

    return lons, lats, clim

def attach_data(mapObj, lons, lats, data):

    lx, ly, zi = down_sample(mapObj, lons, lats, data)

    levels = np.arange(0, 40, 2)

    im = mapObj.contourf(lx, ly, zi, levels, cmap=cm.get_cmap(MAPCOLOR, len(levels)-1))
    # Rasterize the contour collections
    for imcol in im.collections:
        imcol.set_rasterized(True)

def down_sample(mapObj, lons, lats, data, samp_size=10e3):

    # Using DOWN-SAMPLING
    nx = int((mapObj.xmax - mapObj.xmin)/samp_size)+1
    ny = int((mapObj.ymax - mapObj.ymin)/samp_size)+1
    # define the grid space
    xi = np.linspace(lonStart, lonEnd, nx)
    yi = np.linspace(latStart, latEnd, ny)

    msg = 'resolution is {0} x {1}'.format(len(xi), len(yi))
    print(msg)

    # re-grid at a lower resolution
    zi = griddata(lons, lats, data, xi, yi, interp='nn')
    lx, ly = mapObj(*meshgrid(xi, yi))

    return lx, ly, zi

def paint_map(ax_0):

    if (lonStart < 0) and (lonEnd < 0):
        lon_0 = -(abs(lonEnd) + abs(lonStart))/2.0
    elif (lonStart > 0) and (lonEnd > 0):
        lon_0 = (abs(lonEnd) + abs(lonStart))/2.0
    else:
        lon_0 = (lonEnd + lonStart)/2.0

    oz_map = Basemap(llcrnrlon=lonStart, llcrnrlat=latEnd, urcrnrlon=lonEnd, urcrnrlat=latStart, \
             resolution='i', projection='tmerc', lat_0=latStart, lon_0=lon_0, ax=ax_0)
    oz_map.drawmapboundary(fill_color='dimgray')
    oz_map.fillcontinents(color='white',lake_color='dimgray')

    # add savanna bioregion polygon
    bio_file = os.path.splitext(SHAPEPATH)[0]
    bio_name = os.path.basename(bio_file)
    oz_map.readshapefile(bio_file, bio_name)
    return 1

if __name__ == '__main__':

    DATAPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/ANUCLIM/mean30yr/")
    SHAPEPATH = os.path.expanduser("~/Savanna/Data/GiS/Savanna-Boundary-crc-g/crc-g.shp")

    MAPCOLOR = 'jet'

    latStart = -5
    latEnd = -30
    lonStart = 110
    lonEnd = 155

    plot_map()
