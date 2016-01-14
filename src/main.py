#!/usr/bin/env python3

from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
from scipy.interpolate import griddata
from matplotlib.mlab import griddata as griddata2
#from pylab import meshgrid

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

def import_data(fname, doSum=False):

    f_hd = ['lon', 'lat'] + ['m' + str(i) for i in range(1, 13)]

    data_matrix = pd.read_csv(DATAPATH + fname, header=None, names=f_hd)
    id_lons = np.around(np.array(data_matrix.ix[:, 'lon']), 3)
    id_lats = np.around(np.array(data_matrix.ix[:, 'lat']), 3)
    yval = filter(lambda x: re.search('^m[0-9]', x), f_hd)
    if doSum is False:
        id_clim = np.array(data_matrix.ix[:, yval].mean(axis=1))
    else:
        id_clim = np.array(data_matrix.ix[:, yval].sum(axis=1))

    return {'lon':id_lons, 'lat':id_lats, 'data':id_clim}

def down_sample(mapObj, _lons, _lats, _data, samp_size=10e3):

    # Using DOWN-SAMPLING
    nx = complex(0, int((mapObj.xmax - mapObj.xmin)/samp_size))
    ny = complex(0, int((mapObj.ymax - mapObj.ymin)/samp_size))
    msg = 'resolution is {0} x {1}'.format(nx, ny)
    print(msg)
    # define the grid space
    xl, yl = np.mgrid[lonWest:lonEast:nx, latSouth:latNorth:ny]
    ds_zi = griddata(np.column_stack((_lons, _lats)), \
                     _data, (xl, yl), method='nearest')
    return xl, yl, ds_zi

def paint_map(ax_0, dataset):

    if (lonWest < 0) and (lonEast < 0):
        lon_0 = -(abs(lonEast) + abs(lonWest))/2.0
    elif (lonWest > 0) and (lonEast > 0):
        lon_0 = (abs(lonEast) + abs(lonWest))/2.0
    else:
        lon_0 = (lonEast + lonWest)/2.0

    oz_map = Basemap(llcrnrlon=lonWest, llcrnrlat=latSouth, urcrnrlon=lonEast, urcrnrlat=latNorth, \
             resolution='i', projection='tmerc', lat_0=latNorth, lon_0=lon_0, ax=ax_0)
    oz_map.drawmapboundary(fill_color='dimgray')
    oz_map.drawcoastlines(color='black', linewidth=0.5)
    oz_map.fillcontinents(color='white', lake_color='dimgray', zorder=0)

    #attach_data(oz_map, dataset)
    lx, ly, zi = down_sample(oz_map, dataset['lon'], dataset['lat'], dataset['data'])

    levels = np.arange(0, 40, 2)
    x, y = oz_map(lx, ly)
    im = oz_map.contourf(x, y, zi, levels, cmap=get_cmap(MAPCOLOR, len(levels)-1))

    # Rasterize the contour collections
    for imcol in im.collections:
        imcol.set_rasterized(True)

    # add savanna bioregion polygon
    bio_file = os.path.splitext(SHAPEPATH)[0]
    bio_name = os.path.basename(bio_file)
    oz_map.readshapefile(bio_file, bio_name)
    return im

def plot_map():

    temp = shapefile.Reader(os.path.splitext(SHAPEPATH)[0])

    fig = plt.figure(figsize=(12, 9), frameon=False)
    fig.add_axes([0, 0, 1.0, 1.0])

    tair = import_data("anuclim_5km_mat.csv", doSum=False)
    rain = import_data("anuclim_5km_ppt.csv", doSum=True)

    n_plots = 2
    grid = gridspec.GridSpec(n_plots, 1)
    subaxs = [plt.subplot(grid[i]) for i in range(n_plots)]

    subaxs[0].set_title("Temperature")
    map1 = paint_map(subaxs[0], tair)
    bar1 = plt.colorbar(map1)
    bar1.ax.set_xlabel('$\degree$C')

    subaxs[1].set_title("Rainfall")
    map2 = paint_map(subaxs[1], rain)
    bar2 = plt.colorbar(map2)
    bar2.ax.set_xlabel('mm')

    plt.show()


if __name__ == '__main__':

    DATAPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/ANUCLIM/mean30yr/")
    SHAPEPATH = os.path.expanduser("~/Savanna/Data/GiS/Savanna-Boundary-crc-g/crc-g.shp")

    MAPCOLOR = 'GnBu'

    latNorth = -5
    latSouth = -30
    lonWest = 110
    lonEast = 155

    plot_map()
