#!/usr/bin/env python3

from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep

#from matplotlib.mlab import griddata as griddata2
#from pylab import meshgrid

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import datetime, time
import shapefile
import os, re

import fiona
#import colormaps as cmaps


__author__ = 'Rhys Whitley'
__email__ = 'rhys.whitley@gmail.com'
__created__ = datetime.datetime(2015, 12, 14)
__modified__ = time.strftime("%c")
__version__ = '0.1'

def import_data(fname, doSum=False):
    """
    Reads in an geospatial data in a XYZ-flattened format saved as a CSV using
    pandas. Format of the file must be
    first column: longitude,
    second column: latitude,
    columns 3 to 14: values at months of the year (JFMAMJJASOND)

    Null values represent sea pixels which contain no land-surface information
    and are given a value of -999.

    Function returns an aggregated yearly value of the quantity, which can be
    returned as a sum or mean value.

    ** To do later: add option to extract monthly values
    """

    # header information
    f_hd = ['lon', 'lat'] + ['m' + str(i) for i in range(1, 13)]

    # import data
    data_matrix = pd.read_csv(DATAPATH + fname, header=None, names=f_hd, na_values=-999)

    # extract lat and lon data streams
    id_lons = np.around(np.array(data_matrix.ix[:, 'lon']), 3)
    id_lats = np.around(np.array(data_matrix.ix[:, 'lat']), 3)

    # extract the monthly value columns
    yval = filter(lambda x: re.search('^m[0-9]', x), f_hd)

    # depends on option: either sum or average monthly columns
    if doSum is False:
        id_clim = np.array(data_matrix.ix[:, yval].mean(axis=1))
    else:
        id_clim = np.array(data_matrix.ix[:, yval].sum(axis=1))
        # sea values will be summed, so reset those cells
        id_clim[id_clim < -999.] = -999.

    # return to user as a dict
    return pd.DataFrame({'lon':id_lons, 'lat':id_lats, 'data':id_clim})

def down_sample(mapObj, _lons, _lats, _data, samp_size=1e4):

    print(mapObj.xmax)
    print(mapObj.xmin)
    print(mapObj.ymax)
    print(mapObj.ymin)

    # create x, y complex dimensions for creating the new grid
    nx = complex(0, int((mapObj.xmax - mapObj.xmin)/samp_size))
    ny = complex(0, int((mapObj.ymax - mapObj.ymin)/samp_size))

    # echo to user
    msg = 'resolution is {0} x {1}'.format(nx, ny)
    print(msg)

    # define the grid space based on the new dimensions
    xl, yl = np.mgrid[lonWest:lonEast:nx, latSouth:latNorth:ny]

    # re-grid the high-res data at lower-res conserve memory
    ds_zi = griddata(np.column_stack((_lons, _lats)), \
                     _data, (xl, yl), method='linear')

    # return the new grid
    return xl, yl, ds_zi

def paint_map(ax_0, dataset, levels):

    if (lonWest < 0) and (lonEast < 0):
        lon_0 = -(abs(lonEast) + abs(lonWest))/2.0
    elif (lonWest > 0) and (lonEast > 0):
        lon_0 = (abs(lonEast) + abs(lonWest))/2.0
    else:
        lon_0 = (lonEast + lonWest)/2.0

    oz_map = Basemap(llcrnrlon=lonWest, llcrnrlat=latSouth, urcrnrlon=lonEast, urcrnrlat=latNorth, \
             resolution='i', projection='cyl', lat_0=latNorth, lon_0=lon_0, ax=ax_0)
    oz_map.drawmapboundary(fill_color='dimgray')
    oz_map.drawcoastlines(color='black', linewidth=0.5)
    oz_map.fillcontinents(color='white', lake_color='dimgray', zorder=0)
    # draw parallels and meridians.
    #oz_map.drawparallels(np.arange(-80, 90, 10), color='grey', labels=[1, 0, 0, 0])
    #oz_map.drawmeridians(np.arange(0, 360, 10), color='grey', labels=[0, 0, 0, 1])

    lx, ly, zi = down_sample(oz_map, dataset['lon'], dataset['lat'], dataset['data'], 0.05)
    x, y = oz_map(lx, ly)
    cs = oz_map.contourf(x, y, zi, levels, cmap=get_cmap(MAPCOLOR, 100))

    # add savanna bioregion polygon
    bio_file = os.path.splitext(SHAPEPATH)[0]
    sf = shapefile.Reader(bio_file)

    verts = [
        (130, -30), # left, bottom
        (130, -20), # left, top
        (140, -20), # right, top
        (140, -30), # right, bottom
        (130, -30) # ignored
        ]

    codes = [Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
            ]

    path = Path(verts, codes)
    print(path)
    patch = PathPatch(path, transform=ax_0.transData)


#    shp_info = oz_map.readshapefile(os.path.splitext(SHAPEPATH)[0], 'savanna', drawbounds=True)
#    #print(shp_info[-1]._paths[0].vertices)
#    paths = []
#    for line in shp_info[-1]._paths:
#        moves = len(line.vertices) - 2
#        my_codes = [Path.MOVETO] + [Path.LINETO]*moves + [Path.CLOSEPOLY]
#        my_verts = [(vx, vy) for (vx, vy) in line.vertices]
#        paths.append(Path(line.vertices, my_codes))
#
#    print(paths[0])
#    coll = PatchCollection(paths, linewidths=1, facecolors='pink', zorder=2)
#    ax_0.add_collection(patch)

#    for shape in oz_map.savanna:
#        print(len(shape))
#        patches.append(Polygon(np.array(shape), True))

    #ax_0.add_collection(PatchCollection(poly, facecolor='m', edgecolor='k', linewidths=1., zorder=2))

    # Rasterize the contour collections
#    plt.gca().add_patch(patch)
    for contour in cs.collections:
        contour.set_clip_path(patch)
        #contour.set_rasterized(True)

    return None

def plot_map():

    #shape = shapefile.Reader(os.path.splitext(SHAPEPATH)[0])
#    shape = fiona.open(SHAPEPATH)
#    pol = shape.next()
#    geom = pol['geometry']
#    poly_data = pol["geometry"]["coordinates"]
#    poly_data2 = poly_data[len(poly_data)-1][0]
#    poly = Polygon(poly_data2)


    tair = import_data("anuclim_5km_mat.csv", doSum=False)
    #rain = import_data("anuclim_5km_ppt.csv", doSum=True)

    fig = plt.figure(figsize=(12, 9), frameon=False)
    fig.add_axes([0, 0, 1.0, 1.0])
#    n_plots = 2
#    grid = gridspec.GridSpec(n_plots, 1)
#    subaxs = [plt.subplot(grid[i]) for i in range(n_plots)]
    subaxs = plt.subplot(111)

    #subaxs[0].set_title("Temperature")
    map1 = paint_map(subaxs, tair, np.arange(0, 32, 1))
#    map1 = paint_map(subaxs[0], tair, max_val=32)
#    bar1 = plt.colorbar(map1)
#    bar1.ax.set_xlabel('$\degree$C')

#    subaxs[1].set_title("Rainfall")
#    map2 = paint_map(subaxs[1], rain, max_val=4100)
#    bar2 = plt.colorbar(map2)
#    bar2.ax.set_xlabel('mm')

    plt.show()


if __name__ == '__main__':

    DATAPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/ANUCLIM/mean30yr/")
    SHAPEPATH = os.path.expanduser("~/Savanna/Data/GiS/Savanna-Boundary-crc-g/crc-g.shp")

    #MAPCOLOR = 'GnBu'
    MAPCOLOR = 'jet'

    latNorth = -5
    latSouth = -30
    lonWest = 110
    lonEast = 155

    plot_map()



#    # add savanna bioregion polygon
#    bio_file = os.path.splitext(SHAPEPATH)[0]
##    bio_name = os.path.basename(bio_file)
##    sf = oz_map.readshapefile(bio_file, bio_name)
#    sf = shapefile.Reader(bio_file)
#
#    for shape_rec in sf.shapeRecords():
#        vertices = []
#        codes = []
#        pts = shape_rec.shape.points
#        prt = list(shape_rec.shape.parts) + [len(pts)]
#        for i in range(len(prt) - 1):
#            for j in range(prt[i], prt[i+1]):
#                vertices.append((pts[j][0], pts[j][1]))
#            codes += [Path.MOVETO]
#            codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
#            codes += [Path.CLOSEPOLY]
#        clip = PathPatch(Path(vertices, codes), transform=ax_0.transData)
#
#    print(type(clip))
#    return 1
#
