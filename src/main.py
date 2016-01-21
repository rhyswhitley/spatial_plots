#!/usr/bin/env python3

from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from shapely.geometry import shape, Point, Polygon, MultiPoint, MultiPolygon

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import datetime, time
import os, re

import fiona


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

def grid_resample(mapObj, _lons, _lats, _data, proj_res=0.05):
    """
    Resamples (or doesn't depending on what you pass) the gridded data to a new
    resolution. Resolution is defined by the proj_res parameter, which is the
    size of each grid cell in the maps units of projection.

    Currently works out the new grid based on the basemap object extents. Not
    sure this is the best way, and perhaps doing this independent of the
    basemap projection would be better?
    """

    # create x, y complex dimensions for creating the new grid
    nx = complex(0, int((mapObj.xmax - mapObj.xmin)/proj_res))
    ny = complex(0, int((mapObj.ymax - mapObj.ymin)/proj_res))

    # echo to user
    msg = 'Map resolution is {0} x {1}'.format(nx, ny)
    print(msg)

    # define the grid space based on the new dimensions
    xl, yl = np.mgrid[lonWest:lonEast:nx, latSouth:latNorth:ny]

    # re-grid the high-res data at lower-res conserve memory
    ds_zi = griddata(np.column_stack((_lons, _lats)), \
                     _data, (xl, yl), method='linear')

    # return the new grid
    return xl, yl, ds_zi

def define_clipping(_shapePath, *args, **kwargs):
    """
    Reads in a shapefile from some folder and creates a Matplotlib Patch artist
    from which one can clip gridded data plotted on a basemap object. The Patch
    object is defined using vertices (lat/lon coordinates) and codes (drawing
    commands), which make up the final PatchPath that is returned to the user.
    Additionally, a polygon object is also created to extract data points from
    a gridded dataset that exist with the polygon's extents.
    """

    # import the shapefile using fiona
    fshape = fiona.open(_shapePath)

    # extract the vertices of the polygon (the coord system) **weirdly stored
    # as a list of lists
    vert_2Dlist = [vl[0] for vl in fshape.next()["geometry"]["coordinates"]]
    # flatten 2D list
    vert_1Dlist = list_flat(vert_2Dlist)

    # define the path by which the lines of the polygon are drawn
    code_2Dlist = [create_codes(len(vl)) for vl in vert_2Dlist]
    # flatten 2D list
    code_1Dlist = list_flat(code_2Dlist)

    # create the art path that will clip the data (Multipolygons are flattened)
    clip = PathPatch(Path(vert_1Dlist, code_1Dlist), *args, **kwargs)

    # create a multi-polygon object using the same list of coordinates
    mpoly = MultiPolygon([shape(vl["geometry"]) for vl in fshape])

    # return to user
    return {'patch': clip, 'polygon': mpoly}

def list_flat(List2D):
    """Flattens a 2D list"""
    return [item for sublist in List2D for item in sublist]

def create_codes(plen):
    """
    Returns a list of matplotlib artist drawing codes based on the number of
    polygon coordinates; First index is the starting point, Last index closes
    the polygon, and all other indices draw the polygon (coordinates always
    loop back to origin)
    """
    return [Path.MOVETO] + [Path.LINETO]*(plen-2) + [Path.CLOSEPOLY]

def paint_map(ax_0, dataset, levels):
    """
    Creates a plot canvas using basemap to visualise geospatial data, which is
    passed to this function as 'dataset'. Extents of the map are based on the
    global settings at the bottom of the file [need to change this to calculate
    extents from the imported data].

    The projection used here is cylindrical, although you could change this to
    anything else, but be warned, this will created problems with the clipping,
    which is basic lat/lon. If you change the projection, the clipping projection
    must also be changed [not currently automatic].
    """

    if (lonWest < 0) and (lonEast < 0):
        lon_0 = -(abs(lonEast) + abs(lonWest))/2.0
    elif (lonWest > 0) and (lonEast > 0):
        lon_0 = (abs(lonEast) + abs(lonWest))/2.0
    else:
        lon_0 = (lonEast + lonWest)/2.0

    # create a Basemap canvas to plot the data on [NOTE: be aware of projection
    # geo-coordinate system]
    oz_map = Basemap(llcrnrlon=lonWest, llcrnrlat=latSouth,
                     urcrnrlon=lonEast, urcrnrlat=latNorth, \
                     resolution='i', projection='cyl', \
                     lat_0=latNorth, lon_0=lon_0, ax=ax_0)

    # draw spatial extras to denote land and sea
    oz_map.drawmapboundary(fill_color='dimgray')
    oz_map.drawcoastlines(color='black', linewidth=0.5)
    oz_map.fillcontinents(color='lightgray', lake_color='dimgray', zorder=0)

    # draw parallels and meridians.
    oz_map.drawparallels(np.arange(-80, 90, 5), color='grey', labels=[1, 0, 0, 0])
    oz_map.drawmeridians(np.arange(0, 360, 5), color='grey', labels=[0, 0, 0, 1])

    # draw gridded data onto the map canvas
    lx, ly, zi = grid_resample(oz_map, dataset['lon'], dataset['lat'], \
                               dataset['data'], res=0.05)
    x, y = oz_map(lx, ly)
    cs = oz_map.contourf(x, y, zi, levels, cmap=get_cmap(MAPCOLOR, 100))

    # import savanna bioregion polygon and create a clipping region
    sav_geom = define_clipping(SHAPEPATH, transform=ax_0.transData)

    # Clip and Rasterize the contour collections
    for contour in cs.collections:
        contour.set_clip_path(sav_geom['patch'])
        contour.set_rasterized(True)

    return oz_map

def plot_map():
    """
    Top level function that draws the final figure: could be one or
    multiple maps.

    Warning: This isn't complete yet.
    """

    tair = import_data("anuclim_5km_mat.csv", doSum=False)
    #rain = import_data("anuclim_5km_ppt.csv", doSum=True)

    fig = plt.figure(figsize=(12, 9), frameon=False)
    fig.add_axes([0, 0, 1.0, 1.0])
#    n_plots = 2
#    grid = gridspec.GridSpec(n_plots, 1)
#    subaxs = [plt.subplot(grid[i]) for i in range(n_plots)]
    subaxs = plt.subplot(111)

    #subaxs[0].set_title("Temperature")
    map1 = paint_map(subaxs, tair, np.linspace(20, 32, 100))
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

    MAPCOLOR = 'GnBu'
    #MAPCOLOR = 'Viridis' # need to register

    latNorth = -5
    latSouth = -30
    lonWest = 110
    lonEast = 155

    plot_map()


# JUNK BELOW

#def shape_to_Patch(shape_obj):
#    """
#    This is a temporary solution to create a Patch object that is used to clip
#    the Basemap object for plotting data bounded by a shapefile.
#    """
#    for shape_rec in shape_obj.shapeRecords():
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
#
#        clip = PathPatch(Path(vertices, codes), edgecolor='k', facecolor=None, transform=ax_0.transData)
#
#        return clip
#
#    print(zi)
#    mask = np.zeros((len(x), len(y)))
#    for i, p_lon in enumerate(x):
#        for j, p_lat in enumerate(y):
#            test = sav_geom['polygon'].contains(Point(x, y))
#            if test is True:
#                mask[i, j] = 1
#
