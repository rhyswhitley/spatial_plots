#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

from operator import itemgetter

from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import PowerNorm

from shapely.geometry import shape, Point, Polygon, MultiPoint, MultiPolygon

import fiona

# --------------------------------------------------------------------------------
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

    # extract the vertices of the polygon (the coord system)
    vert_2Dlist = [vl["geometry"]["coordinates"][0] for vl in fshape \
                   if vl["properties"]["GEZ_TERM"] in PFTS]

    # flatten 2D list
    vert_1Dlist = list_flat(vert_2Dlist)

    # define the path by which the lines of the polygon are drawn
    code_2Dlist = [create_codes(len(vl)) for vl in vert_2Dlist]
    # flatten 2D list
    code_1Dlist = list_flat(code_2Dlist)

    # create the art path that will clip the data (Multipolygons are flattened)
    part1 = Path(vert_1Dlist, code_1Dlist)
    clip = PathPatch(part1, *args, **kwargs)

    # create a multi-polygon object using the same list of coordinates
    #mpoly = MultiPolygon([shape(vl["geometry"]) for vl in fshape])

    x_low, y_low = map(min, zip(*vert_1Dlist))
    x_high, y_high = map(max, zip(*vert_1Dlist))

    # extents for the polygon
    extent = {'lon':[x_low, x_high], 'lat':[y_low, y_high]}

    # return to user
    return {'patch': clip, 'extent': extent}

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

# --------------------------------------------------------------------------------

def make_map(ax_0, cru_data, title, **kargs):
    # extract coordinate information
    lat = cru_data["lat"]
    lon = cru_data["lon"]
    data_var = cru_data['value']
    data_units = cru_data['meta']['units']

    # now create a global map canvas to plot on
    globe_map = Basemap(llcrnrlon=-120, llcrnrlat=-40, \
                        urcrnrlon=max(lon), urcrnrlat=40, \
                        resolution='l', projection='cyl', \
                        lat_0=0, lon_0=0, ax=ax_0)

    # draw spatial extras to denote land and sea
    globe_map.drawmapboundary(fill_color='dimgray')
    globe_map.drawcoastlines(color='black', linewidth=0.5)
    globe_map.fillcontinents(color='lightgray', lake_color='dimgray', zorder=0)
    globe_map.drawparallels(np.arange(-90, 90, 20), color='grey', labels=[1, 0, 0, 0])
    globe_map.drawmeridians(np.arange(0, 360, 30), color='grey', labels=[0, 0, 0, 1])

    # compute map proj coordinates
    lons, lats = np.meshgrid(lon, lat)
    x, y = globe_map(lons, lats)

    # plot data on the map
    cs = globe_map.contourf(x, y, data_var, **kargs)
    # add a colorbar
    cbar = globe_map.colorbar(cs, location='right', pad="2%", size="2%")
    cbar.set_label(data_units)

    # Title
    ax_0.set_title(title)

    return globe_map

    # import savanna bioregion polygon and create a clipping region
    sav_geom = define_clipping(SHAPEPATH, transform=ax_0.transData)
    # Clip and Rasterize the contour collections
    for contour in cs.collections:
        contour.set_clip_path(sav_geom['patch'])
        contour.set_rasterized(True)

    return globe_map

# --------------------------------------------------------------------------------

def pickle3_load(bin_file):
    with open(bin_file, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

# --------------------------------------------------------------------------------

def main():

    get_file = lambda x: "cru_ts3.23.1901.2014.{0}.100mean.pkl".format(x)

    tair_data = pickle3_load(FILEPATH + get_file('tmp'))
    rain_data = pickle3_load(FILEPATH + get_file('pre'))

    fig = plt.figure(figsize=(10, 6), frameon=False)
    fig.add_axes([0, 0, 1.0, 1.0])

    n_plots = 2
    grid = gridspec.GridSpec(n_plots, 1, hspace=0.3)
    subaxs = [plt.subplot(grid[i]) for i in range(n_plots)]

    # Mean Annual Temperature plot
    make_map(subaxs[0], tair_data, cmap=get_cmap(MAPCOLOR), levels=np.arange(15, 35, 0.5), \
             title="Savanna Bioregion \\\\ Mean Annual Temperature (1901 to 2013)")

    # Mean Annual Rainfall plot
    make_map(subaxs[1], rain_data, cmap=get_cmap(MAPCOLOR), levels=np.arange(0, 600, 10), \
             norm=PowerNorm(gamma=0.5), \
             title="Savanna Bioregion \\\\ Mean Annual Precipitation (1901 - 2013)")

    plt.savefig(SAVEPATH)

if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/CRU/CRU_TS3/")
    SHAPEPATH = os.path.expanduser("~/Savanna/Data/GiS/ecofloristic_zones/ecofloristic_zones.shp")
    SAVEPATH = os.path.expanduser("~/Work/Research_Work/Working_Publications/Savannas/SavReview/figures/Fig1_globalsav.pdf")

    # PFTS that <broadly> define/encompass global savannas
    PFTS = ["Tropical moist deciduous forest", \
            "Tropical dry forest", \
            "Subtropical dry forest", \
            "Tropical shrubland"]

    MAPCOLOR = 'viridis'

    main()

