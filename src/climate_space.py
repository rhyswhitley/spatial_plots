#!/usr/bin/env python2.7

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import SymLogNorm #PowerNorm

from shapely.geometry import Polygon, MultiPolygon, Point

#import seaborn as sns

# --------------------------------------------------------------------------------

def make_map(ax_0, cru_data, clipping, cticks, title, **kargs):

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

    sea_color = 'dimgray'
    globe_map.drawmapboundary(fill_color=sea_color)
    globe_map.drawcoastlines(color='black', linewidth=0.5)
    globe_map.fillcontinents(color='lightgray', lake_color=sea_color, zorder=0)
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
    cbar.set_ticks(cticks)

    # Title
    ax_0.set_title(title, fontsize=12)

    sav_geom = PathPatch(clipping, transform=ax_0.transData)

    # Clip and Rasterize the contour collections
    for contour in cs.collections:
        contour.set_clip_path(sav_geom)
        contour.set_rasterized(True)

    return globe_map

# --------------------------------------------------------------------------------

def pickle3_load(bin_file):
    """
    There is some bug with unpacking binary values from pickle objects in
    python 3 - this is my temporary fix.
    """
    with open(bin_file, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

# --------------------------------------------------------------------------------

def main():

    get_file = lambda x: "cru_ts3.23.1901.2014.{0}.100mean.pkl".format(x)

    # import data
    tair_data = pickle.load(open(FILEPATH + get_file('tmp'), 'rb'))
    rain_data = pickle.load(open(FILEPATH + get_file('pre'), 'rb'))

    # import savanna bioregion geometry
    sav_geom = pickle.load(open(PATCHPATH, 'rb'))

    # convert vertices to multipolygon object
    #sav_poly = MultiPolygon([Polygon(pl) for pl in sav_geom['poly']])
    sav_poly = [Polygon(pl) for pl in sav_geom['poly']]

    # transform geo-coords values to grids (same dim as the climate data)
    lat_grid, lon_grid = np.meshgrid(tair_data['lon'], tair_data['lat'])

    # flatten all the data to store as a pandas dataframe so we can use with
    # seaborn graphics
    tair = tair_data['value'].flatten()
    # ** scale rainfall up to an annual value (mm/month -> mm/year)
    rain = rain_data['value'].flatten() * 12
    lons = lon_grid.flatten()
    lats = lat_grid.flatten()

    # create labels to differentiate savanna from global points
    output = []

# >> slow as hell
#    for i, (gx, gy) in enumerate(zip(lons, lats)):
#        print "percent: {0:.6f}".format(float(i)/float(len(lons)))
#        for j, sp in enumerate(sav_poly):
#            if Point(gx, gy).contains(sp):
#                output.append("savanna")

# >> faster but still way too slow
#    for i, (gx, gy) in enumerate(zip(lons, lats)):
#        print "percent: {0:.6f}".format(float(i)/float(len(lons)))
#        if sav_geom['clip'].contains_point((gx, gy)):
#            output.append("savanna")

    moo = sav_geom['clip'].contains_points(zip(lons, lats))

    print(len(output))

    return 1


    print(len(tair))
    print(len())


    fig = plt.figure(figsize=(10, 6), frameon=False)

    plt.plot(rain*12, tair, 'o', color='gray', markeredgecolor='none', alpha=0.2)

    plt.xlim([1e-1, 2e4])
    plt.ylim([-35, 35])

    plt.xscale('log')

    #plt.show()


if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/CRU/CRU_TS3/")
    IMAGEPATH = os.path.expanduser("~/Work/Research_Work/GiS_Data/Images/blue_marble/noaa_world_topo_bathymetric_lg.jpg")
    PATCHPATH = os.path.expanduser("~/Savanna/Data/GiS/Savanna_Bioregion_Path.pkl")
    SAVEPATH = os.path.expanduser("~/Work/Research_Work/Working_Publications/Savannas/SavReview/figures/Fig1_globalsav.pdf")

    # PFTS that <broadly> define/encompass global savannas
    PFTS = ["Tropical moist deciduous forest", \
            "Tropical dry forest", \
            "Subtropical dry forest", \
            "Tropical shrubland"]

    MAPCOLOR = 'viridis'

    main()

