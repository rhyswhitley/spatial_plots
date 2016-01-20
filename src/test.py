from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def draw_screen_poly(lats, lons, m):
    x, y = m(lons, lats)
    xy = zip(x, y)
    poly = Polygon(xy, facecolor='red', alpha=0.4)
    plt.gca().add_patch(poly)

def main():

#    lats = [-30, 30, 30, -30 ]
#    lons = [-50, -50, 50, 50 ]
    lons = [130, 130, 140, 140]
    lats = [-30, -20, -20, -30]

    #m = Basemap(projection='tmerc',lon_0=0, lat_1=0)
    m = Basemap(llcrnrlon=lonWest, llcrnrlat=latSouth, urcrnrlon=lonEast, urcrnrlat=latNorth, \
             resolution='i', projection='cyl', lat_0=latNorth, lon_0=-10)
    m.drawcoastlines()
    m.drawmapboundary()
    draw_screen_poly(lats, lons, m)

    plt.show()

if __name__ == "__main__":

    latNorth = -5
    latSouth = -50
    lonWest = 110
    lonEast = 155

    main()
