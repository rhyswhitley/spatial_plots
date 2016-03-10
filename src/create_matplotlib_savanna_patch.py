#!/usr/bin/env python2.7

import os
import pickle
from matplotlib.path import Path

import fiona

def define_clipping(_shapePath):
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
    clip = Path(vert_1Dlist, code_1Dlist)
    #clip = PathPatch(part1, *args, **kwargs)

    # create a multi-polygon object using the same list of coordinates
    #mpoly = MultiPolygon([shape(vl["geometry"]) for vl in fshape])

    x_low, y_low = map(min, zip(*vert_1Dlist))
    x_high, y_high = map(max, zip(*vert_1Dlist))

    # extents for the polygon
    extent = {'lon':[x_low, x_high], 'lat':[y_low, y_high]}

    # return to user
    return {'clip': clip, 'extent': extent}

# --------------------------------------------------------------------------------

def list_flat(List2D):
    """Flattens a 2D list"""
    return [item for sublist in List2D for item in sublist]

# --------------------------------------------------------------------------------

def create_codes(plen):
    """
    Returns a list of matplotlib artist drawing codes based on the number of
    polygon coordinates; First index is the starting point, Last index closes
    the polygon, and all other indices draw the polygon (coordinates always
    loop back to origin)
    """
    return [Path.MOVETO] + [Path.LINETO]*(plen-2) + [Path.CLOSEPOLY]

# --------------------------------------------------------------------------------

def main():

    # import savanna bioregion polygon and create a clipping region
    sav_geom = define_clipping(SHAPEPATH)

    pickle.dump(sav_geom, open(SAVEPATH, "wb"))

    return 1

if __name__ == "__main__":

    SHAPEPATH = os.path.expanduser("~/Savanna/Data/GiS/ecofloristic_zones/ecofloristic_zones.shp")
    SAVEPATH = os.path.expanduser("~/Savanna/Data/GiS/Savanna_Bioregion_Path.pkl")

    # PFTS that <broadly> define/encompass global savannas
    PFTS = ["Tropical moist deciduous forest", \
            "Tropical dry forest", \
            "Subtropical dry forest", \
            "Tropical shrubland"]

    main()

