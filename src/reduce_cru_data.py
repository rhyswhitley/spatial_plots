#!/usr/bin/env python3

import os
import netCDF4 as nc
import numpy as np
import pickle as pickle

from joblib import Parallel, delayed
from multiprocessing import cpu_count

def determine_tseries_aggvalue(cru_data, f_month2year, f_summary):
    """
    This function reduces the original CRU dataset from a Z-length array
    of gridded monthly values to a summary value for the entire time-series
    of the dataset.

    f_month2year :: the function that aggregates the monthly values to an annual value
    - accepts any funciton but recommends numpy default library to preserve the masking

    f_summary :: summarises the annual values across the time series to some reduce metric
    - accepts any funciton but recommends numpy default library to preserve the masking
    """
    # we need to summarise the 100+ year information to an aggregate map
    data = cru_data[:]
    # this just determines the average of days NOT YEARS
    n_years = data.shape[0]/12
    # split the array into a list by the number of years to reduce on
    data_list = np.split(data, n_years)
    # reduce the monthly values to an aggregate annual value using F2(X)
    data_years = [f_month2year(rl, axis=0) for rl in data_list]
    # reduce annual values to an aggregated value for the entire time-seires using F1(X)
    data_ts = f_summary(np.ma.dstack(data_years), axis=2)

    # return final reduce map to user
    return data_ts

# --------------------------------------------------------------------------------

def extract_CRU_data(fpath, *args, **kwards):

    # connect to netcdf file
    cru_data = nc.Dataset(os.path.join(fpath[0], fpath[1]))

    # echo to user
    print('Opening and reducing CRU file >> ' + fpath[1])

    # extract coordinate information
    lat = cru_data.variables["lat"][:]
    lon = cru_data.variables["lon"][:]

    # get information from the file name (according to the CRU publish standard)
    fname_list = str.split(fpath[1], '.')
    val_lab = fname_list[-3]
    data_var = cru_data.variables[val_lab]
    data_units = data_var.units

    # reduce the entire time-series down to one aggregate value
    data_value = determine_tseries_aggvalue(data_var, *args, **kwards)

    # dictionary to store extract meta data information from NC file
    meta = {'var':val_lab, 'units':data_units, \
                'start_time':int(fname_list[-5]), 'end_time':int(fname_list[-4])}

    # return a final dictionary of summarised information
    return {'lat':lat, 'lon':lon, 'value':data_value, 'meta':meta}

# --------------------------------------------------------------------------------

def find_CRU_files(loc):

    # given a folder location find all CRU NC files there
    cru_files = [(dp, f) for (dp, _, fn) in os.walk(loc) \
                    for f in fn if f.endswith('.nc')]

    return cru_files

# --------------------------------------------------------------------------------

def main():

    # Get the number of available cores for multi-proc
    num_cores = cpu_count()

    # get the file paths and names of all CRU files at the location
    cru_files = find_CRU_files(FILEPATH)

    # process each CRU file into a summarised dictionary
    f_label = lambda x: str.split(x, '.')
    cru_list = Parallel(n_jobs=num_cores)(delayed(extract_CRU_data) \
                (cfile, np.ma.average, np.ma.average) \
                for cfile in cru_files)

    cru_dict = {cdf['meta']['var']: cdf for cdf in cru_list}

    print("...")
    print("Pickled objects to be stored @:%s" % cru_files[0][0])

    # write to flat binary files for quick plotting later
    for (c_label, c_dict) in cru_dict.iteritems():
        new_name = '.'.join(f_label(cru_files[0][1])[:-3])
        pkl_fname = FILEPATH + new_name + '.' + c_label + ".100mean.pkl"

        print('Writing flat binary file for CRU tseries >> ' + c_label)
        pickle.dump(c_dict, open(pkl_fname, "wb"))

    return 1

if __name__ == "__main__":

    FILENAME = "cru_ts3.23.1901.2014.pre.dat.nc"
    FILEPATH = os.path.expanduser("~/Work/Research_Work/Climatologies/CRU/CRU_TS3/")

    main()

# SEQUENTIAL PROCESS (replace lines 87-93)
#    cru_dict = {f_label(cfile[1])[-3]: \
#                extract_CRU_data(cfile, np.ma.average, np.ma.average) \
#                for cfile in cru_files}


