import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats, optimize, interpolate
import itertools
import re #regular expression package

import netCDF4 # module that reads in .nc files (built on top of HDF5 format)
import pandas as pd
import geopandas as gpd
import xarray
import rioxarray
from tqdm import tqdm

from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon
from datetime import datetime, timedelta
from cftime import num2date, date2num, DatetimeGregorian
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.path as mpltPath

from shapely.geometry import mapping
from matplotlib.patches import Rectangle
from pyproj import CRS, Transformer # for transforming projected coordinates to elliptical coordinates
import cartopy.crs as ccrs # for defining and transforming coordinate systems
import cartopy.feature as cfeature # to add features to a cartopy map
import cartopy.io.shapereader as shpreader

# Specifying the paths for various data directories

data_dir= "../data"
pred_input_path= "/12km/"
resp_input_path= "/firelist/"

ecoregion_data= netCDF4.Dataset(data_dir + pred_input_path + "landcover/ecoregions/bailey_ecoprovince.nc", 'r')

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

def coord_transform(coord_a, coord_b, input_crs= 'WGS84', output_crs= 'EPSG:5070'):
    #function to convert coordinates between different reference systems with a little help from pyproj.Transformer
    #custom crs i/o with https://gis.stackexchange.com/questions/427786/pyproj-and-a-custom-crs
    
    crs_4326 = CRS("WGS84")
    crs_proj = CRS(output_crs)
    
    if input_crs != 'WGS84':
        transformer= Transformer.from_crs(crs_proj, crs_4326) #ESRI:54008
    else:
        transformer= Transformer.from_crs(crs_4326, crs_proj)
        
    # we add another if-else loop to account for differences in input size: for different sizes, we first construct a meshgrid,
    # before transforming coordinates. Thus, the output types will differ depending on the input.
    if len(coord_a) == len(coord_b):
        return transformer.transform(coord_a, coord_b) 
    else:
        coord_grid_a, coord_grid_b= np.meshgrid(coord_a, coord_b)
        return transformer.transform(coord_grid_a, coord_grid_b)
    
def ecoprovince_grid_mask(variable, region, plot= True):
    
    if region == 'central_south_coast':
        region_mask= ecoregion_data['bailey_ecoprovince'][3] + ecoregion_data['bailey_ecoprovince'][4]
    elif region == 'north_coast_sierra':
        region_mask= ecoregion_data['bailey_ecoprovince'][5] + ecoregion_data['bailey_ecoprovince'][18] 
    
    masked_variable= np.multiply(variable, region_mask)
    
    if plot:
        #for plotting; in case of data analysis, we will need to use np.nan_to_num()
        rows, cols= np.where(region_mask==0)    
        mask_plot_grid= np.ones(208*155)
        mask_plot_grid= np.reshape(mask_plot_grid, ((208, 155)))
        
        for i, j in np.array(list(zip(rows, cols))):
            mask_plot_grid[i, j]*= np.nan
            
        masked_variable_plot= np.multiply(masked_variable, mask_plot_grid)
        return masked_variable_plot
    
    else:
        return masked_variable
    
def bailey_ecoprovince_shp(region, lflag = 'L3', sbflag= True, coord= False):
    
    #reading in the shape file publcily available on the EPA website here: https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states
    #sbflag = False for plotting without state boundaries
    if lflag == 'L3':
        if not sbflag:
            ecoregionshp= gpd.read_file("../data/us_eco_l3/us_eco_l3.shp", crs="epsg:5070")
        else:
            ecoregionshp= gpd.read_file("../data/us_eco_l3_state_boundaries/us_eco_l3_state_boundaries.shp", crs="epsg:5070")
    elif lflag == 'L4':
        ecoregionshp= gpd.read_file("../data/us_eco_l4_state_boundaries/us_eco_l4.shp", crs="epsg:5070")
    
    if not sbflag:
        if region == "ca_south_coast":
            regshp= ecoregionshp[((ecoregionshp['US_L3CODE'] == '8')|(ecoregionshp['US_L3CODE'] == '85'))]
        elif region == "ca_cent_coast":
            regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '6')]
        elif region == "ca_sierra":
            regshp= ecoregionshp[((ecoregionshp['US_L3CODE'] == '4')|(ecoregionshp['US_L3CODE'] == '5')|(ecoregionshp['US_L3CODE'] == '9'))]
        elif region == "ca_north_coast":
            regshp= ecoregionshp[((ecoregionshp['US_L3CODE'] == '1')|(ecoregionshp['US_L3CODE'] == '78'))];
        elif region == "pnw_mts":
            regshp= ecoregionshp[((ecoregionshp['US_L3CODE'] == '1')| \
                (ecoregionshp['US_L3CODE'] == '4')|(ecoregionshp['US_L3CODE'] == '9')|(ecoregionshp['US_L3CODE'] == '77')|(ecoregionshp['US_L3CODE'] == '78'))]
        elif region == "northern_great_plains":
            regshp= ecoregionshp[((ecoregionshp['US_L3CODE'] == '42')|(ecoregionshp['US_L3CODE'] == '43'))]
    else: 
        if region == "ca_south_coast":
            regshp= ecoregionshp[(ecoregionshp['STATE_NAME'] == 'California')&
                     ((ecoregionshp['US_L3CODE'] == '8')|(ecoregionshp['US_L3CODE'] == '85'))]
        elif region == "ca_cent_coast":
            regshp= ecoregionshp[(ecoregionshp['STATE_NAME'] == 'California')&(ecoregionshp['US_L3CODE'] == '6')]
        elif region == "ca_sierra":
            regshp= ecoregionshp[(ecoregionshp['STATE_NAME'] == 'California')&
                     ((ecoregionshp['US_L3CODE'] == '4')|(ecoregionshp['US_L3CODE'] == '5')|(ecoregionshp['US_L3CODE'] == '9'))]
        elif region == "ca_north_coast":
            regshp= ecoregionshp[(ecoregionshp['STATE_NAME'] == 'California')&
                     ((ecoregionshp['US_L3CODE'] == '1')|(ecoregionshp['US_L3CODE'] == '78'))]
        elif region == "ca_total":
            regshp= ecoregionshp[(ecoregionshp['STATE_NAME'] == 'California')&
                     ((ecoregionshp['US_L3CODE'] == '1')|(ecoregionshp['US_L3CODE'] == '4')|(ecoregionshp['US_L3CODE'] == '5')\
                      |(ecoregionshp['US_L3CODE'] == '6')|(ecoregionshp['US_L3CODE'] == '8')|(ecoregionshp['US_L3CODE'] == '9')\
                      |(ecoregionshp['US_L3CODE'] == '78')|(ecoregionshp['US_L3CODE'] == '85'))];
        elif region == "pnw_mts":
            regshp= ecoregionshp[((ecoregionshp['STATE_NAME'] == 'Washington')|(ecoregionshp['STATE_NAME'] == 'Oregon'))&((ecoregionshp['US_L3CODE'] == '1')| \
                (ecoregionshp['US_L3CODE'] == '4')|(ecoregionshp['US_L3CODE'] == '9')|(ecoregionshp['US_L3CODE'] == '77')|(ecoregionshp['US_L3CODE'] == '78'))]
        elif region == "northern_great_plains":
            regshp= ecoregionshp[((ecoregionshp['STATE_NAME'] == 'Montana')|(ecoregionshp['STATE_NAME'] == 'Wyoming'))&\
                             ((ecoregionshp['US_L3CODE'] == '42')|(ecoregionshp['US_L3CODE'] == '43'))]
            
    if region == "columbia_plateau":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '10')]
    elif region == "northern_rockies":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '15')|(ecoregionshp['US_L3CODE'] == '41')]
    elif region == "middle_rockies":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '11')|(ecoregionshp['US_L3CODE'] == '16')|(ecoregionshp['US_L3CODE'] == '17')]
    elif region == "southern_rockies":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '19')|(ecoregionshp['US_L3CODE'] == '21')]
    elif region == "colorado_plateau":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '20')|(ecoregionshp['US_L3CODE'] == '22')]    
    elif region == "am_semidesert":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '14')|(ecoregionshp['US_L3CODE'] == '81')]
    elif region == "aznm_mts":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '23')|(ecoregionshp['US_L3CODE'] == '79')]
    elif region == "im_semidesert":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '12')|(ecoregionshp['US_L3CODE'] == '18')|(ecoregionshp['US_L3CODE'] == '80')]
    elif region == "im_desert":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '13')]
    elif region == "ch_desert":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '24')]
    elif region == "high_plains":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '25')]
    elif region == "sw_tablelands":
        regshp= ecoregionshp[(ecoregionshp['US_L3CODE'] == '26')]
    
    return regshp
    
def bailey_ecoprovince_mask(filepath, region, lflag= 'L3', l4indx= None):
    
    #inspired by the tutorial here: https://corteva.github.io/rioxarray/stable/examples/clip_geom.html
    #and https://gis.stackexchange.com/questions/354782/masking-netcdf-time-series-data-from-shapefile-using-python/354798#354798
    
    if type(region) == int:
        print("Enter the region name!")
    
    file= xarray.open_dataarray(filepath)
    file.rio.set_spatial_dims(x_dim="X", y_dim="Y", inplace=True)
    file.rio.write_crs("epsg:5070", inplace=True)
    
    regshp= bailey_ecoprovince_shp(region, lflag)
    
    if lflag == 'L3':
        clipped= file.rio.clip(regshp.geometry.apply(mapping), regshp.crs, drop=False)
    elif lflag == 'L4':
        clipped= file.rio.clip(regshp[regshp['US_L4CODE'] == l4indx].geometry.apply(mapping), regshp.crs, drop=False)
    return clipped

def reg_indx_func(region, firegdf, lflag = 'L3'):
    
    #inspiration: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    
    reg= bailey_ecoprovince_shp(region, lflag)
    pointInPolys= gpd.tools.sjoin(firegdf, reg, how='left') #note: the CRS of both data frames should match!!
    grouped= pointInPolys.groupby('index_right')
    
    if lflag == 'L3':
        loc_arr= np.sort(np.hstack([grouped.get_group(i).index for i in list(grouped.groups.keys())]))
        return loc_arr

    elif lflag == 'L4':
        loc_arr= np.hstack([grouped.get_group(i).index for i in list(grouped.groups.keys())])
        l4indx_arr= np.hstack([grouped.get_group(i)['US_L4CODE'] for i in list(grouped.groups.keys())])
        return loc_arr, l4indx_arr
        

def update_reg_indx(firegdf, lflag = 'L3'):
    
    #updates the region index for every point the wildfire frequency file
    
    regname= {1: "ca_sierra", 2: "ca_north_coast", 3: "ca_cent_coast", 4: "ca_south_coast", 5: "pnw_mts", 6: "columbia_plateau", 7:"northern_rockies", \
          8: "middle_rockies", 9: "southern_rockies", 10: "am_semidesert", 11: "aznm_mts", 12: "im_semidesert", 13: "im_desert", 14: "northern_great_plains", \
          15: "high_plains", 16: "colorado_plateau", 17: "sw_tablelands", 18: "ch_desert"}
    
    if lflag == 'L3':
        tmp_reg_indx_arr= np.zeros(len(firegdf), dtype= int)
        for i in tqdm(np.linspace(1, len(regname), len(regname), dtype= int)):
            tmplocarr= reg_indx_func(regname[i], firegdf, lflag)
            np.add.at(tmp_reg_indx_arr, tmplocarr, i)

        return tmp_reg_indx_arr
    
    elif lflag == 'L4':
        tmp_reg_indx_arr= np.empty(len(firegdf), dtype= 'U8')
        for i in np.linspace(1, len(regname), len(regname), dtype= int):
            tmplocarr, tmpl4indxarr= reg_indx_func(regname[i], firegdf, lflag)
            ind_arr= np.argsort(tmplocarr)
            np.add.at(tmp_reg_indx_arr, np.sort(tmplocarr), tmpl4indxarr[ind_arr])

        return tmp_reg_indx_arr


def ann_fire_freq(wildfiredf, regindx, final_year= 2018, start_year= 1984):
    
    #returns the annual fire frequency per region 
    
    n_years= final_year + 1 - start_year
    tmpgrp= wildfiredf[(wildfiredf['reg_indx'] == regindx) & (wildfiredf['final_year'] <= final_year)].groupby(['final_year'])
    tmpkeys= np.linspace(start_year, final_year, n_years, dtype= int)
    tmpfreq_arr= np.zeros(len(tmpkeys), dtype= int)
    
    for i, k in enumerate(tmpkeys):
        try:
            tmpfreq_arr[i]= len(tmpgrp.get_group(k))
        except KeyError:
            tmpfreq_arr[i]= 0
    
    return tmpfreq_arr


def mon_fire_freq(wildfiredf, regindx, start_year= 1984, final_year= 2019, threshold= False):
    
    #returns the monthly fire frequency per region 
    
    n_years= final_year + 1 - start_year
    if threshold:
        tmpgrp_1= wildfiredf[(wildfiredf['reg_indx'] == regindx) & (wildfiredf['final_year'] <= final_year) & (wildfiredf['final_area_ha'] > 405)].groupby(['final_year'])
    else:
        tmpgrp_1= wildfiredf[(wildfiredf['reg_indx'] == regindx) & (wildfiredf['final_year'] <= final_year)].groupby(['final_year'])
    tmpkeys_1= np.linspace(start_year, final_year, n_years, dtype= int)
    tmpkeys_2= np.linspace(1, 12, 12, dtype= int)
    tmpfreq_grid= np.zeros((len(tmpkeys_1), len(tmpkeys_2)), dtype= int)
    
    for i, k1 in enumerate(tmpkeys_1):
        for j, k2 in enumerate(tmpkeys_2):
            try:
                tmpgrp_2= tmpgrp_1.get_group(k1).groupby(['final_month'])                
                try:
                    tmpfreq_grid[i, j]= len(tmpgrp_2.get_group(k2)) #10/18 added the -1 to ensure indexing matches fire size index
                except KeyError:
                    tmpfreq_grid[i, j]= 0
            
            except KeyError:
                tmpfreq_grid[i, :]= 0
                j= j+1
    
    return tmpfreq_grid

def mon_fire_freq_2(wildfiredf, regindx, l4indx, start_year= 1984, final_year= 2019, threshold= False):
    
    #returns the monthly fire frequency per region 
    
    n_years= final_year + 1 - start_year
    if threshold:
        tmpgrp_1= wildfiredf[(wildfiredf['reg_indx'] == regindx) & (wildfiredf['L4_indx'] == l4indx) & (wildfiredf['final_year'] <= final_year) & (wildfiredf['final_area_ha'] > 405)].groupby(['final_year'])
    else:
        tmpgrp_1= wildfiredf[(wildfiredf['reg_indx'] == regindx) & (wildfiredf['L4_indx'] == l4indx) & (wildfiredf['final_year'] <= final_year)].groupby(['final_year'])
    tmpkeys_1= np.linspace(start_year, final_year, n_years, dtype= int)
    tmpkeys_2= np.linspace(1, 12, 12, dtype= int)
    tmpfreq_grid= np.zeros((len(tmpkeys_1), len(tmpkeys_2)), dtype= int)
    
    for i, k1 in enumerate(tmpkeys_1):
        for j, k2 in enumerate(tmpkeys_2):
            try:
                tmpgrp_2= tmpgrp_1.get_group(k1).groupby(['final_month'])                
                try:
                    tmpfreq_grid[i, j]= len(tmpgrp_2.get_group(k2)) #10/18 added the -1 to ensure indexing matches fire size index
                except KeyError:
                    tmpfreq_grid[i, j]= 0
            
            except KeyError:
                tmpfreq_grid[i, :]= 0
                j= j+1
    
    return tmpfreq_grid


def mon_burned_area(firefile, regindx, lflag= 'L3', l4indx= None, start_year= 1984, final_year= 2019):
    
    # returns the monthly burned area for the specified region
    
    regname= {1: "ca_sierra", 2: "ca_north_coast", 3: "ca_cent_coast", 4: "ca_south_coast", 5: "pnw_mts", 6: "columbia_plateau", 7:"northern_rockies", \
          8: "middle_rockies", 9: "southern_rockies", 10: "am_semidesert", 11: "aznm_mts", 12: "im_semidesert", 13: "im_desert", 14: "northern_great_plains", \
          15: "high_plains", 16: "colorado_plateau", 17: "sw_tablelands", 18: "ch_desert"}
    
    reg_burnarea= bailey_ecoprovince_mask(firefile, region= regname[regindx], lflag= lflag, l4indx= l4indx);
    stop_ind= (final_year + 1 - start_year)*12 - len(reg_burnarea) #returns a non-positive number
    
    if stop_ind:
        return np.sum(reg_burnarea, axis= (1, 2))[:stop_ind]
    else:
        return np.sum(reg_burnarea, axis= (1, 2)) #returns the array for the full time range
    

def tindx_func(startmon, duration, tim_size= 420, mov_avg= False):
    
    # returns the appropriate index arrays for various monthly ranges used in our analysis,
    #note: 0 --> January, 1 --> February and so on...
    
    if mov_avg:
        tindx_1= np.linspace(startmon, tim_size + (11 - duration) , tim_size, dtype= int)
        tindx_2= tindx_1 + duration
    else:
        tot_years= int(tim_size/12)
        tindx_1= np.linspace(startmon, tim_size - (12 - startmon), tot_years, dtype= int)
        tindx_2= tindx_1 + duration
                
    return tindx_1, tindx_2

def seas_burnarea(firefile, season, regindx, start_year= 1984, final_year= 2019):
    
    # returns the sum of burned areas for a given season and region
    
    tot_months= (final_year + 1 - start_year)*12
    if season == "annual":
        indx_1, indx_2= tindx_func(startmon= 0, duration= 12, tim_size= tot_months)
    elif season == "summer":
        indx_1, indx_2= tindx_func(startmon= 4, duration= 5, tim_size= tot_months)
    
    burnarea_arr= np.asarray([np.sum(mon_burned_area(firefile= firefile, regindx= regindx)[indx_1[i]:indx_2[i]]) for i in range(len(indx_1))])
    
    return burnarea_arr

def fire_tim_ind_func(filepath, start_year= 1984, final_year= 2019, antecedent= False, mov_avg= False):
    
    # returns the indices for climate predictor variables corresponding to the wildfire time series
    
    #solar_data= netCDF4.Dataset(data_dir + pred_input_path + "climate/primary/solar.nc", 'r')
    clim_data= netCDF4.Dataset(filepath, 'r')
    
    clim_times= clim_data['time']
    clim_dates= num2date(clim_times[:], units=clim_times.units)
    
    if antecedent:
        fire_tim_ind= (clim_dates.data > DatetimeGregorian(start_year - 3, 12, 15, 0, 0, 0, 0)) \
                                & (clim_dates.data < DatetimeGregorian(final_year, 1, 15, 0, 0, 0, 0))
    elif mov_avg:
        fire_tim_ind= (clim_dates.data > DatetimeGregorian(start_year - 2, 12, 15, 0, 0, 0, 0)) \
                                & (clim_dates.data < DatetimeGregorian(final_year + 1, 1, 15, 0, 0, 0, 0))
    else:
        fire_tim_ind= (clim_dates.data > DatetimeGregorian(start_year - 1, 12, 15, 0, 0, 0, 0)) \
                                & (clim_dates.data < DatetimeGregorian(final_year + 1, 1, 15, 0, 0, 0, 0))
        
    return fire_tim_ind

def human_var_interp(var_name, sav_flag= False):
    
    # Interpolating population and human density variables from the Silvis et al. data set
    
    tmpdf= xarray.open_dataarray("../data/12km/population/silvis/%s.nc"%var_name)
    popdf_1= tmpdf[0:1].copy(deep= True)
    for i in range(6):
        popdf_1= xarray.concat([popdf_1, tmpdf[0:1].copy().assign_coords(year= [1984+i])], dim= "year")
    popdf_1= xarray.concat([popdf_1[1:], popdf_1[0:1]], dim= "year")

    popdf_2= xarray.concat([tmpdf.interp(year=[1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]), tmpdf[1:2], \
                                tmpdf.interp(year=[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])], dim= "year")

    popdf_3= tmpdf[2:].copy(deep= True)
    for i in range(10):
        popdf_3= xarray.concat([popdf_3, tmpdf[2:].copy().assign_coords(year= [2011+i])], dim= "year")
        
    popdf_tot= xarray.concat([popdf_1, popdf_2, popdf_3], dim= "year")
    if sav_flag:
        popdf_tot.to_netcdf("../data/12km/population/%s_interp.nc"%var_name)
        print("Saved interpolated .nc file for %s"%var_name)
    else:
        return popdf_tot

def clim_pred_var(pred_file_indx, pred_seas_indx= None, regindx= None, lflag= 'L3', l4indx= None, tscale= "yearly", savg= True, start_year= 1984, final_year= 2019, burnarr_len= 0):
    
    # returns an array of climate predictor variable data indexed by season
    
    regname= {1: "ca_sierra", 2: "ca_north_coast", 3: "ca_cent_coast", 4: "ca_south_coast", 5: "pnw_mts", 6: "columbia_plateau", 7:"northern_rockies", \
          8: "middle_rockies", 9: "southern_rockies", 10: "am_semidesert", 11: "aznm_mts", 12: "im_semidesert", 13: "im_desert", 14: "northern_great_plains", \
          15: "high_plains", 16: "colorado_plateau", 17: "sw_tablelands", 18: "ch_desert", 19: "ca_total"}
    pred_flabel_arr= {1: ["climate/primary/tmax.nc"], 2: ["climate/primary/es.nc", "climate/primary/ea.nc"], 3: ["climate/primary/prec.nc"], \
                     4: ["climate/primary/prec.nc"], 5: ["climate/primary/ETo_co2.nc"], 6: ["landcover/nlcd/forest.nc"], 7: ["climate/primary/solar.nc"], \
                     8: ["climate/ucla_era5_wrf/windmax_max1.nc"], 9: ["topography/elev.nc"], 10: ["landcover/nlcd/grassland.nc"], \
                     11: ["landcover/fuel_winslow/study_regions/deadbiomass_litter.nc"], 12: ["landcover/fuel_winslow/study_regions/livebiomass_leaf.nc"], \
                     13: ["landcover/fuel_winslow/study_regions/connectivity.nc"], 14: ["climate/primary/rh.nc"], 15: ["climate/gridmet/fm1000.nc"], \
                     16: ["climate/era5/cape.nc"], 17: ["landcover/nlcd/urban.nc"], 18: ["climate/ucla_era5_wrf/ffwi_max3.nc"], 19: ["climate/primary/tmin.nc"], \
                     20: ["climate/ucla_era5_wrf/windmax_max3.nc"], 21: ["population/campdist.nc"], 22: ["population/campnum.nc"], 23: ["population/roaddist.nc"], \
                     24: ["climate/ucla_era5_wrf/vpd_max3.nc"], 25: ["climate/ucla_era5_wrf/vpd_max7.nc"], 26: ["climate/ucla_era5_wrf/tmax_max3.nc"], \
                     27: ["climate/ucla_era5_wrf/tmax_max7.nc"], 28: ["climate/ucla_era5_wrf/tmin_max3.nc"], 29: ["climate/ucla_era5_wrf/tmin_max7.nc"], \
                     30: ["climate/ucla_era5_wrf/ffwi_max7.nc"], 31: ["topography/slope.nc"], 32: ["topography/southness.nc"], 33: ["nsidc/swemean.nc"], 34: ["nsidc/swemax.nc"], \
                     35: ["climate/primary/tmax.nc", "climate/primary/tmin.nc"], 36: ["landcover/biomass_aboveground.nc"], 37: ["population/silvis/PopDenOf10perkm_distance_interp.nc"], \
                     38: ["population/silvis/house_density_interp.nc"], 39: ["lightning/strike_density_interp.nc"], 40: ["climate/ucla_era5_wrf/rh_min3.nc"], 41: ["landcover/nlcd/shrub.nc"]}
    pred_season_arr= {1: "warm", 2: "antecedent_lag1", 3: "annual", 4: "static", 5: "moving_average_3mo", 6: "moving_average_4mo", 7: "moving_average_2mo", \
                      8: "antecedent_lag2", 9: "antecedent_avg"} 
                      #be careful about indexing since this is correlated with input for multivariate regression 

    if len(pred_flabel_arr[pred_file_indx]) > 1:
        pred_file= data_dir + pred_input_path + pred_flabel_arr[pred_file_indx][0]
        pred_file_add= data_dir + pred_input_path + pred_flabel_arr[pred_file_indx][1]
        if regindx == None:
            pred_data= xarray.open_dataarray(pred_file) - xarray.open_dataarray(pred_file_add)
        else:
            pred_data= bailey_ecoprovince_mask(pred_file, region= regname[regindx], lflag= lflag, l4indx= l4indx) - bailey_ecoprovince_mask(pred_file_add, region= regname[regindx], lflag= lflag, l4indx= l4indx);
    else:
        pred_file= data_dir + pred_input_path + pred_flabel_arr[pred_file_indx][0]
        if regindx == None:
            pred_data= xarray.open_dataarray(pred_file)
        else:
            pred_data= bailey_ecoprovince_mask(pred_file, region= regname[regindx], lflag= lflag, l4indx= l4indx);
     
    tot_months= (final_year + 1 - start_year)*12
    
    if tscale == "yearly":
        pred_season= pred_season_arr[pred_seas_indx] 
        if pred_season == "warm":
            fire_tim_ind= fire_tim_ind_func(pred_file, start_year, final_year) # aligns the climate array indexing to the fire array's
            seas_indx_1, seas_indx_2= tindx_func(startmon= 2, duration= 8, tim_size= tot_months)
            pred_season_data= np.asarray([np.mean(pred_data[fire_tim_ind][seas_indx_1[i]:seas_indx_2[i]]).values for i in range(len(seas_indx_1))])
        elif pred_season == "antecedent":
            fire_tim_ind_ant= fire_tim_ind_func(pred_file, start_year, final_year, antecedent= True)
            seas_indx_1, seas_indx_2= tindx_func(startmon= 2, duration= 20, tim_size= tot_months)
            pred_season_data= np.asarray([np.mean(pred_data[fire_tim_ind_ant][seas_indx_1[i]:seas_indx_2[i]]).values for i in range(len(seas_indx_1))])
        elif pred_season == "annual":
            stop_ind= burnarr_len - len(pred_data)  #note: should return a negative number
            if stop_ind < 0:
                pred_season_data= np.mean(pred_data[:stop_ind], axis= (1, 2)).values
            else:
                pred_season_data= np.mean(pred_data, axis= (1, 2)).values
        
        return pred_season_data
    
    elif tscale == "monthly":
        pred_season= pred_season_arr[pred_seas_indx]
        if pred_season == "warm": #replace warm with fire month 
            fire_tim_ind= fire_tim_ind_func(pred_file, start_year, final_year)
            if savg: #savg = True ==> spatial average for fire frequency 
                return np.mean(pred_data[fire_tim_ind], axis= (1, 2)).values
            else:
                return pred_data[fire_tim_ind].values
            
        elif pred_season == "moving_average_2mo":
            fire_tim_ind_mavg= fire_tim_ind_func(pred_file, start_year, final_year, mov_avg= True)
            seas_indx_1, seas_indx_2= tindx_func(startmon= 10, duration= 2, tim_size= tot_months, mov_avg= True)
            if savg:
                return np.asarray([np.mean(np.mean(pred_data[fire_tim_ind_mavg][seas_indx_1[i]:seas_indx_2[i]], axis= (1, 2)), axis= 0) for i in range(len(seas_indx_1))])
            
            else:
                return np.asarray([np.mean(pred_data[fire_tim_ind_mavg][seas_indx_1[i]:seas_indx_2[i]], axis= 0) for i in range(len(seas_indx_1))])
        
        elif pred_season == "moving_average_3mo":
            fire_tim_ind_mavg= fire_tim_ind_func(pred_file, start_year, final_year, mov_avg= True)
            seas_indx_1, seas_indx_2= tindx_func(startmon= 9, duration= 3, tim_size= tot_months, mov_avg= True)
            if savg:
                return np.asarray([np.mean(np.mean(pred_data[fire_tim_ind_mavg][seas_indx_1[i]:seas_indx_2[i]], axis= (1, 2)), axis= 0) for i in range(len(seas_indx_1))])
            else:
                return np.asarray([np.mean(pred_data[fire_tim_ind_mavg][seas_indx_1[i]:seas_indx_2[i]], axis= 0) for i in range(len(seas_indx_1))])
        
        elif pred_season == "moving_average_4mo":
            fire_tim_ind_mavg= fire_tim_ind_func(pred_file, start_year, final_year, mov_avg= True)
            seas_indx_1, seas_indx_2= tindx_func(startmon= 8, duration= 4, tim_size= tot_months, mov_avg= True)
            if savg:
                return np.asarray([np.mean(np.mean(pred_data[fire_tim_ind_mavg][seas_indx_1[i]:seas_indx_2[i]], axis= (1, 2)), axis= 0) for i in range(len(seas_indx_1))])
            else:
                return np.asarray([np.mean(pred_data[fire_tim_ind_mavg][seas_indx_1[i]:seas_indx_2[i]], axis= 0) for i in range(len(seas_indx_1))])
            
        elif pred_season == "antecedent_avg":
            fire_tim_ind_ant= fire_tim_ind_func(pred_file, start_year, final_year, antecedent= True)
            seas_indx_1, seas_indx_2= tindx_func(startmon= 2, duration= 20, tim_size= tot_months) #based on Park's 2019 paper "best" correlation
            if savg:
                pred_data= np.asarray([np.mean(pred_data[fire_tim_ind_ant][seas_indx_1[i]:seas_indx_2[i]], axis=(1, 2)).values for i in range(len(seas_indx_1))])
                return np.repeat(np.mean(pred_data, axis= 1), 12) # assumption: antecedent precipitation is the same for all fire months
            else:
                pred_data= np.asarray([np.mean(pred_data[fire_tim_ind_ant][seas_indx_1[i]:seas_indx_2[i]], axis= 0) for i in range(len(seas_indx_1))])
                return np.kron(pred_data, np.ones((12, 1, 1))) # assumption: antecedent precipitation is the same for all fire months
        
        elif pred_season == "antecedent_lag2":
            fire_tim_ind_ant= fire_tim_ind_func(pred_file, start_year, final_year, antecedent= True)
            seas_indx_1, seas_indx_2= tindx_func(startmon= 0, duration= 12, tim_size= tot_months)
            if savg:
                pred_data= np.asarray([np.mean(pred_data[fire_tim_ind_ant][seas_indx_1[i]:seas_indx_2[i]], axis=(1, 2)).values for i in range(len(seas_indx_1))])
                return np.repeat(np.mean(pred_data, axis= 1), 12) # assumption: antecedent precipitation is the same for all fire months
            else:
                pred_data= np.asarray([np.mean(pred_data[fire_tim_ind_ant][seas_indx_1[i]:seas_indx_2[i]], axis= 0) for i in range(len(seas_indx_1))])
                return np.kron(pred_data, np.ones((12, 1, 1))) # assumption: antecedent precipitation is the same for all fire months
        
        elif pred_season == "antecedent_lag1":
            fire_tim_ind_ant= fire_tim_ind_func(pred_file, start_year, final_year, antecedent= True)
            seas_indx_1, seas_indx_2= tindx_func(startmon= 12, duration= 12, tim_size= tot_months)
            if savg:
                pred_data= np.asarray([np.mean(pred_data[fire_tim_ind_ant][seas_indx_1[i]:seas_indx_2[i]], axis=(1, 2)).values for i in range(len(seas_indx_1))])
                return np.repeat(np.mean(pred_data, axis= 1), 12) # assumption: antecedent precipitation is the same for all fire months
            else:
                pred_data= np.asarray([np.mean(pred_data[fire_tim_ind_ant][seas_indx_1[i]:seas_indx_2[i]], axis= 0) for i in range(len(seas_indx_1))])
                return np.kron(pred_data, np.ones((12, 1, 1))) # assumption: antecedent precipitation is the same for all fire months
            
        elif pred_season == "annual":
            if savg:
                pred_data= np.mean(pred_data, axis= (1, 2)).values
                return np.repeat(pred_data, 12)[0:tot_months]
            else:
                pred_data= np.kron(pred_data, np.ones((12, 1, 1)))
                return pred_data[0:tot_months]
        elif pred_season == "static":
            if savg:
                pred_data= np.tile(pred_data, (tot_months, 1, 1)) #technically, this doesn't make sense; I have only used this option to create a monthly time series for cross-sectional purposes
                return np.nanmean(pred_data, axis= (1, 2))
            else:
                return pred_data.values


def init_fire_freq_df(firedf, regindx= None, lflag= 'L3', start_year= 1984, final_year= 2019): 
    
    #constructs the input dataframe for a NN-based likelihood model of fire frequency
    
    regname= {1: "ca_sierra", 2: "ca_north_coast", 3: "ca_cent_coast", 4: "ca_south_coast", 5: "pnw_mts", 6: "columbia_plateau", 7:"northern_rockies", \
          8: "middle_rockies", 9: "southern_rockies", 10: "am_semidesert", 11: "aznm_mts", 12: "im_semidesert", 13: "im_desert", 14: "northern_great_plains", \
          15: "high_plains", 16: "colorado_plateau", 17: "sw_tablelands", 18: "ch_desert"}
    
    tot_months= (final_year + 1 - start_year)*12
    
    if lflag == 'L3':
        reg_df= pd.DataFrame({'Tmax': [], 'VPD': [], 'Prec': [], 'Antprec_lag1': [], 'Forest': [], 'Solar': [], 'Wind': [], 'Grassland': [], 'Elev': [], \
                          'RH': [], 'FM1000': [], 'Ant_Tmax': [], 'AvgVPD_3mo': [], 'Avgprec_3mo': [], 'Ant_RH': [], 'CAPE': [], 'Urban': [], 'FFWI_max3': [], \
                          'FFWI_max7': [], 'Tmin': [], 'Camp_dist': [], 'Camp_num': [], 'Road_dist': [], 'Antprec_lag2': [], 'Avgprec_4mo': [], 'Avgprec_2mo': [],\
                          'VPD_max3': [], 'VPD_max7': [], 'Tmax_max3': [], 'Tmax_max7': [], 'Slope': [], 'Southness': [], 'AvgVPD_4mo': [], 'AvgVPD_2mo': [], \
                          'SWE_mean': [], 'SWE_max': [], 'AntSWE_3mo': [], 'Delta_T': [], 'Biomass': [], 'Popdensity': [], 'Housedensity': [], 'Lightning': [], \
                          'fire_freq': pd.Series(dtype= 'int'), 'month': pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int')})
    
        for r in (1 + np.arange(len(regname), dtype= int)):
            print("Creating dataframe for %s"%regname[r])
            
            #Climate variables 
            reg_tmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_vpd_mean= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_prec= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_antprec_1= clim_pred_var(pred_file_indx= 4, pred_seas_indx= 2, regindx= r, tscale= "monthly", savg= True)
            reg_antprec_2= clim_pred_var(pred_file_indx= 4, pred_seas_indx= 8, regindx= r, tscale= "monthly", savg= True)
            reg_solar= clim_pred_var(pred_file_indx= 7, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_wind= clim_pred_var(pred_file_indx= 8, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_rh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_fm1000= clim_pred_var(pred_file_indx= 15, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_anttmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_avgvpd_3mo= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_avgvpd_4mo= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 6, regindx= r, tscale= "monthly", savg= True)
            reg_avgvpd_2mo= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 7, regindx= r, tscale= "monthly", savg= True)
            reg_avgprec_3mo= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_avgprec_4mo= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 6, regindx= r, tscale= "monthly", savg= True)
            reg_avgprec_2mo= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 7, regindx= r, tscale= "monthly", savg= True)
            reg_antrh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_cape= clim_pred_var(pred_file_indx= 16, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            #reg_capexp= np.multiply(reg_cape, reg_prec)*10**-5/24 # lightning flash rate in km^-2 hr^-1
            reg_tmin= clim_pred_var(pred_file_indx= 19, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            
            #Extreme weather correlates
            reg_ffwi_max3= clim_pred_var(pred_file_indx= 18, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_ffwi_max7= clim_pred_var(pred_file_indx= 30, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_vpd_max3= clim_pred_var(pred_file_indx= 24, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_vpd_max7= clim_pred_var(pred_file_indx= 25, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_tmax_max3= clim_pred_var(pred_file_indx= 26, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_tmax_max7= clim_pred_var(pred_file_indx= 27, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            
            #Landcover variables 
            reg_forest= clim_pred_var(pred_file_indx= 6, pred_seas_indx= 3, regindx= r, tscale= "monthly", savg= True)
            reg_grass= clim_pred_var(pred_file_indx= 10, pred_seas_indx= 3, regindx= r, tscale= "monthly", savg= True)
            reg_urban= clim_pred_var(pred_file_indx= 17, pred_seas_indx= 3, regindx= r, tscale= "monthly", savg= True)
            
            #Topographic variables 
            reg_elev= clim_pred_var(pred_file_indx= 9, pred_seas_indx= 4, regindx= r, tscale= "monthly", savg= True)
            reg_slope= clim_pred_var(pred_file_indx= 31, pred_seas_indx= 4, regindx= r, tscale= "monthly", savg= True)
            reg_southness= clim_pred_var(pred_file_indx= 32, pred_seas_indx= 4, regindx= r, tscale= "monthly", savg= True)
            reg_campdist= clim_pred_var(pred_file_indx= 21, pred_seas_indx= 4, regindx= r, tscale= "monthly", savg= True)
            reg_campnum= clim_pred_var(pred_file_indx= 22, pred_seas_indx= 4, regindx= r, tscale= "monthly", savg= True)
            reg_roaddist= clim_pred_var(pred_file_indx= 23, pred_seas_indx= 4, regindx= r, tscale= "monthly", savg= True)

            reg_fire_freq= mon_fire_freq(wildfiredf= firedf, regindx= r, start_year= start_year, final_year= final_year).flatten()
            month_arr= np.linspace(0, tot_months - 1, tot_months, dtype= int)

            reg_df= reg_df.append(pd.DataFrame({'Tmax': reg_tmax, 'VPD': reg_vpd_mean, 'Prec': reg_prec, 'Antprec_lag1': reg_antprec_1, 'Forest': reg_forest, \
                'Solar': reg_solar, 'Wind': reg_wind, 'Grassland': reg_grass, 'Elev': reg_elev, 'RH': reg_rh, 'FM1000': reg_fm1000, 'Ant_Tmax': reg_anttmax, \
                'AvgVPD_3mo': reg_avgvpd_3mo, 'Avgprec_3mo': reg_avgprec_3mo, 'Ant_RH': reg_antrh, 'CAPE': reg_cape, 'Urban': reg_urban, 'FFWI_max3': reg_ffwi_max3, \
                'FFWI_max7': reg_ffwi_max7, 'Tmin': reg_tmin, 'Camp_dist': reg_campdist, 'Camp_num': reg_campnum, 'Road_dist': reg_roaddist, \
                'Antprec_lag2': reg_antprec_2, 'Avgprec_4mo': reg_avgprec_4mo, 'Avgprec_2mo': reg_avgprec_2mo, 'VPD_max3': reg_vpd_max3, 'VPD_max7': reg_vpd_max7, \
                'Tmax_max3': reg_tmax_max3, 'Tmax_max7': reg_tmax_max7, 'Slope': reg_slope, 'Southness': reg_southness, 'AvgVPD_2mo': reg_avgvpd_2mo, \
                'AvgVPD_4mo': reg_avgvpd_4mo, 'fire_freq': reg_fire_freq, 'month': month_arr, 'reg_indx': r*np.ones(tot_months, dtype= int)}), ignore_index=True)
    
    elif lflag == 'L4':
        reg_df= reg_df= pd.DataFrame({'Tmax': [], 'VPD': [], 'Prec': [], 'Antprec_lag1': [], 'Forest': [], 'Solar': [], 'Wind': [], 'Grassland': [], 'Elev': [], \
                        'RH': [], 'FM1000': [], 'Ant_Tmax': [], 'AvgVPD_3mo': [], 'Avgprec_3mo': [], 'Ant_RH': [], 'CAPE': [], 'Urban': [], 'FFWI_max3': [], \
                        'FFWI_max7': [], 'Tmin': [], 'Camp_dist': [], 'Camp_num': [], 'Road_dist': [], 'Antprec_lag2': [], 'Avgprec_4mo': [], 'Avgprec_2mo': [],\
                        'VPD_max3': [], 'VPD_max7': [], 'Tmax_max3': [], 'Tmax_max7': [], 'Slope': [], 'Southness': [], 'AvgVPD_4mo': [], 'AvgVPD_2mo': [], \
                        'fire_freq': pd.Series(dtype= 'int'), 'month': pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int'), 'l4_indx': pd.Series(dtype='U8')})
    
        print("Creating dataframe for %s"%regname[regindx])
        regshp= bailey_ecoprovince_shp(region= regname[regindx], lflag= 'L4')
        l4regs= regshp['US_L4CODE'].unique()

        for l in l4regs:
            #Climate variables 
            reg_tmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_vpd_mean= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_prec= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_antprec_1= clim_pred_var(pred_file_indx= 4, pred_seas_indx= 2, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_antprec_2= clim_pred_var(pred_file_indx= 4, pred_seas_indx= 8, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_solar= clim_pred_var(pred_file_indx= 7, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_wind= clim_pred_var(pred_file_indx= 8, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_rh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_fm1000= clim_pred_var(pred_file_indx= 15, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_anttmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_avgvpd_3mo= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_avgvpd_4mo= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 6, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_avgvpd_2mo= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 7, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_avgprec_3mo= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_avgprec_4mo= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 6, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_avgprec_2mo= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 7, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_antrh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_cape= clim_pred_var(pred_file_indx= 16, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            #reg_capexp= np.multiply(reg_cape, reg_prec)*10**-5/24 # lightning flash rate in km^-2 hr^-1
            reg_tmin= clim_pred_var(pred_file_indx= 19, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            
            #Extreme weather correlates
            reg_ffwi_max3= clim_pred_var(pred_file_indx= 18, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_ffwi_max7= clim_pred_var(pred_file_indx= 30, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_vpd_max3= clim_pred_var(pred_file_indx= 24, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_vpd_max7= clim_pred_var(pred_file_indx= 25, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_tmax_max3= clim_pred_var(pred_file_indx= 26, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_tmax_max7= clim_pred_var(pred_file_indx= 27, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            
            #Landcover variables 
            reg_forest= clim_pred_var(pred_file_indx= 6, pred_seas_indx= 3, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_grass= clim_pred_var(pred_file_indx= 10, pred_seas_indx= 3, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_urban= clim_pred_var(pred_file_indx= 17, pred_seas_indx= 3, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            
            #Topographic variables 
            reg_elev= clim_pred_var(pred_file_indx= 9, pred_seas_indx= 4, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_slope= clim_pred_var(pred_file_indx= 31, pred_seas_indx= 4, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_southness= clim_pred_var(pred_file_indx= 32, pred_seas_indx= 4, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_campdist= clim_pred_var(pred_file_indx= 21, pred_seas_indx= 4, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_campnum= clim_pred_var(pred_file_indx= 22, pred_seas_indx= 4, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)
            reg_roaddist= clim_pred_var(pred_file_indx= 23, pred_seas_indx= 4, regindx= regindx, lflag= 'L4', l4indx= l, tscale= "monthly", savg= True)

            reg_fire_freq= mon_fire_freq_2(wildfiredf= firedf, regindx= regindx, l4indx= l, start_year= start_year, final_year= final_year).flatten()
            month_arr= np.linspace(0, tot_months - 1, tot_months, dtype= int)

            reg_df= reg_df.append(pd.DataFrame({'Tmax': reg_tmax, 'VPD': reg_vpd_mean, 'Prec': reg_prec, 'Antprec_lag1': reg_antprec_1, 'Forest': reg_forest, \
                'Solar': reg_solar, 'Wind': reg_wind, 'Grassland': reg_grass, 'Elev': reg_elev, 'RH': reg_rh, 'FM1000': reg_fm1000, 'Ant_Tmax': reg_anttmax, \
                'AvgVPD_3mo': reg_avgvpd_3mo, 'Avgprec_3mo': reg_avgprec_3mo, 'Ant_RH': reg_antrh, 'CAPE': reg_cape, 'Urban': reg_urban, 'FFWI_max3': reg_ffwi_max3, \
                'FFWI_max7': reg_ffwi_max7, 'Tmin': reg_tmin, 'Camp_dist': reg_campdist, 'Camp_num': reg_campnum, 'Road_dist': reg_roaddist, \
                'Antprec_lag2': reg_antprec_2, 'Avgprec_4mo': reg_avgprec_4mo, 'Avgprec_2mo': reg_avgprec_2mo, 'VPD_max3': reg_vpd_max3, 'VPD_max7': reg_vpd_max7, \
                'Tmax_max3': reg_tmax_max3, 'Tmax_max7': reg_tmax_max7, 'Slope': reg_slope, 'Southness': reg_southness, 'AvgVPD_2mo': reg_avgvpd_2mo, \
                'AvgVPD_4mo': reg_avgvpd_4mo, 'fire_freq': reg_fire_freq, 'month': month_arr, 'reg_indx': regindx*np.ones(tot_months, dtype= int), \
                'l4_indx': np.repeat(l, tot_months).astype('U8')}), ignore_index=True)
    
    return reg_df #, reg_fire_ind

def init_fire_size_df(firefile, regindx= None, lflag= 'L3', start_year= 1984, final_year= 2019): 
    
    #constructs the input dataframe for a NN-based likelihood model of fire frequency
    
    regname= {1: "ca_sierra", 2: "ca_north_coast", 3: "ca_cent_coast", 4: "ca_south_coast", 5: "pnw_mts", 6: "columbia_plateau", 7:"northern_rockies", \
          8: "middle_rockies", 9: "southern_rockies", 10: "am_semidesert", 11: "aznm_mts", 12: "im_semidesert", 13: "im_desert", 14: "northern_great_plains", \
          15: "high_plains", 16: "colorado_plateau", 17: "sw_tablelands", 18: "ch_desert"}
    
    tot_months= (final_year + 1 - start_year)*12
    
    if lflag == 'L3':
        reg_df= pd.DataFrame({'Tmax': [], 'VPD': [], 'Prec': [], 'Antprec': [], 'Forest': [], 'Solar': [], 'Wind': [], 'Grassland': [], \
                          'RH': [], 'FM1000': [], 'Ant_Tmax': [], 'Ant_VPD': [], 'Avgprec': [], 'Ant_RH': [], 'Urban': [], 'FFWI': [], 'Tmin': [],  'fire_size': [], \
                          'month': pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int')})
    
        for r in (1 + np.arange(len(regname), dtype= int)):
            print("Creating dataframe for %s"%regname[r])
            reg_tmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_vpd= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_prec= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_antprec= clim_pred_var(pred_file_indx= 4, pred_seas_indx= 2, regindx= r, tscale= "monthly", savg= True)
            reg_forest= clim_pred_var(pred_file_indx= 6, pred_seas_indx= 3, regindx= r, tscale= "monthly", savg= True)
            reg_solar= clim_pred_var(pred_file_indx= 7, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_wind= clim_pred_var(pred_file_indx= 20, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_grass= clim_pred_var(pred_file_indx= 10, pred_seas_indx= 3, regindx= r, tscale= "monthly", savg= True)
            reg_rh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_fm1000= clim_pred_var(pred_file_indx= 15, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_anttmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_antvpd= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_avgprec= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_antrh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 5, regindx= r, tscale= "monthly", savg= True)
            reg_urban= clim_pred_var(pred_file_indx= 17, pred_seas_indx= 3, regindx= r, tscale= "monthly", savg= True)
            reg_ffwi= clim_pred_var(pred_file_indx= 18, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)
            reg_tmin= clim_pred_var(pred_file_indx= 19, pred_seas_indx= 1, regindx= r, tscale= "monthly", savg= True)


            reg_fire_size= mon_burned_area(firefile, regindx= r, lflag= lflag, start_year= start_year, final_year= final_year).values
            month_arr= np.linspace(0, tot_months - 1, tot_months, dtype= int)

            reg_df= reg_df.append(pd.DataFrame({'Tmax': reg_tmax, 'VPD': reg_vpd, 'Prec': reg_prec, 'Antprec': reg_antprec, 'Forest': reg_forest, 'Solar': reg_solar, \
                        'Wind': reg_wind, 'Grassland': reg_grass, 'RH': reg_rh, 'FM1000': reg_fm1000, \
                        'Ant_Tmax': reg_anttmax, 'Ant_VPD': reg_antvpd, 'Avgprec': reg_avgprec, 'Ant_RH': reg_antrh, 'Urban': reg_urban, \
                        'FFWI': reg_ffwi, 'Tmin': reg_tmin, 'fire_size': reg_fire_size, 'month': month_arr, 'reg_indx': r*np.ones(tot_months, dtype= int)}), ignore_index=True)
    
    elif lflag == 'L4':
        reg_df= pd.DataFrame({'Tmax': [], 'VPD': [], 'Prec': [], 'Antprec': [], 'Forest': [], 'Solar': [], 'Wind': [], 'Grassland': [], \
                          'RH': [], 'FM1000': [], 'Ant_Tmax': [], 'Ant_VPD': [], 'Avgprec': [], 'Ant_RH': [], 'Urban': [], 'FFWI': [], 'Tmin': [], 'fire_size': [], \
                          'month': pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int'), 'l4_indx': pd.Series(dtype= 'U8')})
    
        print("Creating dataframe for %s"%regname[regindx])
        regshp= bailey_ecoprovince_shp(region= regname[regindx], lflag= 'L4')
        l4regs= regshp['US_L4CODE'].unique()

        for l in l4regs:
            reg_tmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_vpd= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_prec= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_antprec= clim_pred_var(pred_file_indx= 4, pred_seas_indx= 2, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_forest= clim_pred_var(pred_file_indx= 6, pred_seas_indx= 3, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_solar= clim_pred_var(pred_file_indx= 7, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_wind= clim_pred_var(pred_file_indx= 20, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_grass= clim_pred_var(pred_file_indx= 10, pred_seas_indx= 3, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_rh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_fm1000= clim_pred_var(pred_file_indx= 15, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_anttmax= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_antvpd= clim_pred_var(pred_file_indx= 2, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_avgprec= clim_pred_var(pred_file_indx= 3, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_antrh= clim_pred_var(pred_file_indx= 14, pred_seas_indx= 5, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_urban= clim_pred_var(pred_file_indx= 17, pred_seas_indx= 3, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_ffwi= clim_pred_var(pred_file_indx= 18, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)
            reg_tmin= clim_pred_var(pred_file_indx= 19, pred_seas_indx= 1, regindx= regindx, lflag= 'L4', l4indx =l, tscale= "monthly", savg= True)

            reg_fire_size= mon_burned_area(firefile, regindx= regindx, lflag= lflag, l4indx= l, start_year= start_year, final_year= final_year).values
            month_arr= np.linspace(0, tot_months - 1, tot_months, dtype= int)

            reg_df= reg_df.append(pd.DataFrame({'Tmax': reg_tmax, 'VPD': reg_vpd, 'Prec': reg_prec, 'Antprec': reg_antprec, 'Forest': reg_forest, 'Solar': reg_solar, 'Wind': reg_wind, \
                        'Grassland': reg_grass, 'RH': reg_rh, 'FM1000': reg_fm1000, 'Ant_Tmax': reg_anttmax, 'Ant_VPD': reg_antvpd, 'Avgprec': reg_avgprec, 'Ant_RH': reg_antrh, \
                        'Urban': reg_urban, 'FFWI': reg_ffwi, 'Tmin': reg_tmin, 'fire_size': reg_fire_size, 'month': month_arr, 'reg_indx': regindx*np.ones(tot_months, dtype= int), \
                        'l4_indx': np.repeat(l, tot_months).astype('U8')}), ignore_index=True)
    
    return reg_df #, reg_fire_ind

def init_grid(firedat, res, tot_months):
    
    #initializes a raster grid based on Park's climate/fire data
    
    if res == '24km':
        xmin, xmax= [firedat[:, 1:-1:2, :-3:2].X.values.min(), firedat[:, 1:-1:2, :-3:2].X.values.max()]
        ymin, ymax= [firedat[:, 1:-1:2, :-3:2].Y.values.min(), firedat[:, 1:-1:2, :-3:2].Y.values.max()]
        cellwidth= abs(firedat.X[0].values - firedat.X[2].values)
        new_firedat= (firedat[0:tot_months, 1:-1:2, :-1:2] - firedat[0:tot_months, 1:-1:2, :-1:2])
    elif res == '12km':
        xmin, xmax= [firedat[:, :, :].X.values.min(), firedat[:, :, :].X.values.max()]
        ymin, ymax= [firedat[:, :, :].Y.values.min(), firedat[:, :, :].Y.values.max()]
        cellwidth= abs(firedat.X[0].values - firedat.X[1].values)
        new_firedat= (firedat[0:tot_months, :, :] - firedat[0:tot_months, :, :])
    elif res == '1km':
        xmin, xmax= [firedat[:, :, :].X.values.min(), firedat[:, :, :].X.values.max()]
        ymin, ymax= [firedat[:, :, :].Y.values.min(), firedat[:, :, :].Y.values.max()]
        cellwidth= abs(firedat.X[0].values - firedat.X[1].values)
        new_firedat= (firedat[0:tot_months, :, :] - firedat[0:tot_months, :, :])
    
    cols= list(np.arange(xmin, xmax + cellwidth, cellwidth))
    rows= list(np.arange(ymax, ymin - cellwidth, -cellwidth))
    
    polygons = []
    for y in rows:
        for x in cols:
            polygons.append(Polygon([(x,y), (x+cellwidth, y), (x+cellwidth, y+cellwidth), (x, y+cellwidth)])) 

    grid= gpd.GeoDataFrame({'geometry': gpd.GeoSeries(polygons)})
    grid['grid_indx']= grid.index.to_numpy()
    grid= grid.set_crs('EPSG:5070')
    
    return grid, rows, cols

#def init_clim_pred_grid():
    
        
def init_fire_alloc_gdf(firedat, firegdf, res= '24km', start_year= 1984, final_year= 2019, fire_grid= False): 
    
    # function to allocate individual fires from the firelist.txt file to a raster grid of varying resolutions. This serves two roles: 1) allows the predictions of fire probability for individual grid cells;
    # 2) enables the calculation of a (weighted) average for climate variables for each fire.
    
    tot_months= (final_year + 1 - start_year)*12
    grid, rows, cols= init_grid(firedat, res, tot_months)
    cellwidth= int(re.findall(r'\d+', res)[0])*1000
    if res == '24km':
        new_firedat= (firedat[0:tot_months, 1:-1:2, :-1:2] - firedat[0:tot_months, 1:-1:2, :-1:2])
    else:
        new_firedat= (firedat[0:tot_months, :, :] - firedat[0:tot_months, :, :])

    print("Constructed a raster grid with %s grid cell size"%res);
    
    firepts= gpd.GeoSeries(firegdf['geometry'].buffer(np.sqrt(firegdf['final_area_ha']*1e4/np.pi))) #currently buffer is a circle with radius = sqrt(A/pi) [in m]
    firepts_gdf= gpd.GeoDataFrame({'geometry': firepts, 'fire_indx': firegdf.index.to_numpy(), \
                               'fire_month': (firegdf['final_month'] - 1) + (firegdf['final_year'] - 1984)*12, \
                               'fire_size': firegdf['final_area_ha']*1e4, 'reg_indx': firegdf['reg_indx'], \
                               'L4_indx': firegdf['L4_indx']})
                               
    
    print("Created a GeoDataFrame of all fires");
    
    merged= gpd.overlay(firepts_gdf, grid, how= 'intersection')
    merged= merged.sort_values(by= ['fire_indx']).reset_index()
    merged= merged.drop('index', axis= 1)
    merged= merged[merged['fire_month'] <= (tot_months - 1)]

    coord_arr= np.array(list(itertools.product(np.linspace(0, len(rows) - 1, len(rows), dtype= int), np.linspace(0, len(cols) - 1, len(cols), dtype= int))))
    areagroups= merged.groupby('fire_indx')
    gridfracarr= np.hstack([((areagroups.get_group(k).area/cellwidth**2)/np.linalg.norm(areagroups.get_group(k).area/cellwidth**2, 1)).to_numpy() \
                                                                                                                 for k in areagroups.groups.keys()])
    rastercoords= [np.insert(coord_arr[merged['grid_indx'].loc[[ind]]], 0, merged['fire_month'].loc[[ind]]) for ind in merged.index]
    
    merged['cell_frac']= gridfracarr
    merged['raster_coords']= rastercoords
    merged['grid_y']= [coord_arr[merged['grid_indx'].loc[[ind]]][0][0] for ind in merged.index]
    merged['grid_x']= [coord_arr[merged['grid_indx'].loc[[ind]]][0][1] for ind in merged.index]
    
    print("Overlayed the fire points on the raster grid to obtain cell fraction for each fire");
    
    if fire_grid == True:
        for m in merged.index.to_numpy():
            rc= merged['raster_coords'].loc[m]
            if np.nan_to_num(new_firedat[dict(time= rc[0], Y = rc[1], X= rc[2])]) == 0:
                new_firedat[dict(time= rc[0], Y = rc[1], X= rc[2])]= (merged['cell_frac'].loc[m] * merged['fire_size'].loc[m])/1e6
            else:
                new_firedat[dict(time= rc[0], Y = rc[1], X= rc[2])]+= (merged['cell_frac'].loc[m] * merged['fire_size'].loc[m])/1e6
        new_firedat.to_netcdf('../data/burnarea_%s.nc'%res, mode='w')

        print("Created a new fire burned area raster grid file!");

        return merged

    else:
        pred_flabel_arr= {1: ['Tmax', 'warm'], 2: ['VPD', 'warm'], 3: ['Prec', 'warm'], 4: ['Prec', 'antecedent_lag1', 'Antprec_lag1'], 5: ['Forest', 'annual'], \
                        6: ['Solar', 'warm'], 7: ['Wind', 'warm'], 8: ['Elev', 'static'], 9: ['Grassland', 'annual'], 10: ['RH', 'warm'], 11: ['FM1000', 'warm'], \
                        12: ['Tmax', 'moving_average_3mo', 'Ant_Tmax'], 13: ['VPD', 'moving_average_3mo', 'AvgVPD_3mo'], \
                        14: ['Prec', 'moving_average_3mo', 'Avgprec_3mo'], 15: ['RH', 'moving_average_3mo', 'Ant_RH'], 16: ['CAPE', 'warm'], 17: ['Urban', 'annual'],\
                        18: ['FFWI_max3', 'warm'], 19:['FFWI_max7', 'warm'], 20: ['Tmin', 'warm'], 21: ['Camp_dist', 'static'], 22: ['Camp_num', 'static'], \
                        23: ['Road_dist', 'static'], 24: ['Prec', 'antecedent_lag2', 'Antprec_lag2'], 25: ['Prec', 'moving_average_4mo', 'Avgprec_4mo'], \
                        26: ['Prec', 'moving_average_2mo', 'Avgprec_2mo'], 27: ['VPD_max3', 'warm'], 28: ['VPD_max7', 'warm'], 29: ['Tmax_max3', 'warm'], \
                        30: ['Tmax_max7', 'warm'], 31: ['Tmin_max3', 'warm'], 32: ['Tmin_max7', 'warm'], 33: ['Slope', 'static'], 34:['Southness', 'static'], \
                        35: ['VPD', 'moving_average_4mo', 'AvgVPD_4mo'], 36: ['VPD', 'moving_average_2mo', 'AvgVPD_2mo'], 37: ['SWE_mean', 'warm'], 38: ['SWE_max', 'warm'], \
                        39: ['SWE_mean', 'moving_average_3mo', 'AvgSWE_3mo'], 40: ['Delta_T', 'warm'], 41: ['Biomass', 'static'], 42: ['Popdensity', 'annual'], \
                        43: ['Housedensity', 'annual'], 44: ['Lightning', 'warm'], 45: ['RH_min3', 'warm'], 46: ['Shrub', 'annual']}
        
        pred_findx_arr= {'Tmax': 1, 'VPD': 2, 'Prec': 3, 'Forest': 6, 'Solar': 7, 'Wind': 20, 'Elev': 9, 'Grassland': 10, 'RH': 14, 'FM1000': 15, 'CAPE': 16, \
                         'Urban': 17, 'FFWI_max3': 18, 'FFWI_max7': 30, 'Tmin': 19, 'Camp_dist': 21, 'Camp_num': 22, 'Road_dist': 23, \
                         'VPD_max3': 24, 'VPD_max7': 25, 'Tmax_max3': 26, 'Tmax_max7': 27, 'Tmin_max3': 28, 'Tmin_max7': 29, 'Slope': 31, 'Southness': 32, \
                         'SWE_mean': 33, 'SWE_max': 34, 'Delta_T': 35, 'Biomass': 36, 'Popdensity': 37, 'Housedensity': 38, 'Lightning': 39, 'RH_min3': 40, 'Shrub': 41}
        pred_sindx_arr= {"warm": 1, "antecedent_lag1": 2, "annual": 3, "static": 4, "moving_average_3mo": 5, "moving_average_4mo": 6, "moving_average_2mo": 7, \
                         "antecedent_lag2": 8, "antecedent_avg": 9} 
        
        for i in tqdm(range(len(pred_flabel_arr))): 

            pred_var= pred_flabel_arr[i+1][0]
            seas_var= pred_flabel_arr[i+1][1]
            if pred_sindx_arr[seas_var] in [2, 5, 6, 7, 8, 9]:
                gdf_var= pred_flabel_arr[i+1][2]
            else:
                gdf_var= pred_var

            clim_var_data= clim_pred_var(pred_file_indx= pred_findx_arr[pred_var], pred_seas_indx= pred_sindx_arr[seas_var], tscale= "monthly", savg= False, \
                                         start_year= start_year, final_year= final_year)
            if res == '24km':
                if seas_var == 'static':
                    clim_var_arr= np.nanmean(sliding_window_view(clim_var_data[:-1 , :], (2, 2), axis= (0, 1)), axis= (2, 3))[::2, ::2]
                else:
                    clim_var_arr= np.nanmean(sliding_window_view(clim_var_data[:, :-1 , :], (2, 2), axis= (1, 2)), axis= (3, 4))[:, ::2, ::2]
            elif res == '12km':
                clim_var_arr= clim_var_data
            elif res == '1km':
                print("There is no climate functionality for this resolution currently!")

            if seas_var == 'static':
                merged[gdf_var]= [clim_var_arr[tuple(s[1:])] for s in merged['raster_coords']]
            else:
                merged[gdf_var]= [clim_var_arr[tuple(s)] for s in merged['raster_coords']]

        return merged

def file_io_func(firefile= None, firedf= None, lflag= 'L4', fflag= 'freq', io_flag= 'output'):
    
    regname= {1: "ca_sierra", 2: "ca_north_coast", 3: "ca_cent_coast", 4: "ca_south_coast", 5: "pnw_mts", 6: "columbia_plateau", 7:"northern_rockies", \
          8: "middle_rockies", 9: "southern_rockies", 10: "am_semidesert", 11: "aznm_mts", 12: "im_semidesert", 13: "im_desert", 14: "northern_great_plains", \
          15: "high_plains", 16: "colorado_plateau", 17: "sw_tablelands", 18: "ch_desert"}
    
    if fflag == 'freq':
        file_dir= '../data/clim_reg_%s'%lflag + '_fire_freqs/'
    elif fflag == 'size':
        file_dir= '../data/clim_reg_%s'%lflag + '_fire_sizes/'
    
    if io_flag == 'output':
        for r in tqdm(range(len(regname))):
            if fflag == 'size':
                data_df= init_fire_size_df(firefile= firefile, regindx= r+1, lflag= lflag)
            elif fflag == 'freq':
                data_df= init_fire_freq_df(firedf= firedf, regindx= r+1, lflag= lflag)
            data_df.to_hdf(file_dir + 'clim_%s'%regname[r+1] + '_%s'%lflag + '_fire_%s'%fflag + '_data.h5', key= 'df', mode= 'w')
    
    elif io_flag == 'input':
        dfs= [pd.read_hdf(file_dir + 'clim_%s'%regname[r+1] + '_%s'%lflag + '_fire_%s'%fflag + '_data.h5') for r in tqdm(range(len(regname)))]
        tmpdf= pd.concat(dfs, ignore_index= True)
        tmpdf= tmpdf.dropna().reset_index().drop(columns=['index'])
        tmpdf.to_hdf('../data/clim_%s'%lflag + '_fire_%s'%fflag + '_data.h5', key= 'df', mode= 'w')
        
def init_eff_clim_fire_df(firegdf, start_month= 372, tot_test_months= 60, hyp_flag= False):
    
    # creates a df of 'effective' climate for the fire size model by returning the average predictor variable value weighted by the burned area in each grid cell
    
    final_month= start_month + tot_test_months
    firetestgdf= firegdf[(firegdf['fire_month'] >= start_month)&(firegdf['fire_month'] < final_month)]
    firegdf= firegdf.drop(firetestgdf.index)
    firegroups= firegdf.groupby('fire_indx')
    
    newdf= firegdf.iloc[:, 0:8] # the first 8 columns contain geometry information 
    newdf= newdf.drop_duplicates(subset=['fire_indx']).reset_index().drop(columns= ['index'])
    climdf= pd.DataFrame(columns= firegdf.columns[8:])
    
    if hyp_flag:
        for k in firegroups.groups.keys():
            climdf= climdf.append(pd.DataFrame(data= np.reshape(np.average(firegroups.get_group(k).iloc[:, 8:], axis= 0, \
                                    weights= firegroups.get_group(k)['cell_frac']), (1, len(firegdf.columns[8:]))), \
                                    columns= firegdf.columns[8:]), ignore_index= True)
    else:
        for k in tqdm(firegroups.groups.keys()):
            climdf= climdf.append(pd.DataFrame(data= np.reshape(np.average(firegroups.get_group(k).iloc[:, 8:], axis= 0, \
                                    weights= firegroups.get_group(k)['cell_frac']), (1, len(firegdf.columns[8:]))), \
                                    columns= firegdf.columns[8:]), ignore_index= True)
    
    climdf= climdf.reset_index().drop(columns= ['index'])
    newdf= newdf.join(climdf)
    newdf= newdf.drop(columns= ['cell_frac'])
    
    return newdf

def init_clim_fire_grid(res= '12km', tscale= 'monthly', start_year= 1984, final_year= 2019, scaled= False, startmon= None, totmonths= None, seas_arr= None):
    
    # Initializes a dataframe with climate and fire frequency information at each grid point; startmon/totmonths is for test data!
    
    pred_flabel_arr= {1: ['Tmax', 'warm'], 2: ['VPD', 'warm'], 3: ['Prec', 'warm'], 4: ['Prec', 'antecedent_lag1', 'Antprec_lag1'], 5: ['Forest', 'annual'], \
                        6: ['Solar', 'warm'], 7: ['Wind', 'warm'], 8: ['Elev', 'static'], 9: ['Grassland', 'annual'], 10: ['RH', 'warm'], 11: ['FM1000', 'warm'], \
                        12: ['Tmax', 'moving_average_3mo', 'Ant_Tmax'], 13: ['VPD', 'moving_average_3mo', 'AvgVPD_3mo'], \
                        14: ['Prec', 'moving_average_3mo', 'Avgprec_3mo'], 15: ['RH', 'moving_average_3mo', 'Ant_RH'], 16: ['CAPE', 'warm'], 17: ['Urban', 'annual'],\
                        18: ['FFWI_max3', 'warm'], 19:['FFWI_max7', 'warm'], 20: ['Tmin', 'warm'], 21: ['Camp_dist', 'static'], 22: ['Camp_num', 'static'], \
                        23: ['Road_dist', 'static'], 24: ['Prec', 'antecedent_lag2', 'Antprec_lag2'], 25: ['Prec', 'moving_average_4mo', 'Avgprec_4mo'], \
                        26: ['Prec', 'moving_average_2mo', 'Avgprec_2mo'], 27: ['VPD_max3', 'warm'], 28: ['VPD_max7', 'warm'], 29: ['Tmax_max3', 'warm'], \
                        30: ['Tmax_max7', 'warm'], 31: ['Tmin_max3', 'warm'], 32: ['Tmin_max7', 'warm'], 33: ['Slope', 'static'], 34:['Southness', 'static'], \
                        35: ['VPD', 'moving_average_4mo', 'AvgVPD_4mo'], 36: ['VPD', 'moving_average_2mo', 'AvgVPD_2mo'], 37: ['SWE_mean', 'warm'], 38: ['SWE_max', 'warm'], \
                        39: ['SWE_mean', 'moving_average_3mo', 'AvgSWE_3mo'], 40: ['Delta_T', 'warm'], 41: ['Biomass', 'static'], 42: ['Popdensity', 'annual'], \
                        43: ['Housedensity', 'annual'], 44: ['Lightning', 'warm'], 45: ['RH_min3', 'warm'], 46: ['Shrub', 'annual']}
    pred_findx_arr= {'Tmax': 1, 'VPD': 2, 'Prec': 3, 'Forest': 6, 'Solar': 7, 'Wind': 20, 'Elev': 9, 'Grassland': 10, 'RH': 14, 'FM1000': 15, 'CAPE': 16, 'Urban': 17, 'FFWI_max3': 18, \
                     'FFWI_max7': 30, 'Tmin': 19, 'Camp_dist': 21, 'Camp_num': 22, 'Road_dist': 23, 'VPD_max3': 24, 'VPD_max7': 25, 'Tmax_max3': 26, 'Tmax_max7': 27, 'Tmin_max3': 28, \
                     'Tmin_max7': 29, 'Slope': 31, 'Southness': 32, 'SWE_mean': 33, 'SWE_max': 34, 'Delta_T': 35, 'Biomass': 36, 'Popdensity': 37, 'Housedensity': 38, 'Lightning': 39, \
                     'RH_min3': 40, 'Shrub': 41}
    pred_sindx_arr= {"warm": 1, "antecedent_lag1": 2, "annual": 3, "static": 4, "moving_average_3mo": 5, "moving_average_4mo": 6, "moving_average_2mo": 7, "antecedent_lag2": 8, "antecedent_avg": 9} 
    
    clim_fire_df= pd.DataFrame([])
    tot_months= (final_year + 1 - start_year)*12
    
    for i in tqdm(range(len(pred_flabel_arr))): 
        pred_var= pred_flabel_arr[i+1][0]
        seas_var= pred_flabel_arr[i+1][1]
        if pred_sindx_arr[seas_var] in [2, 5, 6, 7, 8, 9]: #variables whose antecedent characteristics are important
            gdf_var= pred_flabel_arr[i+1][2]
        else:
            gdf_var= pred_var

        clim_var_data= clim_pred_var(pred_file_indx= pred_findx_arr[pred_var], pred_seas_indx= pred_sindx_arr[seas_var], tscale= "monthly", savg= False, \
                                                                                                            start_year= start_year, final_year= final_year)
        if res == '24km':
            if seas_var == 'static':
                clim_var_arr= np.nanmean(sliding_window_view(clim_var_data[:-1 , :], (2, 2), axis= (0, 1)), axis= (2, 3))[::2, ::2]
            else:
                clim_var_arr= np.nanmean(sliding_window_view(clim_var_data[:, :-1 , :], (2, 2), axis= (1, 2)), axis= (3, 4))[:, ::2, ::2]
        elif res == '12km':
            clim_var_arr= clim_var_data
        elif res == '1km':
            print("There is no climate functionality for this resolution currently!")
       
        if tscale == 'monthly':
            if seas_var == 'static':
                var_arr= xarray.DataArray(data= np.tile(clim_var_arr, (tot_months, 1, 1)),
                            dims=["time", "Y", "X"],
                            coords=dict(
                            X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                            Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),
                            time= (["time"], np.linspace(0, tot_months - 1, tot_months, dtype= np.int64)),),)
            else:
                var_arr= xarray.DataArray(data= clim_var_arr,
                            dims=["time", "Y", "X"],
                            coords=dict(
                            X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                            Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),
                            time= (["time"], np.linspace(0, tot_months - 1, tot_months, dtype= np.int64)),),)
                
            if scaled == True:
                endmon= startmon + totmonths
                var_arr_train= xarray.concat([var_arr.where(var_arr.time < startmon, drop= True), var_arr.where(var_arr.time >= endmon, drop= True)], dim= "time")
                if (seas_var == 'static') | (seas_var == 'annual'):
                    var_arr= (var_arr - var_arr_train.mean(axis= (1, 2)))/var_arr_train.std(axis= (1, 2))
                else:
                    var_arr= (var_arr - var_arr_train.mean(axis= 0))/var_arr_train.std(axis= 0)
            
        elif tscale == 'longterm':
            if seas_var == 'static':
                var_arr= xarray.DataArray(data= clim_var_arr,
                            dims=["Y", "X"],
                            coords=dict(
                            X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                            Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),),)
            else:
                var_arr= xarray.DataArray(data= np.nanmean(clim_var_arr[seas_arr], axis= 0),
                                dims=["Y", "X"],
                                coords=dict(
                                X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                                Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),),)
        
        elif tscale == 'seasonal':
            if seas_var == 'static':
                var_arr= xarray.DataArray(data= clim_var_arr,
                            dims=["Y", "X"],
                            coords=dict(
                            X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                            Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),),)
            else:
                var_arr= xarray.DataArray(data= np.nanmean(clim_var_arr[seas_arr], axis= 0),
                                dims=["Y", "X"],
                                coords=dict(
                                X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                                Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),),)

        var_df= var_arr.to_dataframe(name= gdf_var).reset_index() #.dropna()
        clim_fire_df= pd.concat([clim_fire_df, var_df[gdf_var]], axis= 1)
        
    return clim_fire_df

def init_clim_fire_freq_df(res= '12km', tscale= 'monthly', start_year= 1984, final_year= 2019, scaled= False, startmon= None, totmonths= None, threshold= None, seas_arr= None):
    
    #creates a dataframe with climate variables and fire frequencies at monthly and annual resolutions
    
    clim_df= init_clim_fire_grid(res, tscale, start_year, final_year, scaled, startmon, totmonths, seas_arr) #time: ~ 8 mins
    tot_months= (final_year + 1 - start_year)*12
    
    if tscale == 'monthly':
        clim_fire_grid_df= xarray.open_dataarray('../data/12km/climate/primary/tmax.nc').sel(time=slice('%s'%str(start_year), '%s'%str(final_year))).to_dataframe(name='Tmax').reset_index() #.dropna()
        clim_fire_grid_gdf= gpd.GeoDataFrame(clim_fire_grid_df.Tmax, crs= 'EPSG:5070', geometry= gpd.points_from_xy(clim_fire_grid_df.X, clim_fire_grid_df.Y))
        reg_indx_arr= update_reg_indx(clim_fire_grid_gdf) #time: ~ 1.5 hrs
        clim_fire_grid_df['reg_indx']= reg_indx_arr

        tmax_arr= xarray.DataArray(data= clim_pred_var(pred_file_indx= 1, pred_seas_indx= 1, tscale= "monthly", savg= False, start_year= start_year, final_year= final_year),
            dims=["month", "Y", "X"],
            coords=dict(
                X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
                Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),
                time= (["month"], np.linspace(0, tot_months - 1, tot_months, dtype= np.int64)),),)
        tmax_df= tmax_arr.to_dataframe(name='Tmax').reset_index() #.dropna()
        
        coord_df= tmax_df[['X', 'Y', 'month']]
        coord_df['fire_freq']= np.zeros_like(len(coord_df), dtype= int)
        if final_year != 2020:
            fires_df= pd.read_hdf('../data/clim_fire_size_12km_train_data.h5').append(pd.read_hdf('../data/clim_fire_size_12km_test_data.h5'), ignore_index= True)
        else:
            fires_df= pd.read_hdf('../data/clim_fire_size_12km_train_w2020_data.h5').append(pd.read_hdf('../data/clim_fire_size_12km_test_w2020_data.h5'),\
                                                                                                                                               ignore_index= True)
        if threshold != None:
            fires_df= fires_df[fires_df['fire_size']/1e6 > 4].reset_index().drop(columns= ['index'])
            
        for ind in tqdm(range(len(fires_df))):
            freqind= coord_df[(coord_df.X == fires_df.loc[[ind]]['grid_x'][ind]) & (coord_df.Y == fires_df.loc[[ind]]['grid_y'][ind]) \
                                                                       & (coord_df.month == fires_df.loc[[ind]]['fire_month'][ind])].index[0]
            coord_df.loc[freqind, 'fire_freq']+= 1
            
        clim_df= pd.concat([clim_df, coord_df['fire_freq'], coord_df['month'], clim_fire_grid_df['reg_indx'], coord_df['X'], coord_df['Y']], axis=1)
        
    
    elif (tscale == 'longterm') | (tscale == 'seasonal'):
        clim_fire_grid_df= xarray.open_dataarray('../data/12km/climate/primary/tmax.nc').sel(time=slice('%s'%str(start_year), '%s'%str(final_year))).mean(dim= 'time').to_dataframe(name='Tmax').reset_index() #.dropna()
        clim_fire_grid_gdf= gpd.GeoDataFrame(clim_fire_grid_df.Tmax, crs= 'EPSG:5070', geometry= gpd.points_from_xy(clim_fire_grid_df.X, clim_fire_grid_df.Y))
        reg_indx_arr= update_reg_indx(clim_fire_grid_gdf)
        clim_fire_grid_df['reg_indx']= reg_indx_arr

        tmax_arr= xarray.DataArray(data= np.nanmean(clim_pred_var(pred_file_indx= 1, pred_seas_indx= 1, tscale= "monthly", savg= False), axis= 0),
        dims=["Y", "X"],
        coords=dict(
            X=(["X"], np.linspace(0, 154, 155, dtype= np.int64)),
            Y=(["Y"], np.linspace(0, 207, 208, dtype= np.int64)),),)
        tmax_df= tmax_arr.to_dataframe(name='Tmax').reset_index() #.dropna()

        coord_df= tmax_df[['X', 'Y']]
        coord_df['fire_freq']= np.zeros_like(len(coord_df), dtype= int)
        fires_df= pd.read_hdf('../data/clim_fire_size_12km_train_data.h5').append(pd.read_hdf('../data/clim_fire_size_12km_test_data.h5'), ignore_index= True)
        
        if tscale == 'longterm':
            for ind in tqdm(range(len(fires_df))):
                freqind= coord_df[(coord_df.X == fires_df.loc[[ind]]['grid_x'][ind]) & (coord_df.Y == fires_df.loc[[ind]]['grid_y'][ind])].index[0]
                coord_df.loc[freqind, 'fire_freq']+=1

            clim_df= pd.concat([clim_df, coord_df['fire_freq'], clim_fire_grid_df['reg_indx'], coord_df['X'], coord_df['Y']], axis=1)
            #clim_df.loc[:, 'fire_freq']= clim_df['fire_freq']/clim_df['fire_freq'].max()
        elif tscale == 'seasonal':
            for ind in tqdm(range(len(fires_df))): #tqdm
                if fires_df.loc[[ind]]['fire_month'][ind] in seas_arr:
                    freqind= coord_df[(coord_df.X == fires_df.loc[[ind]]['grid_x'][ind]) & (coord_df.Y == fires_df.loc[[ind]]['grid_y'][ind])].index[0]
                    coord_df.loc[freqind, 'fire_freq']+= 1

            clim_df= pd.concat([clim_df, coord_df['fire_freq'], clim_fire_grid_df['reg_indx'], coord_df['X'], coord_df['Y']], axis=1) 
        
        for r in tqdm(range(19)):
            clim_df.loc[clim_df.groupby('reg_indx').get_group(r).index, 'fire_freq'] = clim_df.groupby('reg_indx').get_group(r)['fire_freq']/(clim_df.groupby('reg_indx').get_group(r)['fire_freq'].max())

    return clim_df

def drop_col_func(mod_type, rh_flag= False, vpd_rh_flag= False, add_var_flag= False, add_var_list= None):
    
    if not rh_flag:
        dropcollist= ['CAPE', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'FFWI_max7', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', \
                             'Tmax_max7', 'VPD_max7', 'Tmin_max7', 'RH_min3']
    elif not vpd_rh_flag:
        dropcollist= ['CAPE', 'Solar', 'Ant_Tmax', 'VPD', 'AvgVPD_3mo', 'FFWI_max7', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', \
                             'Tmax_max7', 'VPD_max7', 'Tmin_max7', 'VPD_max3']
    else:
        dropcollist= ['CAPE', 'Solar', 'Ant_Tmax', 'Ant_RH', 'Tmin', 'FFWI_max7', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', \
                             'Tmax_max7', 'VPD_max7', 'Tmin_max7', 'RH_min3']
    if mod_type == 'minimal':
        dropcollist.extend(['Camp_dist', 'Camp_num', 'Slope', 'Southness', 'VPD', 'Tmax', 'Tmax_max3', 'Tmin_max3', \
                            'Prec', 'Elev', 'Tmin', 'Avgprec_3mo', 'Forest', 'Urban'])
    elif mod_type == 'normal':
        if not add_var_flag:
            dropcollist.extend(['AvgVPD_2mo', 'Camp_num', 'Elev'])
        else:
            dropcollist.extend(np.append(['AvgVPD_2mo', 'Camp_num', 'Elev'], add_var_list))
    
    return dropcollist

#archived function for indiviudal climate-fire correlations from Park's .nc file

#def init_fire_df(firefile):
# if fflag == 'size':
#     reg_df= pd.DataFrame({'Tmax': [], 'VPD': [], 'Prec': [], 'Antprec': [], 'ETo': [], 'Forest': [], 'Solar': [], 'Wind': [], 'Elev': [], 'Grassland': [], \
#                               'RH': [], 'FM1000': [], 'Ant_Tmax': [], 'Ant_VPD': [], 'Avgprec': [], 'Ant_RH': [], 'fire_size': [], 'month': pd.Series(dtype= 'int'), \
#                               'reg_indx': pd.Series(dtype= 'int')})
#     savg_flag= False

# if fflag == 'size':
#     reg_burnarea= bailey_ecoprovince_mask(firefile, region= regname[r])[0:tot_months - 1];
#     reg_fire_ind= np.argwhere(np.nan_to_num(reg_burnarea) != 0) #modify argument for fires > 405 ha??
#     reg_fire_sizes= np.asarray([reg_burnarea[tuple(s)].values for s in reg_fire_ind]).flatten()
#     reg_elev= clim_pred_var(pred_file_indx= 9, pred_seas_indx= 4, regindx= r, tscale= "monthly", savg= savg_flag) #no sense in using an 'average' elevation for a region

#     reg_fire_tmax= np.asarray([reg_tmax[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_vpd= np.asarray([reg_vpd[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_prec= np.asarray([reg_prec[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_antprec= np.asarray([reg_antprec[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_eto= np.asarray([reg_eto[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_forest= np.asarray([reg_forest[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_solar= np.asarray([reg_solar[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_wind= np.asarray([reg_wind[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_elev= np.asarray([reg_elev[tuple(s[1:])] for s in reg_fire_ind]).flatten()
#     reg_fire_grass= np.asarray([reg_grass[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_rh= np.asarray([reg_rh[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_fm1000= np.asarray([reg_fm1000[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_anttmax= np.asarray([reg_anttmax[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_antvpd= np.asarray([reg_antvpd[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_avgprec= np.asarray([reg_avgprec[tuple(s)] for s in reg_fire_ind]).flatten()
#     reg_fire_antrh= np.asarray([reg_antrh[tuple(s)] for s in reg_fire_ind]).flatten()

#     reg_df= reg_df.append(pd.DataFrame({'Tmax': reg_fire_tmax, 'VPD': reg_fire_vpd, 'Prec': reg_fire_prec, 'Antprec': reg_fire_antprec, 'ETo': reg_fire_eto, 
#                     'Forest': reg_fire_forest, 'Solar': reg_fire_solar, 'Wind': reg_fire_wind, 'Elev': reg_fire_elev, 'Grassland': reg_fire_grass, 'RH': reg_fire_rh,
#                     'FM1000': reg_fire_fm1000, 'Ant_Tmax': reg_fire_anttmax, 'Ant_VPD': reg_fire_antvpd, 'Avgprec': reg_fire_avgprec, 'Ant_RH': reg_fire_antrh,
#                     'fire_size': reg_fire_sizes, 'month': reg_fire_ind[:, 0].astype(int), 'reg_indx': r*np.ones(len(reg_fire_ind), dtype= int)}), ignore_index=True)

#%time data_df= init_fire_df(firefile= fire_file, fflag= 'size')
#data_df.to_hdf(data_dir + 'clim_fire_size_data.h5', key= 'df', mode= 'w')