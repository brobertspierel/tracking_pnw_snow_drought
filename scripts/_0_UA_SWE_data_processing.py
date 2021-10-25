import xarray as xr
import geopandas as gpd
import xagg as xa
import glob 
import sys 
import json
import os 
import pandas as pd
import pickle
import multiprocessing
from functools import partial
import re
import numpy as np 
import time 

"""
Collect zonal statistics from UA SWE (https://nsidc.org/data/nsidc-0719/versions/1) using either points (SNOTEL) or 
basins (HUCx) from the USGS. 
Inputs: 
-UA SWE netCDF files in a directory 
-shapefile of points of interest OR
-shapefile of polygons of interest 
-make sure to specify the dataset type (pts vs polygons) in the input args 

Note: this script prepares data for _0_combine_ua_swe_w_prism and both require the param file data_collection.txt
"""



def get_data(grid,shp,sort_col):#,slice_start,slice_end): 

    # Gridded data file (netcdf/climate data)
    ds = xr.open_dataset(grid).isel(time=slice(30,225)) #running with a confined period from November to end of April
    
    # Shapefile
    gdf = gpd.read_file(shp)
    #uncommenting this line will just get a small subset of the data for testing
    #gdf = gdf.iloc[:2,:]
    gdf = gdf[[sort_col,'geometry']]
    return ds,gdf

def get_polygon_stats(grid_ds,polygons,sort_col):
    """
    Aggregate stats from a netCDF over a polygon area using xagg and xarray. 
    Notes (from xagg docs): 
    aggregated can now be converted into an xarray dataset (using aggregated.to_dataset()), 
    or a geopandas geodataframe (using aggregated.to_dataframe()), or directly exported 
    to netcdf, csv, or shp files using aggregated.to_csv()/.to_netcdf()/.to_shp() 
    """
    # Get overlap between pixels and polygons
    weightmap = xa.pixel_overlaps(grid_ds['SWE'],polygons)

    # Aggregate data in [ds] onto polygons 
    aggregated = xa.aggregate(grid_ds['SWE'],weightmap)
    
    return aggregated.to_dataframe(), grid_ds['time']

def reformat_pts_output(stats_df,sort_col): 
    """
    Take the output of get_pts_stats func and reformat it to write to csv.
    """
    stats_df = stats_df.sort_values([sort_col,'time'])
    stats_df.drop(columns=['z','lat','lon'],inplace=True)
    stats_df.rename(columns={'time':'date','SWE':'swe'},inplace=True)
    stats_df['date'] = pd.to_datetime(stats_df['date'])

    return stats_df


def get_pts_stats(grid_ds,pts,sort_col): 
    """
    An alternative to the xagg approach for polygons, this only works for point-based datasets i.e. SNOTEL.
    """
    lats = []
    lons = []
    #change the column of shapely point objects into tuples of lon,lat
    coord_list = [(x,y) for x,y in zip(pts['geometry'].x , pts['geometry'].y)]
    #print(coord_list[:5])
    #pts will by default get a numeric id/index. Make a dict with those and site ids 
    id_dict = dict(zip(range(len(coord_list)),list(pts[sort_col])))

    #put the lats and lons into two lists 
    for i in coord_list: #these are tuples like (lon, lat)
        lats.append(i[1])
        lons.append(i[0])
    #convert lats and lons into xarrays 
    lats = xr.DataArray(lats, dims='z') #'z' is an arbitrary name placeholder
    lons = xr.DataArray(lons, dims='z')
    #this is where the data gets extracted by getting the nearest gridcell based on centroid? to a pt 
    data = grid_ds.sel(lat = lats, lon = lons, method = 'nearest')
    #convert the xarray to a pandas df so its easier to write to csv 
    out_df = data[['SWE','time']].to_dataframe().reset_index()
    #add the id col back in based on the z indexing 
    out_df[sort_col] = out_df.z.map(id_dict)
    #sort output by the site id 
    out_df = reformat_pts_output(out_df,sort_col)
    return out_df

def reformat_output(stats_gdf,ds_dates,sort_col): 
    """
    The df that comes out of the xagg function is not formatted correctly for further analysis. 
    Move some columns around and add the dates to make it ready for the next analysis steps. 
    """
    keys = [c for c in stats_gdf if c.startswith('SWE')] #cols by default are the xarray variable and a numberic id like SWE0
    #stack the SWE cols, duplicating items in the sort_col
    stats_gdf = pd.melt(stats_gdf, id_vars=sort_col, value_vars=keys, value_name='swe')
    #sort and rename cols 
    stats_gdf.rename(columns={'variable':'date'},inplace=True)
    time_ids = stats_gdf['date'].unique()
    #cast the date col to a unique list (set) and get datetimes from original data
    try: 
        dates = dict(zip(time_ids,list(ds_dates.values))) #this is for the actual nc data case
    except AttributeError: 
        print('processing a test case')
        dates = dict(zip(time_ids,list(ds_dates)))
    #replace the SWE0 type vals with dates
    stats_gdf.replace(dates,value=None,inplace=True) 
    #sort vals before writing to disk
    stats_gdf = stats_gdf.sort_values(by=[sort_col,'date'])
    #make sure the date col is treated like a date 
    stats_gdf['date'] = pd.to_datetime(stats_gdf['date']) 
    return stats_gdf

def split_dfs(input_df): 
    """
    Split a water year df into seasonal window sub-sections.
    Input: takes as input the output of reformat_output above
    """
    early = input_df.loc[(input_df['date'].dt.month == 11) | (input_df['date'].dt.month == 12)]
    mid = input_df.loc[(input_df['date'].dt.month == 1) | (input_df['date'].dt.month == 2)]
    late = input_df.loc[(input_df['date'].dt.month == 3) | (input_df['date'].dt.month == 4)]
    
    return early, mid, late 


def write_output(input_file,shapefile,output_dir,sort_col,data_type='polygons'):
    """
    Generates csv files for subsets of a winter season.
    """
    WY = re.findall('(\d{4})', os.path.split(input_file)[1])[0] #assumes there is only one possible year 
    #read in the shape and netCDF files 
    ds,gdf = get_data(input_file,shapefile,sort_col) 
    #generate the zonal stats- the processing is a bit different if its pts or polygons so 
    #decide which kind of shapefile we're dealing with based on the data_type variable. 
    if data_type.lower() == 'polygon': 
        output,out_dates = get_polygon_stats(ds,gdf,sort_col)
         #change the dfs to a format for output
        write_gdf = reformat_output(output,out_dates,sort_col)
    elif (data_type.lower() == 'pts') | (data_type.lower() == 'points'): 
        write_gdf = get_pts_stats(ds,gdf,sort_col)   
    else: 
        print('Double check the data_type variable, that can be polygons or pts/points')
    #create a filename for each time period (early, mid, late)
    early_fn = os.path.join(output_dir,f'ua_swe_full_{sort_col}_WY{WY}_11_to_12_months_data_output_formatted.csv')
    mid_fn = os.path.join(output_dir,f'ua_swe_full_{sort_col}_WY{WY}_1_to_2_months_data_output_formatted.csv')
    late_fn = os.path.join(output_dir,f'ua_swe_full_{sort_col}_WY{WY}_3_to_4_months_data_output_formatted.csv')
    #write to disk
    if not os.path.exists(early_fn): #just check one, they should all get made or none  
        split_dfs(write_gdf)[0].to_csv(early_fn) #early
        split_dfs(write_gdf)[1].to_csv(mid_fn) #mid
        split_dfs(write_gdf)[2].to_csv(late_fn) #late
    else: 
        print(f'{out_fn} already exists')

def main(nc_files,shapefile,sort_col,output_dir): 

    t0 = time.time()
    pool = multiprocessing.Pool(processes=30)
    
    make_data=partial(write_output,
                            shapefile=shapefile, 
                            output_dir=output_dir,
                            sort_col=sort_col,
                            data_type='pts'
                            )
    
    result_list = pool.map(make_data, nc_files)
    
    print(f'The process took {(time.time()-t0)/60} minutes to complete')

     ################
    #use for testing
    # fn="/vol/v1/general_files/user_files/ben/rasters/UA_SWE/UA_SWE/4km_SWE_Depth_WY1987_v01.nc"
    # shp = "/vol/v1/general_files/user_files/ben/shapefiles/hucs/huc_6_pnw_filtered_to_snotel.shp"

    # ds,gdf = get_data(fn,shapefile,sort_col)
    # get_pts_stats(ds,gdf,sort_col)
    #print(gdf)

    # write_output(fn,None,None)
    
    # test = pickle.load( open( "/vol/v1/general_files/user_files/ben/pickles/netcdf_gdf_test.p", "rb" ) )
    # from datetime import datetime

    # datelist = pd.date_range(datetime.today(), periods=10).tolist()
    # amended = reformat_output(test,datelist,sort_col)
    # amended['thing_month'] = amended.date.dt.month
    # print(amended)
    # print(split_dfs(amended))
    # early_fn = os.path.join(output_dir,f'ua_swe_full_{sort_col}_WY{1985}_11_to_12_months_data_output_formatted.csv')
    # mid_fn = os.path.join(output_dir,f'ua_swe_full_{sort_col}_WY{1985}_1_to_2_months_data_output_formatted.csv')
    # late_fn = os.path.join(output_dir,f'ua_swe_full_{sort_col}_WY{1985}_3_to_4_months_data_output_formatted.csv')
    # if not os.path.exists(early_fn): #just check one, they should all get made or none  
    #     split_dfs(amended)[0].to_csv(early_fn) #early
    #     split_dfs(amended)[1].to_csv(mid_fn) #mid
    #     split_dfs(amended)[2].to_csv(late_fn) #late
    
    # write_output(fn,
    #             polygons=shapefile, 
    #             output_dir=output_dir,
    #             sort_col=sort_col)

    ################

    
if __name__ == '__main__':
    #main()
    params = sys.argv[1]
    with open(str(params)) as f:
        variables = json.load(f)        
        #construct variables from param file
        input_dir = variables['input_dir']
        agg_shape = variables['agg_shape']
        agg_type = variables['agg_type']
        pickles = variables['pickles']
        stats_dir = variables['stats_dir']
    files = glob.glob(input_dir+'*.nc')

    main(files,
        agg_shape,
        agg_type,
        stats_dir)
    
        

