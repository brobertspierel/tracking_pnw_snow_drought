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
import time 


def get_data(grid,shp):#,slice_start,slice_end): 

    # Gridded data file (netcdf/climate data)
    ds = xr.open_dataset(grid).isel(time=slice(0,250)) #running with a sort of arbitrary end that is beyond what we want 
    # ds['SWE'] = ds['SWE'].to_masked_array()
    #print(ds['time'].values)
    #print(type(ds['time']))
    # Shapefile
    gdf = gpd.read_file(shp)

    #gdf = gdf.iloc[:5,:]
    gdf = gdf[['huc8','geometry']]
    # print(gdf.shape)
    # print(gdf.columns)
    # print(gdf)
    return ds,gdf

def get_stats(grid_ds,polygons,sort_col):
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
    
    #return aggregated.to_dataframe(), grid_ds['time']
   
 # pickle.dump( aggregated.to_dataframe(), open( os.path.join(pickle_dir,"netcdf_gdf_test1.p"), "wb" ) )

    return pd.merge(aggregated.to_dataframe(), polygons, on=sort_col)

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
    print('time',time_ids)
    #cast the date col to a unique list (set) and get datetimes from original data
    dates = dict(zip(time_ids,list(ds_dates.values))) 
    #replace the SWE0 type vals with dates
    stats_gdf.replace(dates,value=None,inplace=True) 
    stats_gdf = stats_gdf.sort_values(by=[sort_col,'date'])
    stats_gdf['date'] = pd.to_datetime(stats_gdf['date']) #make sure the date col is treated like a date 
    return stats_gdf

def write_output(input_file,polygons,output_dir,sort_col):
    """
    Generates csv files for subsets of a winter season.
    """
    year = re.findall('(\d{4})', os.path.split(input_file)[1])[0] #assumes there is only one possible year 

    ds,gdf = get_data(input_file,polygons) 
    output,out_dates = get_stats(ds,gdf,sort_col)
    write_gdf = reformat_output(output,out_dates,sort_col)
    out_fn = os.path.join(output_dir,f'ua_swe_full_WY{year}_data_output_formatted.csv')
    if not os.path.exists(out_fn): 
        write_gdf.to_csv(out_fn)
    else: 
        print(f'{out_fn} already exists')

     #now iterate through the time chunks that we're going to want (early, mid, late)
    #files are already organized in water years so the year var should be all set 
    # for per in [(0,61),(1,2),(3,4)]: 
    #     #deal with leap years- we're indexing by integers, not dates 
    #     if not int(year) % 4 == 0: 
    #         pass
    #     else: 
    #         pass

def main(nc_files,shapefile,sort_col,output_dir): 

    # fn="/vol/v1/general_files/user_files/ben/rasters/UA_SWE/UA_SWE/4km_SWE_Depth_WY1987_v01.nc"
    # shp = "/vol/v1/general_files/user_files/ben/shapefiles/hucs/huc_6_pnw_filtered_to_snotel.shp"
    # write_output(fn,None,None)
    
    # test = pickle.load( open( "/vol/v1/general_files/user_files/ben/pickles/netcdf_gdf_test.p", "rb" ) )
    # from datetime import datetime

    # datelist = pd.date_range(datetime.today(), periods=10).tolist()
    # amended = reformat_output(test,datelist,sort_col)

    # print(test)
    # print(amended)

    t0 = time.time()
    pool = multiprocessing.Pool(processes=25)
    
    make_data=partial(write_output,
                            polygons=shapefile, 
                            output_dir=output_dir,
                            sort_col=sort_col,)
    
    result_list = pool.map(make_data, nc_files)
    
    print(f'The process took {(time.time()-t0)/60} minutes to complete')
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
    
        

