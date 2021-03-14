import os 
import sys
import glob
import pandas as pd 
import numpy as np 
import geopandas as gpd
import json 
import matplotlib.pyplot as plt  
import seaborn as sns 
import remote_sensing_functions as rs_funcs
import _4a_calculate_remote_sensing_snow_droughts as _4a_rs
import _3_obtain_all_data as obtain_data
import _4b_calculate_long_term_sp as _4b_rs 
import re
import math 
from scipy import stats




def main():
	"""
	Link the datatypes together and add summary stats. 
	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		season = variables["season"]
		pnw_shapefile = variables["pnw_shapefile"]
		huc_shapefile = variables['huc_shapefile']
		us_boundary = variables['us_boundary']
		stations = variables["stations"]		
		pickles = variables["pickles"]
		agg_step = variables["agg_step"]
		year_of_interest = int(variables["year_of_interest"])
		hucs_data = variables["hucs_data"]
		sentinel_csv_dir = variables["sentinel_csv_dir"]
		optical_csv_dir = variables["optical_csv_dir"]
		huc_level = variables["huc_level"]
		resolution = variables["resolution"]
		palette = variables["palette"]

		#user defined functions 
		#plot_func = 'quartile'
		elev_stat = 'elev_mean'
		#self,sentinel_data,optical_data,snotel_data,hucs_data,huc_level,resolution): 
		#get all the data 
		snotel_data = pickles+f'short_term_snow_drought_{year_of_interest}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
		
		#instantiate the acquireData class and read in snotel, sentinel and modis/viirs data 
		input_data = obtain_data.AcquireData(sentinel_csv_dir,optical_csv_dir,snotel_data,hucs_data,huc_level,resolution)
		short_term_snow_drought = input_data.get_snotel_data()
		sentinel_data = input_data.get_sentinel_data('filter')
		optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
		
		# pd.set_option("display.max_rows", None, "display.max_columns", None) #change to print an entire df
		# #combine the sentinel and optical data 
		
		#drop redundant columns 
		sentinel_data.drop(columns=['elev_min','elev_mean','elev_max'],inplace=True)
		rs_df=rs_funcs.merge_remote_sensing_data(optical_data,sentinel_data)
		#remove snow persistence values lower than 20% as per (Saavedra et al)
		
		rs_df['wet_snow_by_area'] = rs_df['filter']/rs_df['NDSI_Snow_Cover'] #calculate wet snow as fraction of snow covered area
		
		#make sure that the cols used for merging are homogeneous in type 
		rs_df['huc8'] = pd.to_numeric(rs_df['huc'+huc_level])
		rs_df['date'] = _4a_rs.convert_date(rs_df,'date')
		
		#create the different snow drought type dfs 

		#do dry first 
		dry_combined = _4a_rs.create_snow_drought_subset(short_term_snow_drought,'dry',huc_level)
		#merge em 
		dry_combined=dry_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
		#get the rs data for the time periods of interest for a snow drought type 
		#dry_optical=dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].mean()
		dry_combined.rename(columns={'wet_snow_by_area':'dry_WSCA'},inplace=True)
		dry_sar = dry_combined.groupby('huc'+huc_level)['dry_WSCA',elev_stat].median() #changed col from pct change to filter 2/1/2021
	
		#then do warm 
		warm_combined = _4a_rs.create_snow_drought_subset(short_term_snow_drought,'warm',huc_level)
		#merge em 
		warm_combined=warm_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner')
		#get the rs data for the time periods of interest for a snow drought type 
		#warm_optical=warm_combined.groupby('huc'+huc_level)['ndsi_pct_change'].min() 
		warm_combined.rename(columns={'wet_snow_by_area':'warm_WSCA'},inplace=True)
		warm_sar = warm_combined.groupby('huc'+huc_level)['warm_WSCA',elev_stat].median()
		
		#then do warm/dry
		warm_dry_combined = _4a_rs.create_snow_drought_subset(short_term_snow_drought,'warm_dry',huc_level)
		#merge em 
		warm_dry_combined=warm_dry_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner')
		#get the rs data for the time periods of interest for a snow drought type 
		#warm_dry_optical=warm_dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].sum()
		warm_dry_combined.rename(columns={'wet_snow_by_area':'warm_dry_WSCA'},inplace=True)
		warm_dry_sar = warm_dry_combined.groupby('huc'+huc_level)['warm_dry_WSCA',elev_stat].median()

		#try making a df of time steps that DO NOT have snow droughts for comparing
		no_snow_drought = _4a_rs.create_snow_drought_subset(short_term_snow_drought,'date',huc_level)
		no_drought_combined=no_snow_drought.merge(rs_df, on=['date','huc'+huc_level],how='inner')

		no_drought_combined.rename(columns={'wet_snow_by_area':'no_drought_WSCA'},inplace=True)
		no_drought_sar = no_drought_combined.groupby('huc'+huc_level)['no_drought_WSCA'].median()
		#print(no_drought_sar)
		#print('no drought sar', no_drought_sar.shape)
		
		#dfs = [dry_sar.reset_index(),warm_sar.reset_index(),warm_dry_sar.reset_index()]
		dfs = dry_sar.reset_index().merge(warm_sar.reset_index(),on=['huc'+huc_level],how='outer')
		dfs = dfs.merge(warm_dry_sar.reset_index(),on=['huc'+huc_level],how='outer')
		dfs.drop(columns={f'{elev_stat}_x',f'{elev_stat}_y'},inplace=True)
		print('dfs shape',dfs.shape)
		
		dfs = dfs.merge(no_drought_sar.reset_index(),on=['huc'+huc_level],how='outer')
	
		#do a little bit of cleaning
		dfs.replace(np.inf,np.nan,inplace=True)
		#dfs[['dry_WSCA','warm_WSCA','warm_dry_WSCA','no_drought_WSCA']]>=1 = 1 #=dfs[dfs['dry_WSCA','warm_WSCA','warm_dry_WSCA','no_drought_WSCA']>=1,]
		
		print('dfs look like: ',dfs)
		#anywhere wet snow exceeeds snow covered area ie value is greater than 1, set it to 1
		# dfs['dry_WSCA'] = dfs['dry_WSCA']
		#dfs.loc[dfs['dry_WSCA','warm_WSCA','warm_dry_WSCA'] > 1] = 1  
		dfs.loc[dfs['dry_WSCA'] > 1,'dry_WSCA'] = np.nan
		dfs.loc[dfs['warm_WSCA'] > 1,'warm_WSCA'] = np.nan
		dfs.loc[dfs['warm_dry_WSCA'] > 1,'warm_dry_WSCA'] = np.nan  
		dfs.loc[dfs['no_drought_WSCA'] > 1,'no_drought_WSCA'] = np.nan
		#df.loc[df.ID == 103, ['FirstName', 'LastName']] = 'Matt', 'Jones'

		print('dfs now look like: ', dfs)

if __name__ == '__main__':
    main()