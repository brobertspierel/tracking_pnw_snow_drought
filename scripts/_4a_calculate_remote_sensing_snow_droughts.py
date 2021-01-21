import os 
import sys
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import geopandas as gpd
import _3_obtain_all_data as obtain_data
import remote_sensing_functions as rs_funcs
from mpl_toolkits.axes_grid1 import make_axes_locatable


def convert_date(input_df,col_of_interest): 
	"""Helper function."""
	input_df[col_of_interest] = pd.to_datetime(input_df[col_of_interest])
	return input_df[col_of_interest]


def create_snow_drought_subset(input_df,col_of_interest): 
	"""Helper function."""

	drought_list = ['dry','warm','warm_dry']
	drought_list.remove(col_of_interest)
	df = short_term_snow_drought.drop(columns=drought_list)
	df['huc_id'] = df['huc_id'].astype('int')
	df[col_of_interest] = convert_date(df,'dry')
	#rename cols to match rs data for ease 
	df.rename(columns={col_of_interest:'date','huc_id':'huc8'},inplace=True)
	#get rid of na fields
	df = df.dropna()


def main():
	"""
	Plot the long term snow drought types and trends. 
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
		#self,sentinel_data,optical_data,snotel_data,hucs_data,huc_level,resolution): 
		#get all the data 
		snotel_data = pickles+f'short_term_snow_drought_{season}_{agg_step}_day_time_step_w_hucs'
		
		#instantiate the acquireData class and read in snotel, sentinel and modis/viirs data 
		input_data = obtain_data.AcquireData(sentinel_csv_dir,optical_csv_dir,snotel_data,hucs_data,huc_level,resolution)
		short_term_snow_drought = input_data.get_snotel_data()
		sentinel_data = input_data.get_sentinel_data('filter')
		optical_data = input_data.get_optical_data()

		
		pd.set_option("display.max_rows", None, "display.max_columns", None) #change to print an entire df
		#combine the sentinel and optical data 
		
		rs_df=rs_funcs.merge_remote_sensing_data(optical_data,sentinel_data)
		
		#remove snow persistence values lower than 20% as per (Saavedra et al)
		rs_df = rs_df.loc[rs_df['NDSI_Snow_Cover']>= 0.2]
		
		#calculate percent change values from one time period to the next 
		rs_df['ndsi_pct_change'] = rs_df['NDSI_Snow_Cover'].pct_change()*100
		rs_df['sar_pct_change'] = rs_df['snow_ratio'].pct_change()*100
		
		#make sure that the cols used for merging are homogeneous in type 
		rs_df['huc8'] = pd.to_numeric(rs_df['huc8'])
		rs_df['date'] = convert_date(rs_df,'date')

		#make dfs of different snow drought types 
		# dry_combined = short_term_snow_drought.drop(columns=['warm','warm_dry'])
		
		# #make sure the merging cols are homogeneous 
		# dry_combined['huc_id'] = dry_combined['huc_id'].astype('int')
		# dry_combined['dry'] = convert_date(dry_combined,'dry')
		# #rename cols to match rs data for ease 
		# dry_combined.rename(columns={'dry':'date','huc_id':'huc8'},inplace=True)
		# #get rid of na fields
		# dry_combined = dry_combined.dropna()

		# #create the 
		# warm_dry_combined = short_term_snow_drought.drop(columns=['warm','dry'])

		# warm_dry_combined['huc_id'] = warm_dry_combined['huc_id'].astype('int')
		# warm_dry_combined['warm_dry'] = convert_date(warm_dry_combined,'warm_dry')
		# #rename cols to match rs data for ease 
		# warm_dry_combined.rename(columns={'warm_dry':'date','huc_id':'huc8'},inplace=True)
		# #get rid of na fields
		# warm_dry_combined = warm_dry_combined.dropna()
		dry_combined = create_snow_drought_subset(short_term_snow_drought,'dry')
		#merge em 
		combined=dry.merge(rs_df, on=['date','huc8'], how='inner')
		#for item in combined['huc8'].unique(): 
		dry_optical=combined.groupby('huc8')['ndsi_pct_change'].mean()
		dry_sar = combined.groupby('huc8')['sar_pct_change'].mean()
	
		hucs_shp = gpd.read_file(huc_shapefile)
		us_bounds = gpd.read_file(us_boundary)
		hucs_shp['huc8'] = hucs_shp['huc8'].astype('int32')
		hucs_shp = hucs_shp.merge(dry_optical,how='inner',on='huc8')
		hucs_shp = hucs_shp.merge(dry_sar,how='inner',on='huc8')
		hucs_shp['sar_pct_change'] = hucs_shp['sar_pct_change'].replace(np.inf, np.nan)
		#print(hucs_shp)
		minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds


		fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(15,15))
		us_bounds.plot(ax=ax1,color='white', edgecolor='black')
		hucs_shp.plot(ax=ax1,color='gray',edgecolor='darkgray')
		divider = make_axes_locatable(ax1)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		hucs_shp.plot(ax=ax1,column='sar_pct_change',vmin=hucs_shp['sar_pct_change'].min(),vmax=hucs_shp['sar_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
		
		ax1.set_title('Mean sar change in wet snow area')
		ax1.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		ax1.set_ylim(miny - 1, maxy + 1)

		us_bounds.plot(ax=ax2,color='white', edgecolor='black')
		hucs_shp.plot(ax=ax2,color='gray',edgecolor='darkgray')
		divider = make_axes_locatable(ax2)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		hucs_shp.plot(ax=ax2,column='ndsi_pct_change',vmin=hucs_shp['ndsi_pct_change'].min(),vmax=hucs_shp['ndsi_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
		ax2.set_title('Mean optical change in SP')
		ax2.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		ax2.set_ylim(miny - 1, maxy + 1)
		
		plt.tight_layout()
		plt.show()
		plt.close('all')

if __name__ == '__main__':
    main()










