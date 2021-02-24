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
import re
import _4b_calculate_long_term_sp as _4b_rs
import _4a_calculate_remote_sensing_snow_droughts as _4a_rs
import _3_obtain_all_data as obtain_data




def main():
	"""
	Plot the long term snow drought types and trends. 
	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		sp_data = variables['sp_data']
		csv_dir = variables['csv_dir']	
		resolution = variables['resolution']
		huc_level = variables['huc_level']
		resolution = variables['resolution']
		pickles = variables['pickles']
		year_of_interest = variables['year_of_interest']
		season = variables['season']
		agg_step = variables['agg_step']
		optical_csv_dir = variables['optical_csv_dir']
		palette = variables['palette']
		modis_dir = variables['modis_dir']
		viirs_dir = variables['viirs_dir']
		
		#set a few script specific user params
		plot_type = 'long_term'
		plotting_param = 'SCA'
		#plot_func = 'quartile'
		elev_stat = 'elev_mean'
		

		viirs_dfs = _4b_rs.read_in_and_reformat_data(modis_dir,'huc8','NDSI_Snow_Cover',['.geo','system:index'],resolution,plotting_param,plot_type) #comes out in the form (western,eastern)
		modis_dfs = _4b_rs.read_in_and_reformat_data(viirs_dir,'huc8','NDSI_Snow_Cover',['.geo','system:index'],resolution,plotting_param,plot_type)
		
		#west_grouped = viirs_dfs[0].groupby('year')['NDSI_Snow_Cover'].mean().to_frame().reset_index()
		#print(west_grouped)
		for df1,df2 in zip(viirs_dfs,modis_dfs): 
			df1.rename(columns={'NDSI_Snow_Cover':'NDSI_Snow_Cover_viirs','huc8':'huc8_viirs'},inplace=True)
			df2.rename(columns={'NDSI_Snow_Cover':'NDSI_Snow_Cover_modis','huc8':'huc8_modis'},inplace=True)
		# print(viirs_dfs[0].shape)
		# print(viirs_dfs[0].sort_values('date'))
		#west = viirs_dfs[0].sort_values('date').merge(modis_dfs[0].sort_values('date'),on=['year'],how='inner')
		west = pd.concat([viirs_dfs[0].sort_values('date'),modis_dfs[0].sort_values('date')],axis=1)
		east = pd.concat([viirs_dfs[1].sort_values('date'),modis_dfs[1].sort_values('date')],axis=1)
		
		print('west')
		print(west.NDSI_Snow_Cover_modis.value_counts())
		print(west.NDSI_Snow_Cover_viirs.value_counts())
		print('east')
		print(east.NDSI_Snow_Cover_modis.value_counts())
		print(east.NDSI_Snow_Cover_viirs.value_counts())
		
		west['bias'] = west['NDSI_Snow_Cover_modis']-west['NDSI_Snow_Cover_viirs']
		east['bias'] = east['NDSI_Snow_Cover_modis']-east['NDSI_Snow_Cover_viirs']

		print(west.bias)
		print(east.bias)

		west_med = west.groupby('huc8_viirs')['bias'].median()
		west_mean = west.groupby('huc8_viirs')['bias'].mean()
		west_std = west.groupby('huc8_viirs')['bias'].std()

		east_med = east.groupby('huc8_viirs')['bias'].median()
		east_mean = east.groupby('huc8_viirs')['bias'].mean()
		east_std = east.groupby('huc8_viirs')['bias'].std()
		
		# print('west')
		# print(west_med)
		# print(west_std)
		# print('east')
		# print(east_med)
		# print(east_std)

		# print(west_med.median())
		# print(east_med.median())
		# print(west.bias.median())
		# print(east.bias.median())
		# print(west.bias.std())
		# print(east.bias.std())

		#print(west_med.to_frame().reset_index())
		
		fig,(ax,ax1) = plt.subplots(2)
		west_med.plot(ax=ax,color='green',label='median')
		west_mean.plot(ax=ax,color='blue',label='mean')
		east_med.plot(ax=ax1,color='green')
		east_mean.plot(ax=ax1,color='blue')
		ax.legend()
		# ax.plot(west_med.to_frame().reset_index())
		# #sns.lineplot(x='year',y='NDSI_Snow_Cover_viirs',data=west,ax=ax)
		plt.show()
		plt.close('all')


if __name__ == '__main__':
    main()