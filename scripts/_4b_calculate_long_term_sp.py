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



western = ['1708','1801','1710','1711','1709']
eastern = ['1701','1702','1705','1703','1601','1707','1706','1712','1704']


def read_in_and_reformat_data(input_dir,grouping_col,col_of_interest,drop_cols,resolution,param,plot_type): 
	"""Read in a dir of landsat sp data from GEE and make into a new df."""
	
	
	east_dfs_list = []
	west_dfs_list = []
	files = sorted(glob.glob(input_dir+'*.csv'))
	count = 0 
	for file in files: 
		df = pd.read_csv(file).sort_values(grouping_col)

		year = pd.to_datetime(df['date'].iloc[0]).year
		#print(year)
		try: 
			df.drop(columns=drop_cols,inplace=True)
			if param.upper() == 'SCA': 
				df=rs_funcs.convert_pixel_count_sq_km(df,col_of_interest,resolution)
			else: 
				print('Assuming I am configuring SP data')
			df['year'] = int(year)
			df[grouping_col] = df[grouping_col].astype('str')
		#	print(df)
		except KeyError: 
			continue 
		west_df = df.loc[df[grouping_col].str.contains('|'.join(western))]
		east_df = df.loc[df[grouping_col].str.contains('|'.join(eastern))]
		
		if plot_type.lower() == 'long_term': 
			west_df.drop(columns=[grouping_col],inplace=True)
			east_df.drop(columns=[grouping_col],inplace=True)
		else: 
			pass

		east_dfs_list.append(east_df)
		west_dfs_list.append(west_df)
		#dfs_list.append(df)
		#count += 1 
	output_west_df = pd.concat(west_dfs_list,axis=0)
	output_east_df = pd.concat(east_dfs_list,axis=0)
	#print(output_df)
	return output_west_df,output_east_df


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
		plot_type = 'other'
		plotting_param = 'SCA'

		dfs = read_in_and_reformat_data(csv_dir,'huc8','NDSI_Snow_Cover',['.geo','system:index','elev_min','elev_mean','elev_max'],resolution,plotting_param,plot_type)
		
		print(dfs)
		if plot_type.lower() == 'long_term': 
			fig,(ax,ax1) = plt.subplots(2,1,sharex=True,sharey=True)
			
			ax.grid()
			sns.lineplot(x='year',y='NDSI_Snow_Cover',data=dfs[0],ax=ax,color='#d8b365')
			ax.set_title(f'Western river basins MODIS/VIIRS {plotting_param}')
			ax.set_xticks(dfs[0]['year'].unique())
			ax.set_xlabel(' ')
			
			
			
			ax1.grid()
			sns.lineplot(x='year',y='NDSI_Snow_Cover',data=dfs[1],ax=ax1,color='#5ab4ac')
			ax1.set_title(f'Eastern river basins MODIS/VIIRS {plotting_param}')
			ax1.set_xticks(dfs[1]['year'].unique())
			ax1.set_xlabel(' ')
			
			if plotting_param.upper() == 'SP': 
				ax.set_ylabel('DJF snow persistence')
				ax1.set_ylabel('DJF snow persistence')
			elif plotting_param.upper() == 'SCA': 
				ax.set_ylabel(f'{plotting_param} (sq km)')
				ax1.set_ylabel(f'{plotting_param} (sq km)')
			else: 
				print('Your plotting param seems incorrect, double check and try again.')
			
			#plt.plot(df.set_index('huc8').T)
			plt.show() 
			plt.close('all')
		else: 
			pass
if __name__ == '__main__':
    main()