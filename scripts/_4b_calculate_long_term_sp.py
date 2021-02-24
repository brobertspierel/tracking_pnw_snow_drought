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
import re
import math 

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

		year = pd.to_datetime(df['date'].iloc[-1]).year #double check that this is working correctly. Suspicious that its not actually getting the water year 
		try: 
			df.drop(columns=drop_cols,inplace=True)
			if param.upper() == 'SCA': 
				print('Converting sca pixel count to area')
				df=rs_funcs.convert_pixel_count_sq_km(df,col_of_interest,resolution)
			else: 
				print('Assuming I am configuring SP data')
			df['year'] = int(year)
			df[grouping_col] = df[grouping_col].astype('str')

		except KeyError: 
			continue 
		west_df = df.loc[df[grouping_col].str.contains('|'.join(western))]
		east_df = df.loc[df[grouping_col].str.contains('|'.join(eastern))]

		# if plot_type.lower() == 'long_term': 
		# 	west_df.drop(columns=[grouping_col],inplace=True)
		# 	east_df.drop(columns=[grouping_col],inplace=True)
		# else: 
		# 	pass
		
		east_dfs_list.append(east_df)
		west_dfs_list.append(west_df)
		#dfs_list.append(df)
		#count += 1 
	output_west_df = pd.concat(west_dfs_list,axis=0)
	output_east_df = pd.concat(east_dfs_list,axis=0)

	return output_west_df,output_east_df

def split_basins(input_df,grouping_col): 
	"""Read in a dir of landsat sp data from GEE and make into a new df."""
	input_df[grouping_col] = input_df[grouping_col].astype('str')
	west_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(western))]
	east_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(eastern))]
	
	#replace instances of inf with nan and drop the grouping_col so its not in the mean
	west_df.replace(np.inf,np.nan,inplace=True)
	east_df.replace(np.inf,np.nan,inplace=True)
	
	west_df.drop(columns=[grouping_col,'elev_mean'],inplace=True) #added the hardcoded drop of the elev col to clean up for plotting
	east_df.drop(columns=[grouping_col,'elev_mean'],inplace=True)

	west_mean = west_df.mean(axis=0)
	east_mean = east_df.mean(axis=0)

	return west_mean,east_mean


def merge_dfs(snotel_data,rs_data,huc_level,col_of_interest,elev_stat): 
	"""Merge snotel snow drought data with RS data."""
	dry_combined = _4a_rs.create_snow_drought_subset(snotel_data,'dry',huc_level)
	#merge em 
	dry_combined=dry_combined.merge(rs_data, on=['date','huc'+huc_level], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
	#get the rs data for the time periods of interest for a snow drought type 
	#dry_optical=dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].mean()
	dry_combined.rename(columns={col_of_interest:f'dry_{col_of_interest}'},inplace=True)
	dry_sar = dry_combined.groupby('huc'+huc_level)[f'dry_{col_of_interest}',elev_stat].median() #changed col from pct change to filter 2/1/2021

	#then do warm 
	warm_combined = _4a_rs.create_snow_drought_subset(snotel_data,'warm',huc_level)
	#merge em 
	warm_combined=warm_combined.merge(rs_data, on=['date','huc'+huc_level], how='inner')
	#get the rs data for the time periods of interest for a snow drought type 
	#warm_optical=warm_combined.groupby('huc'+huc_level)['ndsi_pct_change'].min() 
	warm_combined.rename(columns={col_of_interest:f'warm_{col_of_interest}'},inplace=True)
	warm_sar = warm_combined.groupby('huc'+huc_level)[f'warm_{col_of_interest}',elev_stat].median()
	
	#then do warm/dry
	warm_dry_combined = _4a_rs.create_snow_drought_subset(snotel_data,'warm_dry',huc_level)
	#merge em 
	warm_dry_combined=warm_dry_combined.merge(rs_data, on=['date','huc'+huc_level], how='inner')
	#get the rs data for the time periods of interest for a snow drought type 
	#warm_dry_optical=warm_dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].sum()
	warm_dry_combined.rename(columns={col_of_interest:f'warm_dry_{col_of_interest}'},inplace=True)
	#print(warm_dry_combined.shape)
	warm_dry_sar = warm_dry_combined.groupby('huc'+huc_level)[f'warm_dry_{col_of_interest}',elev_stat].median()
	#print(warm_dry_sar.shape)

	#try making a df of time steps that DO NOT have snow droughts for comparing
	no_snow_drought = _4a_rs.create_snow_drought_subset(snotel_data,'date',huc_level)
	no_drought_combined=no_snow_drought.merge(rs_data, on=['date','huc'+huc_level],how='inner')

	no_drought_combined.rename(columns={col_of_interest:f'no_drought_{col_of_interest}'},inplace=True)
	no_drought_sar = no_drought_combined.groupby('huc'+huc_level)[f'no_drought_{col_of_interest}'].median()

		
	dfs = dry_sar.reset_index().merge(warm_sar.reset_index(),on=['huc'+huc_level],how='outer')
	dfs = dfs.merge(warm_dry_sar.reset_index(),on=['huc'+huc_level],how='outer')
	dfs.drop(columns={f'{elev_stat}_x',f'{elev_stat}_y'},inplace=True)
	#print('dfs shape',dfs.shape)
	
	dfs = dfs.merge(no_drought_sar.reset_index(),on=['huc'+huc_level],how='outer')
	#print(dfs)
	return dfs 

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
		testing = variables['testing']
		
		#set a few script specific user params
		plot_type = 'long_term'
		plotting_param = 'SCA'
		#plot_func = 'quartile'
		elev_stat = 'elev_mean'
		

		#dfs = read_in_and_reformat_data(csv_dir,'huc8','NDSI_Snow_Cover',['.geo','system:index','elev_min','elev_max'],resolution,plotting_param,plot_type)
		dfs = read_in_and_reformat_data(modis_dir,'huc8','NDSI_Snow_Cover',['.geo','system:index'],resolution,plotting_param,plot_type)
		
	
		if not plot_type == 'long_term': 
			west_dfs_list = []
			east_dfs_list = []
			years = []
			optical_files = sorted(glob.glob(csv_dir+'*.csv'))
			for file in optical_files: 
				print(file)
				year = int(re.findall('(\d{4})-\d{2}-\d{2}', os.path.split(file)[1])[1]) #gets a list with the start and end of the water year, take the second one 
				print(year)
				
				snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'

				input_data = obtain_data.AcquireData(None,file,snotel_data,None,huc_level,resolution)
				
				short_term_snow_drought = input_data.get_snotel_data()
				
				optical_data = input_data.get_optical_data('NDSI_Snow_Cover')

				#optical_data = dfs[1]
				optical_data[f'huc{huc_level}'] = pd.to_numeric(optical_data['huc'+huc_level]) 
				optical_data['date'] = _4a_rs.convert_date(optical_data,'date')

				#combine the remote sensing and snotel data using the snotel dates of snow droughts to extract rs data 
				merged=merge_dfs(short_term_snow_drought,optical_data,huc_level,'NDSI_Snow_Cover',elev_stat)
				
				split_dfs=split_basins(merged,f'huc{huc_level}') #returns the merged df split into two dfs, west (0) and east (1)

				west_dfs_list.append(split_dfs[0].to_frame().T)
				east_dfs_list.append(split_dfs[1].to_frame().T)
				years.append(year) #save these for labeling 
			
			output_west_df = pd.concat(west_dfs_list,ignore_index=True)
			output_east_df = pd.concat(east_dfs_list,ignore_index=True)
			
			output_west_df['year']=years
			output_east_df['year']=years
			
			output_west_df = output_west_df.set_index('year')
			output_east_df = output_east_df.set_index('year')
			print(output_west_df)
			#plot it
			fig,(ax,ax1) = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,8))

			font = {'family' : 'Times New Roman',
			        'weight' : 'normal',
			        'size'   : 18}

			plt.rc('font', **font)

			#pal=['#a6cee3','#1f78b4','#b2df8a','#33a02c']
			labels=['Dry', 'Warm', 'Warm/dry', 'No drought']
			palette = list(palette.values())
			output_west_df.plot(ax=ax,color=palette,linewidth=3.5,)
			ax.set_xticks(years)
			ax.set_title('Western Basins SCA by drought type')
			ax.set_xlabel(' ')
			ax.set_ylabel('SCA (sq km)')
			ax.grid()
			ax.legend(labels=labels)
			ax.tick_params(axis='x', labelsize=15)

			
			output_east_df.plot(ax=ax1,color=palette,linewidth=3.5,legend=False)
			ax1.set_xticks(years)
			ax1.set_title('Eastern Basins SCA by drought type')
			ax1.set_xlabel(' ')
			ax1.set_ylabel('SCA (sq km)')
			ax1.grid()

			plt.tight_layout()
			plt.show()
			plt.close('all')

		elif plot_type.lower() == 'long_term': #use to make plots of the long term trends in optical data 
			fig,(ax,ax1) = plt.subplots(2,1,sharex=True,sharey=True)
			west_yrs_basins = dfs[0].groupby(['huc8','year'])['NDSI_Snow_Cover'].max().to_frame().reset_index()
			#ci = 1.96 * np.std(y)/np.mean(y)
			#west_yrs_basins['ci']=1.96 * np.std('NDSI_Snow_Cover')/np.mean('NDSI_Snow_Cover')
			print(west_yrs_basins)
			stats = west_yrs_basins.groupby(['year'])['NDSI_Snow_Cover'].agg(['mean', 'count', 'std'])
			# print(stats)
			# print('-'*30)

			ci95_hi = []
			ci95_lo = []

			for i in stats.index:
			    m, c, s = stats.loc[i]
			    ci95_hi.append(m + 1.96*s/math.sqrt(c))
			    ci95_lo.append(m - 1.96*s/math.sqrt(c))

			stats['ci95_hi'] = ci95_hi
			stats['ci95_lo'] = ci95_lo
			print(stats)

			west_yrs = west_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			print(west_yrs)
		
			#dfs[0].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			#linreg_df_west = dfs[0].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			ax.grid()
			sns.lineplot(x='year',y='NDSI_Snow_Cover',data=west_yrs,ax=ax,color='#d8b365')
			ax.set_title(f'Western river basins MODIS/VIIRS {plotting_param}')
			ax.set_xticks(dfs[0]['year'].unique()) 
			ax.set_xlabel(' ')
			
	
			ax.plot(np.unique(west_yrs['year']), np.poly1d(np.polyfit(west_yrs['year'], west_yrs['NDSI_Snow_Cover'], 1))(np.unique(west_yrs['year'])),color='red',linestyle='--')#,label='west')
			#ax.plot(np.unique(east_df[x_col]), np.poly1d(np.polyfit(east_df[x_col], east_df[y_col], 1))(np.unique(east_df[x_col])),color='blue',label='east')
			
			#linreg = rs_funcs.lin_reg_outputs(linreg_df,'year','NDSI_Snow_Cover')[0]
			#print(linreg)
			# east_linreg = rs_funcs.lin_reg_outputs(linreg_df_east,'year','NDSI_Snow_Cover')[0]

			# #Similarly the r-squared value: -
			# #ax.annotate(f'r2 = {round(west_linreg.rvalue,2)}',xy=(0.7,0.75),xycoords='figure fraction')
			# ax1.annotate(f'east r2 = {round(east_linreg.rvalue,2)}',xy=(0.7,0.7),xycoords='figure fraction')	 

			ax1.grid()
			#linreg_df_east = dfs[1].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			east_yrs_basins = dfs[1].groupby(['huc8','year'])['NDSI_Snow_Cover'].max().to_frame().reset_index()
			east_yrs = east_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()

			sns.lineplot(x='year',y='NDSI_Snow_Cover',data=east_yrs,ax=ax1,color='#5ab4ac')
			ax1.set_title(f'Eastern river basins MODIS/VIIRS {plotting_param}')
			ax1.set_xticks(dfs[1]['year'].unique())
			ax1.set_xlabel(' ')

			#add a linear regression line
			ax1.plot(np.unique(east_yrs['year']), np.poly1d(np.polyfit(east_yrs['year'], east_yrs['NDSI_Snow_Cover'], 1))(np.unique(east_yrs['year'])),color='red',linestyle='--')
			#east_linreg = rs_funcs.lin_reg_outputs(east_yrs,'year','NDSI_Snow_Cover')[0]

			#Similarly the r-squared value: -
			#ax.annotate(f'r2 = {round(west_linreg.rvalue,2)}',xy=(0.7,0.75),xycoords='figure fraction')
			#ax1.annotate(f'east r2 = {round(east_linreg.rvalue,2)}',xy=(0.25,0.25),xycoords='figure fraction')	 
			if plotting_param.upper() == 'SP': 
				ax.set_ylabel('DJF snow persistence')
				ax1.set_ylabel('DJF snow persistence')
			elif plotting_param.upper() == 'SCA': 
				ax.set_ylabel(f'{plotting_param} total (sq km)')
				ax1.set_ylabel(f'{plotting_param} total (sq km)')
			else: 
				print('Your plotting param seems incorrect, double check and try again.')
			
			#plt.plot(df.set_index('huc8').T)
			plt.show() 
			plt.close('all')
		else: 
			pass
if __name__ == '__main__':
    main()


# elif plot_type.lower() == 'long_term': #use to make plots of the long term trends in optical data 
# 			fig,(ax,ax1) = plt.subplots(2,1,sharex=True,sharey=True)
			
# 			print(dfs[0])
# 			ax.grid()
# 			sns.lineplot(x='year',y='NDSI_Snow_Cover',data=dfs[0],ax=ax,color='#d8b365')
# 			ax.set_title(f'Western river basins MODIS/VIIRS {plotting_param}')
# 			ax.set_xticks(dfs[0]['year'].unique()) 
# 			ax.set_xlabel(' ')
			
# 			linreg_df_west = dfs[0].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
	
# 			ax.plot(np.unique(linreg_df_west['year']), np.poly1d(np.polyfit(linreg_df_west['year'], linreg_df_west['NDSI_Snow_Cover'], 1))(np.unique(linreg_df_west['year'])),color='red',linestyle='--')#,label='west')
# 			#ax.plot(np.unique(east_df[x_col]), np.poly1d(np.polyfit(east_df[x_col], east_df[y_col], 1))(np.unique(east_df[x_col])),color='blue',label='east')
			
# 			#linreg = rs_funcs.lin_reg_outputs(linreg_df,'year','NDSI_Snow_Cover')[0]
# 			#print(linreg)
# 			# east_linreg = rs_funcs.lin_reg_outputs(linreg_df_east,'year','NDSI_Snow_Cover')[0]

# 			# #Similarly the r-squared value: -
# 			# #ax.annotate(f'r2 = {round(west_linreg.rvalue,2)}',xy=(0.7,0.75),xycoords='figure fraction')
# 			# ax1.annotate(f'east r2 = {round(east_linreg.rvalue,2)}',xy=(0.7,0.7),xycoords='figure fraction')	 

# 			ax1.grid()
# 			sns.lineplot(x='year',y='NDSI_Snow_Cover',data=dfs[1],ax=ax1,color='#5ab4ac')
# 			ax1.set_title(f'Eastern river basins MODIS/VIIRS {plotting_param}')
# 			ax1.set_xticks(dfs[1]['year'].unique())
# 			ax1.set_xlabel(' ')
# 			linreg_df_east = dfs[1].groupby('year')['NDSI_Snow_Cover'].mean().to_frame().reset_index()

# 			#add a linear regression line
# 			ax1.plot(np.unique(linreg_df_east['year']), np.poly1d(np.polyfit(linreg_df_east['year'], linreg_df_east['NDSI_Snow_Cover'], 1))(np.unique(linreg_df_east['year'])),color='red',linestyle='--')
# 			east_linreg = rs_funcs.lin_reg_outputs(linreg_df_east,'year','NDSI_Snow_Cover')[0]

# 			#Similarly the r-squared value: -
# 			#ax.annotate(f'r2 = {round(west_linreg.rvalue,2)}',xy=(0.7,0.75),xycoords='figure fraction')
# 			ax1.annotate(f'east r2 = {round(east_linreg.rvalue,2)}',xy=(0.7,0.7),xycoords='figure fraction')	 
# 			if plotting_param.upper() == 'SP': 
# 				ax.set_ylabel('DJF snow persistence')
# 				ax1.set_ylabel('DJF snow persistence')
# 			elif plotting_param.upper() == 'SCA': 
# 				ax.set_ylabel(f'{plotting_param} (sq km)')
# 				ax1.set_ylabel(f'{plotting_param} (sq km)')
# 			else: 
# 				print('Your plotting param seems incorrect, double check and try again.')
			
# 			#plt.plot(df.set_index('huc8').T)
# 			plt.show() 
# 			plt.close('all')
# 		else: 
# 			pass