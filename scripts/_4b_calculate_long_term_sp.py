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
from scipy import stats


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

		year = int(re.findall('(\d{4})-\d{2}-\d{2}', os.path.split(file)[1])[1]) #gets a list with the start and end of the water year, take the second one 
		#year = pd.to_datetime(df['date'].iloc[-1]).year #double check that this is working correctly. Suspicious that its not actually getting the water year 
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

def split_basins(input_df,grouping_col,**kwargs): 
	"""Read in a dir of landsat sp data from GEE and make into a new df."""
	input_df[grouping_col] = input_df[grouping_col].astype('str')
	west_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(western))]
	east_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(eastern))]
	
	#replace instances of inf with nan and drop the grouping_col so its not in the mean
	west_df.replace(np.inf,np.nan,inplace=True)
	east_df.replace(np.inf,np.nan,inplace=True)
	
	try: 
		west_df.drop(columns=[grouping_col,'elev_mean'],inplace=True) #added the hardcoded drop of the elev col to clean up for plotting
		east_df.drop(columns=[grouping_col,'elev_mean'],inplace=True)
	except Exception as e: 
		pass
		#print(e)
	west_df['year'] = kwargs.get('year')
	east_df['year'] = kwargs.get('year')
	west_mean = west_df.median(axis=0)
	east_mean = east_df.median(axis=0)

	return west_df,east_df


def merge_dfs(snotel_data,rs_data,huc_level,col_of_interest,elev_stat,plot_type): 
	"""Merge snotel snow drought data with RS data."""

	# if plot_type.upper() == 'SP': 
	# 		rs_data = rs_data.loc[rs_data['NDSI_Snow_Cover']>= 0.2]
	# else: 
	# 	print('Assuming this is SCA data and we will not threshold')
	print(rs_data.shape)
	dry_combined = _4a_rs.create_snow_drought_subset(snotel_data,'dry',huc_level)
	#merge em 
	dry_combined=dry_combined.merge(rs_data, on=['date','huc'+huc_level], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
	#get the rs data for the time periods of interest for a snow drought type 
	#dry_optical=dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].mean()
	dry_combined.rename(columns={col_of_interest:f'dry_{col_of_interest}'},inplace=True)
	dry_scaler = (dry_combined[f'dry_{col_of_interest}'].count()/rs_data.shape[0])
	dry_rs = dry_combined.groupby('huc'+huc_level)[f'dry_{col_of_interest}',elev_stat].max().reset_index() #changed col from pct change to filter 2/1/2021
	#dry_rs[f'dry_{col_of_interest}'] = dry_rs[f'dry_{col_of_interest}']*dry_scaler
	#then do warm 
	warm_combined = _4a_rs.create_snow_drought_subset(snotel_data,'warm',huc_level)
	#merge em 
	warm_combined=warm_combined.merge(rs_data, on=['date','huc'+huc_level], how='inner')
	#get the rs data for the time periods of interest for a snow drought type 
	#warm_optical=warm_combined.groupby('huc'+huc_level)['ndsi_pct_change'].min() 
	warm_combined.rename(columns={col_of_interest:f'warm_{col_of_interest}'},inplace=True)
	warm_scaler = (warm_combined[f'warm_{col_of_interest}'].count()/rs_data.shape[0])
	print('warm')
	print(warm_combined[f'warm_{col_of_interest}'].count())
	#print(warm_combined[['date',f'warm_{col_of_interest}','huc8']])
	warm_rs = warm_combined.groupby('huc'+huc_level)[f'warm_{col_of_interest}',elev_stat].max().reset_index()
	#warm_rs[f'warm_{col_of_interest}'] = warm_rs[f'warm_{col_of_interest}']*warm_scaler

	#then do warm/dry
	warm_dry_combined = _4a_rs.create_snow_drought_subset(snotel_data,'warm_dry',huc_level)
	#merge em 
	warm_dry_combined=warm_dry_combined.merge(rs_data, on=['date','huc'+huc_level], how='inner')

	#get the rs data for the time periods of interest for a snow drought type 
	#warm_dry_optical=warm_dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].sum()
	warm_dry_combined.rename(columns={col_of_interest:f'warm_dry_{col_of_interest}'},inplace=True)
	warm_dry_scaler = (warm_dry_combined[f'warm_dry_{col_of_interest}'].count()/rs_data.shape[0])
	#print(warm_dry_combined)
	warm_dry_rs = warm_dry_combined.groupby('huc'+huc_level)[f'warm_dry_{col_of_interest}',elev_stat].max().reset_index()
	#print(warm_dry_rs)
	#warm_dry_rs[f'warm_dry_{col_of_interest}'] = warm_dry_rs[f'warm_dry_{col_of_interest}']*warm_dry_scaler

	#try making a df of time steps that DO NOT have snow droughts for comparing
	no_snow_drought = _4a_rs.create_snow_drought_subset(snotel_data,'date',huc_level)
	no_drought_combined=no_snow_drought.merge(rs_data, on=['date','huc'+huc_level],how='inner')

	no_drought_combined.rename(columns={col_of_interest:f'no_drought_{col_of_interest}'},inplace=True)
	no_drought_scaler = (no_drought_combined[f'no_drought_{col_of_interest}'].count()/rs_data.shape[0])
	no_drought_rs = no_drought_combined.groupby('huc'+huc_level)[f'no_drought_{col_of_interest}'].max().reset_index()
	#no_drought_rs[f'no_drought_{col_of_interest}'] = no_drought_rs[f'no_drought_{col_of_interest}']*no_drought_scaler
	
	#original data
	#original_data = _4a_rs.create_snow_drought_subset(snotel_data,'date',huc_level)
	#no_drought_combined=no_snow_drought.merge(rs_data, on=['date','huc'+huc_level],how='inner')
	original_combined = rs_data
	original_combined.rename(columns={col_of_interest:f'original_{col_of_interest}'},inplace=True)
	#no_drought_scaler = (no_drought_combined[f'no_drought_{col_of_interest}'].count()/rs_data.shape[0])
	original_rs = original_combined.groupby('huc'+huc_level)[f'original_{col_of_interest}'].max().reset_index()
	#combine the dfs
	dfs = dry_rs.reset_index().merge(warm_rs.reset_index(),on=['huc'+huc_level],how='outer')
	dfs = dfs.merge(warm_dry_rs.reset_index(),on=['huc'+huc_level],how='outer')
	dfs.drop(columns={f'{elev_stat}_x',f'{elev_stat}_y'},inplace=True)
	#print('dfs shape',dfs.shape)
	
	dfs = dfs.merge(no_drought_rs.reset_index(),on=['huc'+huc_level],how='outer')
	#print(dfs)
	#print(dry_rs.reset_index())
	return dfs,dry_rs,warm_rs,warm_dry_rs,no_drought_rs,original_rs

def combine_rs_snotel_annually(input_dir,season,elev_stat,plotting_param,drought_type,pickles,agg_step,resolution=500,huc_level='8',col_of_interest='NDSI_Snow_Cover'): 
	"""Get rs data for the dates of different snow drought types from the SNOTEL record."""
	west_dfs_list = []
	east_dfs_list = []
	years = []
	optical_files = sorted(glob.glob(input_dir+'*.csv'))

	for file in optical_files: 
		#print(file)
		year = int(re.findall('(\d{4})-\d{2}-\d{2}', os.path.split(file)[1])[1]) #gets a list with the start and end of the water year, take the second one 

		print(year)

		if 'SP' in file:
			print('SP') 
			snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates_first_day_start'
		elif 'SCA' in file:
			print('SCA') 
			#decide which season length to use depending on the RS aggregation type (SP or SCA)
			snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
		else: 
			print('Your file contains neither sp nor SCA, try again')

		input_data = obtain_data.AcquireData(None,file,snotel_data,None,huc_level,resolution)
		
		short_term_snow_drought = input_data.get_snotel_data()
		optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
		optical_data[f'huc{huc_level}'] = pd.to_numeric(optical_data['huc'+huc_level]) 
		optical_data['date'] = _4a_rs.convert_date(optical_data,'date')

		#convert pixel counts to area
		optical_data=rs_funcs.convert_pixel_count_sq_km(optical_data,col_of_interest,resolution)

		#combine the remote sensing and snotel data using the snotel dates of snow droughts to extract rs data 
		#split the dfs by time period in the winter 
		
		early_df = optical_data.loc[(optical_data['date']>= f'{year-1}-11-01') & (optical_data['date']<f'{year-1}-12-31')] #optical_data[optical_data["date"].isin(pd.date_range(start_date, end_date))]
		mid_df = optical_data.loc[(optical_data['date']>= f'{year}-01-01') & (optical_data['date']<f'{year}-03-01')]
		late_df = optical_data.loc[(optical_data['date']>= f'{year}-03-02') & (optical_data['date']<f'{year}-05-01')]
		
		merged=merge_dfs(short_term_snow_drought,early_df,huc_level,'NDSI_Snow_Cover',elev_stat,plotting_param)[drought_type]	#drought type is passed as an integer starting at one 
		
		split_dfs=split_basins(merged,f'huc{huc_level}',year=year) #returns the merged df split into two dfs, west (0) and east (1)

		west_dfs_list.append(split_dfs[0])#.T)#.to_frame().T)
		east_dfs_list.append(split_dfs[1])#.T)#.to_frame().T)
		
	output_west_df = pd.concat(west_dfs_list,ignore_index=True)
	output_east_df = pd.concat(east_dfs_list,ignore_index=True)

	grouping_col = [i for i in output_west_df.columns if 'NDSI_Snow_Cover' in i][0]


	output_west_df = output_west_df.groupby('year')[grouping_col].sum().reset_index()#.to_frame().reset_index()
	output_east_df = output_east_df.groupby('year')[grouping_col].sum().reset_index()#.to_frame().reset_index()
	
	# output_west_df['year']=years
	# output_east_df['year']=years
	
	# output_west_df = output_west_df.set_index('year')
	# output_east_df = output_east_df.set_index('year')

	return output_west_df,output_east_df

def get_anom_col(input_df,mean_col,skip_col='year'): 
	"""Helper function."""
	long_term_mean=input_df[mean_col].mean()
	print('mean ',long_term_mean)
	#input_df['mean'] = long_term_mean
	for column in input_df.columns: 
		if not (column == mean_col) | (column == skip_col) | (column == 'mean'): 

			print('column is: ', column)
			input_df[column] = input_df[column]-long_term_mean

		elif column == mean_col: 
			pass
	return input_df

def make_long_term_sp_SCA_plots(): 
	"""UPDATE."""
	   #working to make the long-term plots of SCA or SP 
  #   #################################################
  #   font = {'family' : 'normal',
  #       	'weight' : 'normal',
  #       	'size'   : 18}

		# 	plt.rc('font', **font)
		# 	fig,(ax,ax1) = plt.subplots(2,1,sharex=True,sharey=True)

		# 	#decide if we're plotting SP or SCA 
			
		# 	if plotting_param.upper() == 'SP': 
		# 		print('plotting SP')
		# 		linewidth=2.5
		# 		west_yrs=dfs[0]
		# 		east_yrs=dfs[1]

		# 	elif plotting_param.upper() == 'SCA': 
		# 		print('plotting SCA')
		# 		linewidth=3.5
		# 		west_yrs_basins = dfs[0].groupby(['huc8','year'])['NDSI_Snow_Cover'].median().to_frame().reset_index()
		# 		west_yrs = west_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()

		# 		east_yrs_basins = dfs[1].groupby(['huc8','year'])['NDSI_Snow_Cover'].median().to_frame().reset_index()
		# 		east_yrs = east_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			
		# 	else: 
		# 		print('Your plotting param seems incorrect, double check and try again.')
		# 	#linreg_df_west = dfs[0].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()

		# 	#get coeffs of linear fit
		# 	#slope, intercept, r_value, p_value, std_err = stats.linregress(west_yrs['year'],west_yrs['NDSI_Snow_Cover'])
		# 	print(west_yrs['NDSI_Snow_Cover'].mean())
		# 	print(east_yrs['NDSI_Snow_Cover'].mean())
		# 	ax.grid()
		# 	sns.lineplot(x='year',y='NDSI_Snow_Cover',data=west_yrs,ax=ax,color='#565656',linewidth=linewidth)
		# 	ax.set_title(f'Western river basins MODVII {plotting_param}')
		# 	ax.set_xticks(dfs[0]['year'].unique()) 
		# 	ax.set_xlabel(' ')

		# 	ax1.grid()
		# 	#linreg_df_east = dfs[1].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			

		# 	sns.lineplot(x='year',y='NDSI_Snow_Cover',data=east_yrs,ax=ax1,color='#565656',linewidth=linewidth)
		# 	ax1.set_title(f'Eastern river basins MODVII {plotting_param}')
		# 	ax1.set_xticks(dfs[1]['year'].unique())

		# 	ax1.set_xlabel(' ')

		# 	#add ylabels 
		# 	if plotting_param.upper() == 'SP': 
		# 		ax.set_ylabel('DJF snow persistence')
		# 		ax1.set_ylabel('DJF snow persistence')
		# 	elif plotting_param.upper() == 'SCA': 
		# 		ax.set_ylabel(f'{plotting_param} total (sq km)')
		# 		ax1.set_ylabel(f'{plotting_param} total (sq km)')
		# 	plt.xticks(rotation=45)
		# 	plt.show() 
		# 	plt.close('all')
		# else: 
		# 	pass


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
		plot_type = 'long_term' #one of drought_type,long-term or something else 
		plotting_param = 'SCA'
		#plot_func = 'quartile'
		elev_stat = 'elev_mean'
		

		#dfs = read_in_and_reformat_data(csv_dir,'huc8','NDSI_Snow_Cover',['.geo','system:index','elev_min','elev_max'],resolution,plotting_param,plot_type)
		#dfs = read_in_and_reformat_data(csv_dir,'huc8','NDSI_Snow_Cover',['.geo','system:index'],resolution,plotting_param,plot_type)
		
		#print('sp_data')
		#print(dfs)
		if not (plot_type == 'long_term') | (plot_type == 'drought_type'): 
			fig,(ax,ax1)=plt.subplots(2)
			sns.lineplot(dfs[0]['year'],dfs[0]['NDSI_Snow_Cover'],ax=ax)
			sns.lineplot(dfs[1]['year'],dfs[1]['NDSI_Snow_Cover'],ax=ax1)

			plt.show()
			plt.close()

		elif plot_type == 'drought_type': 
			west_dfs_list = []
			east_dfs_list = []
			years = []
			optical_files = sorted(glob.glob(csv_dir+'*.csv'))
			for file in optical_files: 
				print(file)
				year = int(re.findall('(\d{4})-\d{2}-\d{2}', os.path.split(file)[1])[1]) #gets a list with the start and end of the water year, take the second one 
				print(year)
				
				if 'SP' in file:
					print('SP') 
					snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates_first_day_start'
				elif 'SCA' in file:
					print('SCA') 
					snotel_data = pickles+f'short_term_snow_drought_{year_of_interest}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
				else: 
					print('Your file contains neither sp nor SCA, try again')

				input_data = obtain_data.AcquireData(None,file,snotel_data,None,huc_level,resolution)
				
				short_term_snow_drought = input_data.get_snotel_data()
				print(short_term_snow_drought)
				optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
				#optical_data = dfs[1]
				optical_data[f'huc{huc_level}'] = pd.to_numeric(optical_data['huc'+huc_level]) 
				optical_data['date'] = _4a_rs.convert_date(optical_data,'date')

				#combine the remote sensing and snotel data using the snotel dates of snow droughts to extract rs data 
				merged=merge_dfs(short_term_snow_drought,optical_data,huc_level,'NDSI_Snow_Cover',elev_stat,plotting_param)
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
			#print(output_west_df)
			#plot it
			font = {'family' : 'normal',
        	'weight' : 'normal',
        	'size'   : 18}

			plt.rc('font', **font)
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
			ax.set_title('Western Basins SP by drought type')
			ax.set_xlabel(' ')
			ax.set_ylabel('SP')
			ax.grid()
			ax.legend(labels=labels)
			ax.tick_params(axis='x', labelsize=15)

			
			output_east_df.plot(ax=ax1,color=palette,linewidth=3.5,legend=False)
			ax1.set_xticks(years)
			ax1.set_title('Eastern Basins SP by drought type')
			ax1.set_xlabel(' ')
			ax1.set_ylabel('SP')
			ax1.grid()

			plt.tight_layout()
			plt.show()
			plt.close('all')

		elif plot_type.lower() == 'long_term': #use to make plots of the long term trends in optical data 
			font = {'family' : 'normal',
        	'weight' : 'normal',
        	'size'   : 16}

			plt.rc('font', **font)
			fig,(ax,ax1) = plt.subplots(2,1,sharex=True)

			#decide if we're plotting SP or SCA 
			
			if plotting_param.upper() == 'SP': 
				print('plotting SP')
				linewidth=2.5
				west_yrs=dfs[0]
				east_yrs=dfs[1]

			elif plotting_param.upper() == 'SCA': 
				print('plotting SCA')
				drought_types = ['dry','warm','warm_dry','no_drought','original']
				count = 1 #this is used to index the output of merge_dfs function above 
				drought_dict = {}
				for i in drought_types: 
					drought_dfs=combine_rs_snotel_annually(csv_dir,season,elev_stat,plotting_param,count,pickles,agg_step)
					count +=1 
					drought_dict.update({i:drought_dfs}) #make a dict that is of the type {'dry':(west_df,east_df)}
				#print(drought_dict)

				linewidth=3.5
				#print(dfs[0])
				# west_yrs_basins = dfs[0].groupby(['huc8','year'])['NDSI_Snow_Cover'].max().to_frame().reset_index()
				# #print(west_yrs_basins)
				# west_yrs = west_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
				# #print(west_yrs)

				# east_yrs_basins = dfs[1].groupby(['huc8','year'])['NDSI_Snow_Cover'].max().to_frame().reset_index()
				# east_yrs = east_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			
			else: 
				print('Your plotting param seems incorrect, double check and try again.')
			#linreg_df_west = dfs[0].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()

			#get coeffs of linear fit
			#slope, intercept, r_value, p_value, std_err = stats.linregress(west_yrs['year'],west_yrs['NDSI_Snow_Cover'])
			#print(west_yrs['NDSI_Snow_Cover'].mean())

			#make combined dfs 
			 
			plot_west = pd.concat((drought_dict['original'][0],drought_dict['dry'][0],drought_dict['warm'][0],drought_dict['warm_dry'][0]),axis=1)#,drought_dict['no_drought'][0]),axis=1)
			plot_east = pd.concat((drought_dict['original'][1],drought_dict['dry'][1],drought_dict['warm'][1],drought_dict['warm_dry'][1]),axis=1)#,drought_dict['no_drought'][1]),axis=1)
			plot_west = plot_west.sort_index(axis=1).iloc[:,:5]  #this is hardcoded and assumes that 'year' which is the duplicate will come at the end. Note that it must be changed if including no drought 

			plot_east = plot_east.sort_index(axis=1).iloc[:,:5]
			#plot_west['mean'] = plot_west['NDSI_Snow_Cover']
			#plot_east['mean'] = plot_east['NDSI_Snow_Cover']
			# print('original')
			# print(plot_west)
			# print(plot_east)
			plot_west_anom = (get_anom_col(plot_west,'original_NDSI_Snow_Cover','year'))
			plot_east_anom = (get_anom_col(plot_east,'original_NDSI_Snow_Cover','year'))

			plot_west_anom = plot_west_anom.drop(['original_NDSI_Snow_Cover'],axis=1)
			plot_east_anom = plot_east_anom.drop(['original_NDSI_Snow_Cover'],axis=1)
			plot_west_anom.rename(columns={'dry_NDSI_Snow_Cover':'Dry','warm_NDSI_Snow_Cover':'Warm','warm_dry_NDSI_Snow_Cover':'Warm dry'},inplace=True)
			plot_east_anom.rename(columns={'dry_NDSI_Snow_Cover':'Dry','warm_NDSI_Snow_Cover':'Warm','warm_dry_NDSI_Snow_Cover':'Warm dry'},inplace=True)
			# print('anomoly')
			# print(plot_west_anom)
			# print(plot_east_anom)
			#print(plot_west)
			#print(plot_east.groupby('year').first())
			#print(plot_east.columns)
			#print(east_yrs['NDSI_Snow_Cover'].mean())
			ax.grid()


			# Put data in long format in a dataframe.
			# df = pd.DataFrame({
			#     'country': countries2012 + countries2013,
			#     'year': ['2012'] * len(countries2012) + ['2013'] * len(countries2013),
			#     'percentage': percentage2012 + percentage2013
			# })

			# One liner to create a stacked bar chart.
			#plot_west.plot.bar(stacked=False, x='year',ax=ax)

			# ax = sns.histplot(df, x='year', hue='country', weights='percentage',
			#              multiple='stack', palette='tab20c', shrink=0.8)
			#ax.set_ylabel('percentage')
			# Fix the legend so it's not on top of the bars.
			#legend = ax.get_legend()
			#legend.set_bbox_to_anchor((1, 1))
			#sns.lineplot(data=plot_west_anom,ax=ax,linewidth=linewidth)
			plot_west_anom.plot.bar(x='year',ax=ax)#,color=palette.values())
			# sns.lineplot(x='year',y='NDSI_Snow_Cover',data=west_yrs,ax=ax,color='#565656',linewidth=linewidth)
			# count = 0
			# for k,v in drought_dict.items(): 
			# 	plotting_col = [i for i in v[0].columns if 'NDSI_Snow_Cover' in i][0]
			# 	print(plotting_col)
			# 	sns.lineplot(x='year',y=plotting_col,data=v[0],ax=ax,color=palette[drought_types[count]],linewidth=linewidth)
			# 	count += 1 

			ax.set_title(f'Western river basins MODVII {plotting_param}')
			#ax.set_xticks(dfs[0]['year'].unique()) 
			ax.set_xlabel(' ')

			ax1.grid()
			#linreg_df_east = dfs[1].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			plot_east_anom.plot.bar(x='year',ax=ax1,legend=False)#,color=palette.values())

			#plot_east.plot.bar(stacked=False, x='year',ax=ax1,legend=False)
			# east_yrs['mean'] = east_yrs['NDSI_Snow_Cover'].mean()
			# sns.lineplot(x='year',y='NDSI_Snow_Cover',data=east_yrs,ax=ax1,color='#565656',linewidth=linewidth)
			# sns.lineplot(x='year',y='mean',data=east_yrs,ax=ax1,color='#000000',linewidth=linewidth)

			# count = 0 
			# for k,v in drought_dict.items(): 
			# 	plotting_col = [i for i in v[0].columns if 'NDSI_Snow_Cover' in i][0]
			# 	sns.lineplot(x='year',y=plotting_col,data=v[1],ax=ax1,color=palette[drought_types[count]],linewidth=linewidth)
			# 	count +=1
			ax1.set_title(f'Eastern river basins MODVII {plotting_param}')
			#ax1.set_xticks(dfs[1]['year'].unique())

			ax1.set_xlabel(' ')

			#add ylabels 
			if plotting_param.upper() == 'SP': 
				ax.set_ylabel('DJF snow persistence')
				ax1.set_ylabel('DJF snow persistence')
			elif plotting_param.upper() == 'SCA': 
				ax.set_ylabel(f'{plotting_param} total (sq km)')
				ax1.set_ylabel(f'{plotting_param} total (sq km)')
			plt.xticks(rotation=45)
			plt.show() 
			plt.close('all')
		else: 
			pass
if __name__ == '__main__':
    main()
    #working to make the long-term plots of SCA or SP 
  #   #################################################
  #   font = {'family' : 'normal',
  #       	'weight' : 'normal',
  #       	'size'   : 18}

		# 	plt.rc('font', **font)
		# 	fig,(ax,ax1) = plt.subplots(2,1,sharex=True,sharey=True)

		# 	#decide if we're plotting SP or SCA 
			
		# 	if plotting_param.upper() == 'SP': 
		# 		print('plotting SP')
		# 		linewidth=2.5
		# 		west_yrs=dfs[0]
		# 		east_yrs=dfs[1]

		# 	elif plotting_param.upper() == 'SCA': 
		# 		print('plotting SCA')
		# 		linewidth=3.5
		# 		west_yrs_basins = dfs[0].groupby(['huc8','year'])['NDSI_Snow_Cover'].median().to_frame().reset_index()
		# 		west_yrs = west_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()

		# 		east_yrs_basins = dfs[1].groupby(['huc8','year'])['NDSI_Snow_Cover'].median().to_frame().reset_index()
		# 		east_yrs = east_yrs_basins.groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			
		# 	else: 
		# 		print('Your plotting param seems incorrect, double check and try again.')
		# 	#linreg_df_west = dfs[0].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()

		# 	#get coeffs of linear fit
		# 	#slope, intercept, r_value, p_value, std_err = stats.linregress(west_yrs['year'],west_yrs['NDSI_Snow_Cover'])
		# 	print(west_yrs['NDSI_Snow_Cover'].mean())
		# 	print(east_yrs['NDSI_Snow_Cover'].mean())
		# 	ax.grid()
		# 	sns.lineplot(x='year',y='NDSI_Snow_Cover',data=west_yrs,ax=ax,color='#565656',linewidth=linewidth)
		# 	ax.set_title(f'Western river basins MODVII {plotting_param}')
		# 	ax.set_xticks(dfs[0]['year'].unique()) 
		# 	ax.set_xlabel(' ')

		# 	ax1.grid()
		# 	#linreg_df_east = dfs[1].groupby('year')['NDSI_Snow_Cover'].sum().to_frame().reset_index()
			

		# 	sns.lineplot(x='year',y='NDSI_Snow_Cover',data=east_yrs,ax=ax1,color='#565656',linewidth=linewidth)
		# 	ax1.set_title(f'Eastern river basins MODVII {plotting_param}')
		# 	ax1.set_xticks(dfs[1]['year'].unique())

		# 	ax1.set_xlabel(' ')

		# 	#add ylabels 
		# 	if plotting_param.upper() == 'SP': 
		# 		ax.set_ylabel('DJF snow persistence')
		# 		ax1.set_ylabel('DJF snow persistence')
		# 	elif plotting_param.upper() == 'SCA': 
		# 		ax.set_ylabel(f'{plotting_param} total (sq km)')
		# 		ax1.set_ylabel(f'{plotting_param} total (sq km)')
		# 	plt.xticks(rotation=45)
		# 	plt.show() 
		# 	plt.close('all')
		# else: 
		# 	pass

	####################################################################################
   #ax.plot(np.unique(west_yrs['year']), np.poly1d(np.polyfit(west_yrs['year'], west_yrs['NDSI_Snow_Cover'], 1))(np.unique(west_yrs['year'])),color='red',linestyle='--')#,label='west')
			#ax.plot(np.unique(east_df[x_col]), np.poly1d(np.polyfit(east_df[x_col], east_df[y_col], 1))(np.unique(east_df[x_col])),color='blue',label='east')
			
			#linreg = rs_funcs.lin_reg_outputs(linreg_df,'year','NDSI_Snow_Cover')[0]
			#west_linreg = rs_funcs.lin_reg_outputs(west_yrs,'year','NDSI_Snow_Cover')[0]

			# # #Similarly the r-squared value: -
			# ax1.annotate(f'east r2 = {round(east_linreg.rvalue,2)}',xy=(0.7,0.7),xycoords='figure fraction')	 

#add a linear regression line
			#ax1.plot(np.unique(east_yrs['year']), np.poly1d(np.polyfit(east_yrs['year'], east_yrs['NDSI_Snow_Cover'], 1))(np.unique(east_yrs['year'])),color='red',linestyle='--')
			#east_linreg = rs_funcs.lin_reg_outputs(east_yrs,'year','NDSI_Snow_Cover')[0]

			#Similarly the r-squared value: -
			#ax.annotate(f'r2 = {round(west_linreg.rvalue,2)}',xy=(0.7,0.75),xycoords='figure fraction')
			#ax1.annotate(f'east r2 = {round(east_linreg.rvalue,2)}',xy=(0.25,0.25),xycoords='figure fraction')	 
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