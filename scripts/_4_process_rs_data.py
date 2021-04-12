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
import _3_obtain_all_data as obtain_data
import re
import math 
from scipy import stats
from functools import reduce
import _pickle as cPickle
#class CleanOptialRSData(): 

western = ['1708','1801','1710','1711','1709']
eastern = ['1701','1702','1705','1703','1601','1707','1706','1712','1704']

def convert_date(input_df,col_of_interest): 
	"""Helper function."""
	input_df[col_of_interest] = pd.to_datetime(input_df[col_of_interest],errors='coerce')
	return input_df[col_of_interest]

def create_snow_drought_subset(input_df,col_of_interest,huc_level): 
	"""Helper function."""

	drought_list = ['dry','warm','warm_dry','date']
	try: 
		drought_list.remove(col_of_interest)
	except Exception as e: 
		print(f'Error was: {e}')
	df = input_df.drop(columns=drought_list)
	
	df['huc_id'] = df['huc_id'].astype('int')
	
	df[col_of_interest] = convert_date(df,col_of_interest)
	
	#rename cols to match rs data for ease 
	df.rename(columns={col_of_interest:'date','huc_id':'huc'+huc_level},inplace=True)
	#get rid of na fields
	
	df = df.dropna()

	return df

def split_basins(input_df,grouping_col,**kwargs): 
	"""Read in a dir of landsat sp data from GEE and make into a new df."""
	input_df[grouping_col] = input_df[grouping_col].astype('str')
	west_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(western))]
	east_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(eastern))]
	
	#replace instances of inf with nan and drop the grouping_col so its not in the mean
	west_df.replace(np.inf,np.nan,inplace=True)
	east_df.replace(np.inf,np.nan,inplace=True)
	
	# try: #commented out 3/24/2021 might need to uncomment for plotting  
	# 	west_df.drop(columns=[grouping_col,'elev_mean'],inplace=True) #added the hardcoded drop of the elev col to clean up for plotting
	# 	east_df.drop(columns=[grouping_col,'elev_mean'],inplace=True)
	# except Exception as e: 
	# 	pass
		#print(e)
	# west_df['year'] = kwargs.get('year')
	# east_df['year'] = kwargs.get('year')
	# west_mean = west_df.median(axis=0)
	# east_mean = east_df.median(axis=0)

	return west_df,east_df

def split_dfs_within_winter_season(df,region,sp=False): 
	"""Splits a single df by date ranges in a winter season."""
	
	early_df = df.loc[(df['date'].dt.month>=11)] 
	mid_df = df.loc[(df['date'].dt.month>=1)&(df['date'].dt.month<=2)]
	late_df = df.loc[(df['date'].dt.month>=3)&(df['date'].dt.month<=4)]
	if region: 
		return {region:[early_df,mid_df,late_df]}
	else: 
		return [early_df,mid_df,late_df]

def merge_dfs(snotel_data,rs_data,drought_type,huc_level='8',col_of_interest='NDSI_Snow_Cover',resolution=500,**kwargs): #added drought_type arg so the drought type is supplied externally 3/15/2021
	"""Merge snotel snow drought data with RS data."""
	#deal with a circumstance where there is SAR and optical data coming in 
	if 'sar_data' in kwargs: 
		rs_data=kwargs.get('sar_data')
		#make sure that the cols used for merging are homogeneous in type 
		rs_data['huc8'] = pd.to_numeric(rs_data['huc'+huc_level])
		rs_data['date'] = convert_date(rs_data,'date')
		# try: 
		# 	sar_data.drop(columns=['elev_min','elev_mean','elev_max'],inplace=True)
		# except KeyError as e: 
		# 	pass 
		#rs_data=rs_funcs.merge_remote_sensing_data(rs_data,sar_data)
		
		#rs_data = rs_funcs.convert_pixel_count_sq_km(rs_data,'NDSI_Snow_Cover',resolution)
		#rs_data[f'{drought_type}_WSCA'] = rs_data['filter']/rs_data['NDSI_Snow_Cover'] #calculate wet snow as fraction of snow covered area
		if drought_type.lower() == 'total': 
			rs_data.rename(columns={col_of_interest:f'{drought_type}_{col_of_interest}'},inplace=True)
			
			return rs_data
	
	combined = create_snow_drought_subset(snotel_data,drought_type,huc_level)
	if not f'huc{huc_level}' in combined.columns: 
		combined.rename(columns={'huc_id':f'huc_{huc_level}'},inplace=True)
		combined[f'huc_{huc_level}'] = pd.to_numeric(rs_data['huc'+huc_level])

	# print(combined.dtypes)
	# print(rs_data.dtypes)
	# #merge em 
	# print('rs data: ',rs_data.date)
	# print('combined',combined.date)
	combined=combined.merge(rs_data, on=['date',f'huc{huc_level}'], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
	#get the rs data for the time periods of interest for a snow drought type 
	combined.rename(columns={col_of_interest:f'{drought_type}_{col_of_interest}'},inplace=True)
	#combined = combined.groupby([f'huc{huc_level}', 'date'])[f'{drought_type}_{col_of_interest}'].transform(max) #doesn't really matter which stat (max,min,first) because they are all the same 
	# if 'sar_data' in kwargs: 
	# 	combined = combined.sort_values(f'{drought_type}_').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')
	#else:
	combined = combined.sort_values(f'{drought_type}_{col_of_interest}').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')
	# print('combined')
	# print(combined)
	#check if a couple of args are in kwargs, they can be anything that will evaluate to True
	if 'groupby' in kwargs: 
		rs_df = combined.groupby('date')[f'{drought_type}_{col_of_interest}'].sum().reset_index()
		#dry_rs = dry_combined.groupby('huc'+huc_level)[f'dry_{col_of_interest}',elev_stat].max().reset_index() #changed col from pct change to filter 2/1/2021

		if 'scale_it' in kwargs: 
			scaler = (combined[f'{drought_type}_{col_of_interest}'].count()/rs_data.shape[0])
			rs_df[f'{drought_type}_{col_of_interest}'] = rs_df[f'{drought_type}_{col_of_interest}']*scaler

		return rs_df

	else: 
		
		return combined

def combine_rs_snotel_annually(input_dir,season,pickles,agg_step=12,resolution=500,huc_level='8',col_of_interest='NDSI_Snow_Cover',elev_stat='elev_mean',sp=False,total=False,split=True,**kwargs): 
	"""Get RS data for snow drought time steps and return those data split by region."""
	
	west_dfs_list = []
	east_dfs_list = []
	years = []
	optical_files = sorted(glob.glob(input_dir+'*.csv'))

	for file in optical_files: 
		year = re.findall('(\d{4})', os.path.split(file)[1])[1] #gets a list with the start and end of the water year, take the second one. expects files to be formatted a specific way from GEE 
		#print(year)
		#decide which season length to use depending on the RS aggregation type (SP or SCA)
		if 'SP' in file: 
			snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates_first_day_start'
		elif 'SCA' in file:
			snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
		else: 
			print('Your file contains neither sp nor SCA, try again')

		input_data = obtain_data.AcquireData(None,file,snotel_data,None,huc_level,resolution)
		
		short_term_snow_drought = input_data.get_snotel_data()
		optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
		optical_data[f'huc{huc_level}'] = pd.to_numeric(optical_data['huc'+huc_level]) 
		optical_data['date'] = convert_date(optical_data,'date')
			
		# if 'sar_data' in kwargs: 
		# 	sar_data = input_data.get_sentinel_data('filter')


		#convert pixel counts to area
		if not sp: 
			optical_data=rs_funcs.convert_pixel_count_sq_km(optical_data,col_of_interest,resolution)
			optical_data['area'] = optical_data[f'huc{huc_level}'].map(kwargs.get('hucs_data'))
			#normalize snow covered area by basin area
			optical_data[col_of_interest] = optical_data[col_of_interest]/optical_data['area'] #changed 4/9/2021 to update the NDSI_Snow_Cover col in place 
		#optical_data['year'] = optical_data['date'].dt.year

		if not total: 
			#combine the remote sensing and snotel data using the snotel dates of snow droughts to extract rs data 
			merged=merge_dfs(short_term_snow_drought,optical_data,kwargs.get('drought_type')) #snotel_data,rs_data,drought_type
			# print('merged')
			# print(merged)
			# try: 
			# 	print(merged[['NDSI_Snow_Cover','area']])
			# except Exception as e: 
			# 	pass
		else: 
			pass
			#print('Calculating total with no snow droughts')
		#output = split_dfs_within_winter_season
		try: 
			split_dfs=split_basins(merged,f'huc{huc_level}',year=year) #returns the merged df split into two dfs, west (0) and east (1)
		
		except UnboundLocalError as e: 
			print('stopped here')
			split_dfs=split_basins(optical_data,f'huc{huc_level}',year=year) #returns the merged df split into two dfs, west (0) and east (1)
			print(split_dfs)
		
		west_dfs_list.append(split_dfs[0])
		east_dfs_list.append(split_dfs[1])
		
	print('list',west_dfs_list)
	output_west_df = pd.concat(west_dfs_list,ignore_index=True)
	output_east_df = pd.concat(east_dfs_list,ignore_index=True)
	try: 
		if split: 
			return output_west_df,output_east_df #returns two dfs, one for each region for all the years for one drought type 
		else: 
			return merged 
	except UnboundLocalError as e: 
		return optical_data

def get_anom_col(input_df,base_col,skip_col='year'): #not currently in use 3/15/2021
	"""Helper function."""
	#long_term_mean=input_df[mean_col].mean()
	#input_df['mean'] = long_term_mean
	for column in input_df.columns: 
		if not (column == base_col) | (column == skip_col) | (column == 'mean'): 

			input_df[column] = (input_df[column]/input_df[base_col])#*100
	input_df.drop(columns=[base_col],inplace=True)
	return input_df

def generate_output(input_data,sp=False):
	output = {} 
	try: 
		for i,j in zip(input_data,['west','east']): 
			if not sp: 
				chunk = split_dfs_within_winter_season(i,j)
				output.update(chunk)
			else: 
				output.update({j:[i]})
		return output
	except Exception as e: #if this is merged or straight output without changes it will just be a dict and not df and therefore not iterable 
		split_dfs_within_winter_season(input_data)

def aggregate_dfs(input_dict,index,region,drought_type,sp=False): 
	"""Helper function for plotting."""

	#get the sum of the dates in one period- this enables us to pick the time when snow extent is greatest across the AOI
	
	if not sp: #function defaults to working on SCA, need to specify sp = True to run that 
		output = input_dict[region][index].groupby('date')[f'{drought_type}NDSI_Snow_Cover'].sum().reset_index()
		output['year'] = output['date'].dt.year	
		#get the date when snow extent is at its max for that period- maybe mean or median makes more sense? 
		output = output.groupby('year')[f'{drought_type}NDSI_Snow_Cover'].max().reset_index() #change if a different stat is introduced above

	else: 
		output = input_dict[region][index].groupby('date')[f'{drought_type}NDSI_Snow_Cover'].median().reset_index()
		output['year'] = output['date'].dt.year	
		output = output.groupby('year')[f'{drought_type}NDSI_Snow_Cover'].min().reset_index() #not sure about this stat- we're taking the median of dates and then the median of those? maybe max is better? 		

	
	#rename the col so the legend looks better
	output.rename(columns={f'{drought_type}NDSI_Snow_Cover':drought_type.title()})

	return output

def combine_sar_data(dry,warm,warm_dry,total,index): 
	"""Helper function for the plotting script."""
	try: 
		if 'dry_filter' in dry.columns: 
			sar_data = reduce(lambda x,y: pd.merge(x,y,on=['huc8'],
				how='inner'),[dry[['dry_filter','huc8']],warm[['warm_filter','huc8']],warm_dry[['warm_dry_filter','huc8']],total[['total_filter','huc8']]]).fillna(np.nan)
			return sar_data.set_index('huc8')
	except TypeError as e: 
		pass
	else: 
		optical_data=pd.concat([dry[index]['dry_NDSI_Snow_Cover'],warm[index]['warm_NDSI_Snow_Cover'],warm_dry[index]['warm_dry_NDSI_Snow_Cover']],axis=1)
		return optical_data