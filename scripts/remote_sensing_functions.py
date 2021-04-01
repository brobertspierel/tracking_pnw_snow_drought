import os 
import sys
import glob
import numpy as np 
import pandas as pd 
import json
import datetime
import scipy as sp
import statsmodels.api as sm

##################################################################################################################
##################################General Functions###############################################################
##################################################################################################################

def read_csv(input_csv,data_source): 
	"""Read in remote sensing csvs from GEE and make sure dates are properly formatted."""

	df = pd.read_csv(input_csv)

	if data_source.lower() == 'optical': 
		df.rename(columns={'system:time_start':'date'},inplace=True)
		df['date'] = pd.to_datetime(df['date'])
	
	elif data_source.lower() == 'sentinel': 
		print('processing sentinel')
		df['date'] = pd.to_datetime(df['date'])
	else: 
		print('Processing something that is not sar or optical or has no date column')
	
	return df

def convert_pixel_count_sq_km(input_df,data_field,resolution): 
	"""Convert pixel counts for remote sensing data to sq kms."""

	input_df[data_field]=(input_df[data_field]*int(resolution)*int(resolution))/1000000
	return input_df

def merge_remote_sensing_data(optical_data,sentinel_data): 
	"""Combine remote sensing datasets."""

	try: 
		optical_data.drop(columns=['.geo'], inplace=True)
	
	except KeyError: 
		print('Optical df does not have the geo column')
	try: 
		sentinel_data.drop(columns=['.geo'], inplace=True)
	
	except KeyError: 
		print('sentinel df does not have the geo column')
	
	try: 
		merged_df = optical_data.merge(sentinel_data, how = 'inner', on = ['huc8','date']) #hardcoded
	
	except Exception as e: 
		print(f'Tried to merge optical and sar data but got this {e} error message.')

	else: 
		sentinel_data
	
	return merged_df

def lin_reg_outputs(input_df,x_col,y_col): 
	"""Produce a linear regression for a time series."""

	linreg = sp.stats.linregress(input_df[x_col],input_df[y_col])
	X2 = sm.add_constant(input_df[x_col])
	est = sm.OLS(input_df[y_col], X2)
	f_value=est.fit().f_pvalue
	return linreg,f_value
##################################################################################################################
##################################Sentinel specific functions#####################################################
##################################################################################################################

def combine_hucs_w_sentinel(hucs_data,sentinel_data,huc_level,resolution,col_of_interest='filter'): 
	"""
	Read in sentinel 1 wet snow area and hucs data, normalize wet snow area by hucs area.
	"""
	#read in data
	sentinel_df=read_csv(sentinel_data,'sentinel')
	hucs_df = read_csv(hucs_data,'hucs')

	#convert the pixel counts to area
	sentinel_df=convert_pixel_count_sq_km(sentinel_df,col_of_interest,resolution)
	
	#create a dict of hucs ids and areas of basins 
	hucs_dict = hucs_df.set_index('id').to_dict()['area']
	#create a new area col
	sentinel_df['area'] = sentinel_df[f'huc{huc_level}'].map(hucs_dict)
	#normalize snow covered area by basin area
	sentinel_df['snow_ratio'] = sentinel_df[col_of_interest]/sentinel_df['area']
	
	return sentinel_df

##################################################################################################################
##################################MODIS/VIIRS specific functions##################################################
##################################################################################################################
def read_optical_and_convert_to_area(optical_data,resolution,data_source,col_of_interest): 
	"""
	Read in optical data (generally MODIS/VIIRS) and convert pixel counts to area.
	"""
	df = read_csv(optical_data,data_source)
	# if not 'SP' in optical_data: 
	# 	df = convert_pixel_count_sq_km(df,col_of_interest,resolution)
	return df 


##################################################################################################################
##################################Landsat specific functions######################################################
##################################################################################################################