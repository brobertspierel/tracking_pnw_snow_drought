import os 
import sys
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
from snotel_intermittence_functions import PrepPlottingData
from math import sqrt 
import math
import sp_data_analysis as sp_funcs
import datetime

def is_perfect(n): #https://djangocentral.com/python-program-to-check-if-a-number-is-perfect-square/
	root = math.sqrt(int(n))
	if int(root + 0.5) ** 2 == n:
	    return n
	else:
	    print(n, "is not a perfect square")
def read_csv(input_csv): 
	df = pd.read_csv(input_csv,parse_dates=True)
	return df
def get_sentinel_data(csv_file,huc_level,orbit,water_year_start,water_year_end):
	df_list = []
	#for file in glob.glob(csv_dir+'*.csv'): #you can uncomment this and the if statement if you want to grab multiple files from that directory  
	#	if (huc_level in file) and (orbit.upper() in file) and (water_year_start in file) and (water_year_end in file): 
			#print(file)
			#get one year, orbit- this has a bunch of different stations in it
			#df = pd.read_csv(file,parse_dates=True)
	df=PrepPlottingData(None,csv_file,None,None).csv_to_df()
	df = df.drop(columns=['system:index','.geo'],axis=1)

	#df['date_time'] = df['date_time'].str.split('T',expand=False).str.get(0)
	df1=PrepPlottingData(None,None,None,df).clean_gee_data('huc4')
	#print(df1.head())
	#print(type(df.date.iloc[1]))
	#print(df1)
	#return df1 #changed 11/20/2020 to make one big df instead of a bunch of little guys
	df_list.append(df1)
		# else: 
		# 	print('That csv does not match your input params')
	output_df = pd.concat(df_list)
	return output_df
def convert_pixel_count_sq_km(input_df,data_field,resolution): 
	input_df[data_field]=(input_df[data_field]*resolution*resolution)/1000000
	return input_df
def combine_hucs_w_sentinel(hucs_df,sentinel_df,huc_level): 
	sentinel_df=read_csv(sentinel_df)
	sentinel_df=convert_pixel_count_sq_km(sentinel_df,'filter',30)
	label='Sentinel 1 wet snow (km sq)'
	hucs_df = read_csv(hucs_df)
	hucs_dict = hucs_df.set_index('id').to_dict()['area']
	sentinel_df['area'] = sentinel_df[f'huc{huc_level[1]}'].map(hucs_dict)
	sentinel_df['snow_ratio'] = sentinel_df['filter']/sentinel_df['area']
	sentinel_df['date'] = pd.to_datetime(sentinel_df['date'])
	#if sentinel_df['date'] <= datetime.datetime(2018, 1, 15): 
	sentinel_df['date_weight'] = (datetime.datetime(2018, 1, 15)-sentinel_df['date']).dt.days

	#sentinel_df['date_weight'] = 
	return sentinel_df

	#df1 = df1.replace(rename_dict)

def plot_sentinel_data(input_df,huc_level,orbit,water_year_end):
	fig,ax=plt.subplots(5,5,sharex=True,sharey=True) 
	huc_ids = input_df[f'huc{huc_level[1]}'].unique()
	ax = ax.flatten()
	for x in range(5*5): 
		try: 
			df = input_df[input_df[f'huc{huc_level[1]}']==huc_ids[x]] #get just the huc number from huc_level which is otherwise 04 for example
			#df['std'] = df['filter'].rolling('14d').std()
			print(df.head())
			#print(df['filter'])
			#print(df['area'])
			#print(df['snow_ratio'])
			print(type(df['date_weight'].iloc[0]))
		except Exception as e: 
			print(f'exception was {e}')
			continue 
		ax[x].plot(df['date'],df['snow_ratio'],color='darkorange',lw=0.5)
		ax[x].set_title(f'HUC {huc_ids[x]} {water_year_end} water year')
		ax[x].set_ylabel('Wet snow pixel count')
	#plt.xticks(rotation=45)
	#plt.tight_layout()
	plt.show()
	plt.close('all')

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		csv_dir=variables['csv_dir']
		single_csv=variables['single_csv']
		huc_level=variables['huc_level']
		orbit=variables['orbit']
		water_year_start=variables['water_year_start']
		water_year_end=variables['water_year_end']
		hucs_data = variables['hucs_data']
	#df=get_sentinel_data(csv_dir,huc_level,orbit,water_year_start,water_year_end)
	#df=sp_funcs.get_sp_data(single_csv)
	df=read_csv(single_csv)
	df=convert_pixel_count_sq_km(df,'filter',500)
	label='Sentinel 1 wet snow (km sq)'
	hucs = read_csv(hucs_data)
	print(hucs.head())
	combine_hucs_w_sentinel(hucs,df,huc_level)
	#sp_funcs.plot_quartiles(df,'elev_mean','filter','huc8',label)
	plot_sentinel_data(df,huc_level,orbit,water_year_end)
	#is_perfect(16)
if __name__ == '__main__':
    main()

##this is working 12/21/2020 but is generally depreceated. The format of these data coming from GEE has changed and this format is now unnecessary 
# def get_sentinel_data(csv_file,huc_level,orbit,water_year_start,water_year_end):
# 	df_list = []
# 	#for file in glob.glob(csv_dir+'*.csv'): #you can uncomment this and the if statement if you want to grab multiple files from that directory  
# 	#	if (huc_level in file) and (orbit.upper() in file) and (water_year_start in file) and (water_year_end in file): 
# 			#print(file)
# 			#get one year, orbit- this has a bunch of different stations in it
# 			#df = pd.read_csv(file,parse_dates=True)
# 	df=PrepPlottingData(None,csv_file,None,None).csv_to_df()
# 	df = df.drop(columns=['system:index','.geo'],axis=1)

# 	#df['date_time'] = df['date_time'].str.split('T',expand=False).str.get(0)
# 	df1=PrepPlottingData(None,None,None,df).clean_gee_data('huc4')
# 	#print(df1.head())
# 	#print(type(df.date.iloc[1]))
# 	#print(df1)
# 	#return df1 #changed 11/20/2020 to make one big df instead of a bunch of little guys
# 	df_list.append(df1)
# 		# else: 
# 		# 	print('That csv does not match your input params')
# 	output_df = pd.concat(df_list)
# 	return output_df