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

def is_perfect(n): #https://djangocentral.com/python-program-to-check-if-a-number-is-perfect-square/
	root = math.sqrt(int(n))
	if int(root + 0.5) ** 2 == n:
	    return n
	else:
	    print(n, "is not a perfect square")

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
def plot_sentinel_data(input_df,huc_level,orbit,water_year_end):
	fig,ax=plt.subplots(4,4,sharex=True,sharey=True) 
	huc_ids = input_df[f'huc{huc_level[1]}'].unique()
	ax = ax.flatten()
	for x in range(4*4): 
		try: 
			df = input_df[input_df[f'huc{huc_level[1]}']==huc_ids[x]] #get just the huc number from huc_level which is otherwise 04 for example
		except Exception as e: 
			print(f'exception was {e}')
			continue 
		ax[x].plot(df['date_time'],df['filter'],color='darkorange',lw=0.5)
		ax[x].set_title(f'HUC {huc_ids[x]} {water_year_end} water year')
		ax[x].set_ylabel('Wet snow pixel count')
	plt.xticks(rotation=45)
	#plt.tight_layout()
	plt.show()
	plt.close('all')

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		csv_dir=variables['csv_dir']
		huc_level=variables['huc_level']
		orbit=variables['orbit']
		water_year_start=variables['water_year_start']
		water_year_end=variables['water_year_end']
	df=get_sentinel_data(csv_dir,huc_level,orbit,water_year_start,water_year_end)
	plot_sentinel_data(df,huc_level,orbit,water_year_end)
	#is_perfect(16)
if __name__ == '__main__':
    main()
