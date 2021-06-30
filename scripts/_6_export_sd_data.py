import pandas as pd 
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt 
import geopandas as gpd 
import json 
import glob
import datetime
from functools import reduce
from _1_calculate_snow_droughts_mult_sources import FormatData,CalcSnowDroughts
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import seaborn as sns
from scipy.stats import pearsonr
from _3_plot_sd_recentness_comparison import define_snow_drought_recentness

def main(daymet_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',**kwargs):
	
	################################################################
	#first do the daymet data 
	#read in all the files in this dir and combine them into one df
	early=FormatData(glob.glob(daymet_dir+'*_12_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	mid=FormatData(glob.glob(daymet_dir+'*_2_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	late=FormatData(glob.glob(daymet_dir+'*_4_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	#print('MID WINTER',mid[['date','huc8','swe']].head(50))
	#for period in [early,mid,late]: 
		
		#calculate snow droughts for each period 

	################################################################
	#next do the snotel data 
	output=[]

	#read in some pickled objects, these look like a list of dfs with each being a station for the full time period 
	for item in ['PREC','TAVG','WTEQ']:
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		df=FormatData(files,drop_cols=['year','month','day']).read_in_pickles()
		output.append(df)
	
	#join the three enviro params 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on=['date','id']), output)
	#print(output_df)
	
	#convert the temp column from F to C 
	output_df['TAVG'] = (output_df['TAVG']-32)*(5/9) 

	#convert prec and swe cols from inches to cm 
	output_df['PREC'] = output_df['PREC']*2.54
	output_df['WTEQ'] = output_df['WTEQ']*2.54
	
	#remove rows that have one of the data types missing- this might need to be amended because 
	#it means that there are different numbers of records in some of the periods. 
	output_df=output_df.dropna()
	# print('value counts')
	# print(output_df.WTEQ.value_counts())

	# pd.set_option('display.max_columns', None)
	# pd.set_option('display.max_rows', None)
	# print(output_df.head(50))
	#cast the snotel id col to int to add the hucs 
	output_df['id'] = output_df['id'].astype('int')

	#add the as yet nonexistant hucs data to the outputs 
	hucs = kwargs.get('hucs')
	output_df['huc8'] = output_df['id'].map(hucs)

	#there are multiple snotel stations in some of the basins, 
	#combine those so there is just one number per basin like the 
	#daymet and RS data. 

	output_df=output_df.groupby(['huc8','date'])['PREC','WTEQ','TAVG'].mean().reset_index()
	#print(output_df)


	period_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
			#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)
		
		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991).calculate_snow_droughts()
			#print('snotel')
			#print(snotel_drought)
		else: 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG').calculate_snow_droughts()
		
		#get cols of interest 
		snotel_drought=snotel_drought[['huc8','year','dry','warm','warm_dry']]
		#rename cols so they don't get confused when data are merged 
		snotel_drought.columns=['huc8','year']+['s_'+column for column in snotel_drought.columns if not (column =='huc8') | (column=='year')]
		
		#then do the same for daymet  
		if (p1 == 'mid') | (p1 == 'late'): 
			daymet_drought=CalcSnowDroughts(p2,start_year=1991).calculate_snow_droughts()
		else: 
			daymet_drought=CalcSnowDroughts(p2).calculate_snow_droughts()
		#print('daymet',daymet_drought)
		daymet_drought=daymet_drought[['huc8','year','dry','warm','warm_dry']]
		
		daymet_drought.columns=['huc8','year']+['d_'+column for column in daymet_drought.columns if not (column =='huc8') | (column=='year')]
		
		#export the data to csv 
		# snotel_drought.to_csv(os.path.join(kwargs.get('output_dir'),f'{p1}_season_snotel_snow_drought_occurences_by_basin.csv'))
		# daymet_drought.to_csv(os.path.join(kwargs.get('output_dir'),f'{p1}_season_daymet_snow_drought_occurences_by_basin.csv'))

		snotel_long_term = snotel_drought.groupby('huc8')['s_dry','s_warm','s_warm_dry'].count()
		daymet_long_term = daymet_drought.groupby('huc8')['d_dry','d_warm','d_warm_dry'].count()
		print(snotel_long_term)
		snotel_long_term['wd_max'] = np.where((snotel_long_term['s_warm_dry']>snotel_long_term['s_warm'])&
			(snotel_long_term['s_warm_dry']>snotel_long_term['s_dry']),1,0)

		daymet_long_term['wd_max'] = np.where((daymet_long_term['d_warm_dry']>daymet_long_term['d_warm'])&
			(daymet_long_term['d_warm_dry']>daymet_long_term['d_dry']),1,0)

		test = snotel_long_term.loc[(snotel_long_term['s_dry']>5)&(snotel_long_term['s_warm']>5)&(snotel_long_term['s_warm_dry']>5)]
		print(test.shape)
		# snotel_long_term.to_csv(os.path.join(kwargs.get('output_dir'),f'{p1}_season_snotel_counts_for_each_basin_w_wd_max.csv'))
		# daymet_long_term.to_csv(os.path.join(kwargs.get('output_dir'),f'{p1}_season_daymet_counts_for_each_basin_w_wd_max.csv'))
		
		# dry_subset = snotel_long_term.loc[(snotel_long_term['s_dry']>snotel_long_term['s_warm'])&(snotel_long_term['s_dry']>snotel_long_term['s_warm_dry'])]
		# print(dry_subset)
		# print(dry_subset.shape)

		# recentness=define_snow_drought_recentness(daymet_drought,'huc8',None)
		# print(recentness)

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		pickles=variables['pickles']
		stations=variables['stations']
		daymet_dir=variables['daymet_dir']
		palette=variables['palette']
		output_dir=variables['output_dir']
	
	hucs=pd.read_csv(stations)

	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	main(daymet_dir,pickles,hucs=hucs_dict,palette=palette,output_dir=output_dir)