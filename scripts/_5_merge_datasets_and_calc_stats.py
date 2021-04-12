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
import _4bv1_calculate_long_term_sp as _4b_rs 
import re
import math 
from scipy import stats
from functools import reduce
import statsmodels.api as sa
import scikit_posthocs as sp
import _pickle as cPickle
from _4_process_rs_data import generate_output,combine_rs_snotel_annually,aggregate_dfs,merge_dfs,split_basins,combine_sar_data
import _4_process_rs_data as _4_rs


def run_stats(input_df): 
	"""Run Kruskal-Wallis H test. This is analogous to 1 way ANOVA but for non-parametric applications. 
	The conover test is used for post-hoc testing to determine relationship between variables. NOTE that the post hoc tests 
	should only be used when there is a significant result of the omnibus test.""" 

	#deal with cases where all vals in a col are nan 
	input_df=input_df.dropna(axis=1, how='all')
	#set inf to nan 
	input_df=input_df.replace(np.inf,np.nan)

	if input_df.isnull().all().all():
		return None
	#reformat the df cols into arrays to pass to the stats func 
	data = [input_df[column].to_numpy() for column in input_df.columns if not column=='huc8']
	
	#run the kruskal-wallis 
	H,p = stats.kruskal(*data,nan_policy='omit')
	#print(H,p)
	try: 
		#run the post-hoc test 
		#conover = sp.posthoc_conover([input_df.dropna().iloc[:,0].values,input_df.dropna().iloc[:,1].values,input_df.dropna().iloc[:,2].values,input_df.dropna().iloc[:,3].values],p_adjust='holm')
		conover = sp.posthoc_conover(data,p_adjust='holm')
		conover.columns = input_df.columns
		conover.index = input_df.columns
		
		return H,p,conover 
		
	except Exception as e: 
		print('Error is: ', e)
		#print('passing post-hoc test')
def iteratively_generate_stats(input_df): 
	"""Helper function."""
	output_dict = {}
	for x in input_df.index.unique():
		df=input_df[input_df.index==x] 
		try: 
			output_dict.update({x:run_stats(df)[1]}) #run the stats function and get the p-value out 
		except Exception as e: 
			print(f'The offending huc was: {x} and the error was {e}')
	return output_dict


class FormatDf(): 
	
	def __init__(self,input_dfs,col_list,region,index,year,column='date'): 
		self.input_dfs = input_dfs
		self.col_list = col_list
		self.region = region 
		self.index = index
		self.year = year 
		self.column = column

	def get_df_chunk(self,input_data): 
		"""Helper function."""
		df=input_data[self.region][self.index]
		#create a date mask to get the range of data we want 
		mask = (df[self.column]>=pd.to_datetime(f'{self.year-1}-11-01')) & (df[self.column] <= pd.to_datetime(f'{self.year}-04-30')) #hardcoded
		return df.loc[mask]

	def format_output(self): 
		"""Helper function."""
		output_dfs = [self.get_df_chunk(df)[col] for df,col in zip(self.input_dfs,self.col_list)]
		merged = pd.concat(output_dfs,axis=1)
		#sort to put the np.nan values at the bottom
		return merged.transform(np.sort)



def main(sp_data,sca_data,pickles,season,index,data_type,output_dir,year_of_interest,agg_step=12,huc_level='8',resolution=500,**kwargs):
	"""
	Link the datatypes together and add summary stats. 
	"""

	#catch an error before it happens 
	if (data_type.upper() == 'SP') & (index > 0) & (data_type.upper() != 'SAR'): 
		print('You have specified data type SP but an index for SCA. \n reassigning index to 0.')
		index = 0 
	#####################################################################################################################
	#get optical data with first being SP
	# dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='dry',sp=True),sp=True)
	# warm_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm',sp=True),sp=True)
	# warm_dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm_dry',sp=True),sp=True)
	# total_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,sp=True,total=True),sp=True)

	#then SCA
	hucs_data=pd.read_csv(kwargs.get('hucs_data')) 
	hucs_data = dict(zip(hucs_data.id, hucs_data.area))
	
	
	dry=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='dry',split=False,hucs_data=hucs_data))
	warm=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm',split=False,hucs_data=hucs_data))
	warm_dry=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm_dry',split=False,hucs_data=hucs_data)) 
	total = generate_output(combine_rs_snotel_annually(sca_data,season,pickles,total=True,split=False,hucs_data=hucs_data)) 
	
	print(dry)
	cols = ['dry_NDSI_Snow_Cover','warm_NDSI_Snow_Cover','warm_dry_NDSI_Snow_Cover','NDSI_Snow_Cover']
	sca_dfs = [dry,warm,warm_dry,total]
	sp_dfs = [dry_sp,warm_sp,warm_dry_sp,total_sp]

	#####################################################################################################################
	#combine the datasets to try running the KW test on basins for all the years between drought types instead of within one year 
	#These come in the form {'west'[early,mid,late],'east':['early,mid,late']}
	


	#####################################################################################################################
	#run for sentinel data 
	# snotel_data = pickles+f'short_term_snow_drought_{year_of_interest}_water_year_{season}_{agg_step}_day_time_step_w_all_dates_first_day_start'
	
	# #instantiate the acquireData class and read in snotel, sentinel and modis/viirs data 
	# input_data = obtain_data.AcquireData(kwargs.get('sar_data'),sca_data,snotel_data,kwargs.get('hucs_data'),huc_level,resolution)
	# short_term_snow_drought = input_data.get_snotel_data()
	# sar_data = input_data.get_sentinel_data('filter')
	# optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
	# # print('snotel data ')
	# # print(short_term_snow_drought)
	# # print('sar data: ')
	# # print(sar_data)
	# # print(sar_data.columns)
	# dry = merge_dfs(short_term_snow_drought,optical_data,'dry',col_of_interest='filter',sar_data=sar_data)
	# warm = merge_dfs(short_term_snow_drought,optical_data,'warm',col_of_interest='filter',sar_data=sar_data)
	# warm_dry = merge_dfs(short_term_snow_drought,optical_data,'warm_dry',col_of_interest='filter',sar_data=sar_data)
	# total = merge_dfs(short_term_snow_drought,optical_data,'total',col_of_interest='filter',sar_data=sar_data)
	# print(dry)
	# # print(dry[dry['dry_filter']>5.0].count())
	# print(dry.shape)
	# # print(warm[warm['warm_filter']>5.0].count())
	# print(warm)
	# print(warm.shape)
	# # print(warm_dry[warm_dry['warm_dry_filter']>5.0].count())
	# print(warm_dry)
	# print(warm_dry.shape)
	# # print(total[total['total_filter']>5.0].count())
	# # print(total.shape)
	# merged=combine_sar_data(dry,warm,warm_dry,total,None)
	#print(merged)
	#print(merged[merged > 1.0].count())
	# early = combine_sar_data(dry,warm,warm_dry,total,0).transform(np.sort)
	# print(early)
	# mid = combine_sar_data(dry,warm,warm_dry,total,1).transform(np.sort)
	# late = combine_sar_data(dry,warm,warm_dry,total,2).transform(np.sort)
	# cols = ['dry_WSCA','warm_WSCA','warm_dry_WSCA','total_WSCA']
	#####################################################################################################################

	# kw_west = {}
	# conover_west = {}
	# kw_east = {}
	# conover_east = {}

	# if not data_type.upper() == 'SAR': 
	# 	for year in range(2001,2021): #hardcoded for MODIS record 
	# 		print('year is: ',year)

	# 		if data_type.upper() == 'SCA': 
	# 			print('SCA')
	# 			input_dfs = sca_dfs
	# 		elif data_type.upper() == 'SP': 
	# 			input_dfs = sp_dfs
	# 		else: 
	# 			print('Please check specification for var data_type. This can only be one of SP or SCA')
			
	# 		west_merged = FormatDf(input_dfs,cols,'west',index,year).format_output()#west_merged.transform(np.sort)

	# 		east_merged = FormatDf(input_dfs,cols,'east',index,year).format_output()

	# 		west=run_stats(west_merged) #returns a tuple of the type (H-stat, p-val, conovoer df)
	# 		east=run_stats(east_merged)

	# 		try: 
	# 			if west[1] < 0.05: 
	# 				kw_west.update({year:west[1]})
	# 				conover_west.update({year:west[2]})
	# 			if east[1] < 0.05: 
	# 				kw_east.update({year:east[1]})
	# 				conover_east.update({year:east[2]})
	# 		except TypeError as e: 
	# 			print(e)
	
	# else: #run stats for single years for the SAR data 
	# 	pass
		#df3=df1.groupby(['Country'])['Revenue'].apply(lambda x:(x.max() - x.min()) / 2).reset_index()
		# basin_stats=iteratively_generate_stats(merged)
		# print(basin_stats)
		# print(len(basin_stats))
		# early_stats=iteratively_generate_stats(early)
		# print(early_stats)
		# mid_stats=iteratively_generate_stats(mid)
		# late_stats=iteratively_generate_stats(late)
		# print(early)
		# print(early_stats)
		# print(mid_stats)
		# print(late_stats)
	#pickle the results 
	#make a subdirectory to hold the results  
	# output_dir = os.path.join(output_dir,'kw_results')
	# if not os.path.exists(output_dir): 
	# 	os.mkdir(output_dir)

	# kw_filepath = os.path.join(output_dir,f'kw_west_east_type_{data_type}_time_{index}_results.pickle')
	# conover_filepath = os.path.join(output_dir,f'conover_west_east_type_{data_type}_time_{index}_results.pickle')

	# with open(kw_filepath, "wb") as kw_output:
	# 	#if not os.path.exists(kw_filepath): 
	# 	cPickle.dump({'west':kw_west,'east':kw_east}, kw_output)
	# 	# else: 
	# 	# 	print(f'The file {kw_filepath} already exists, passing')

	# with open(conover_filepath, "wb") as conover_output:
	# 	#if not os.path.exists(conover_filepath): 
	# 	cPickle.dump({'west':conover_west,'east':conover_east}, conover_output)
		# else: 
		# 	print(f'The file {conover_filepath} already exists, passing')

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)		
		#construct variables from param file
		sp_data = variables['sp_data']
		sca_data = variables['sca_data']
		pickles = variables['pickles']
		season = variables['season']
		palette = variables['palette'] #"no_drought":"#cbbdb1",
		year_of_interest=variables['year_of_interest']
		hucs_data = variables['hucs_data']
		optical_csv_dir = variables['optical_csv_dir']
		sentinel_csv_dir = variables['sentinel_csv_dir']

	#example function call for just optical data 
	#main(sp_data,sca_data,pickles,season,index=2,data_type='SAR',output_dir=pickles) #note that index can be 0-2 for SCA and only 0 for SP 

	#example call for SAR data included
	main(sp_data,optical_csv_dir,pickles,season,-9999,data_type='SCA',output_dir=pickles,year_of_interest=year_of_interest,hucs_data=hucs_data,sar_data=sentinel_csv_dir) #note that index can be 0-2 for SCA and only 0 for SP 



# def plot_ratios(west,east,palette): 

# 	fig,(ax,ax1) = plt.subplots(2) 

# 	west.plot.scatter(x='dry_ratio',y='dry_NDSI_Snow_Cover',color='black',ax=ax)
# 	west.plot.scatter(x='warm_ratio',y='warm_NDSI_Snow_Cover',color='blue',ax=ax)
# 	west.plot.scatter(x='warm_dry_ratio',y='warm_dry_NDSI_Snow_Cover',color='red',ax=ax)
	
# 	east.plot.scatter(x='dry_ratio',y='dry_NDSI_Snow_Cover',color='black',ax=ax1)
# 	east.plot.scatter(x='warm_ratio',y='warm_NDSI_Snow_Cover',color='blue',ax=ax1)
# 	east.plot.scatter(x='warm_dry_ratio',y='warm_dry_NDSI_Snow_Cover',color='red',ax=ax1)
	
# 	plt.show()
# 	plt.close()
	# east_dry=get_df_chunk(dry,'east',index,'date',year)['dry_NDSI_Snow_Cover']
		# east_warm=get_df_chunk(warm,'east',index,'date',year)['warm_NDSI_Snow_Cover']
		# east_warm_dry=get_df_chunk(warm_dry,'east',index,'date',year)['warm_dry_NDSI_Snow_Cover']
		# east_total=get_df_chunk(total,'east',index,'date',year)['NDSI_Snow_Cover']
# west_dry=get_df_chunk(dry,'west',index,'date',year)['dry_NDSI_Snow_Cover']
		# west_warm=get_df_chunk(warm,'west',index,'date',year)['warm_NDSI_Snow_Cover']
		# west_warm_dry=get_df_chunk(warm_dry,'west',index,'date',year)['warm_dry_NDSI_Snow_Cover']
		# west_total=get_df_chunk(total,'west',index,'date',year)['NDSI_Snow_Cover']
# snotel_data=_2c_sn.main(year_of_interest,season,pickles,agg_step=12,huc_level='8')
		# snotel_data=_4b_rs.split_basins(snotel_data,'huc_id')
		
		# #rename huc col to align with rs data 
		# snotel_data[0].rename(columns={'huc_id':'huc8'},inplace=True)
		# snotel_data[1].rename(columns={'huc_id':'huc8'},inplace=True)
#come back 
	# print('dry')
	#print(dry_sp)
	#print(dry_sp['east'][0].dtypes)


	#dry_rs = dry_sp[]
	#sca_plot=_4b_rs.plot_sp_sca(dry,warm,warm_dry,total,None,palette,2,3,show=False) #returns a tuple of (west,east) dfs with yearly stats for SCA
	
	#sp_plot=_4b_rs.plot_sp_sca(dry_sp,warm_sp,warm_dry_sp,total_sp,None,palette,2,1,sp=True,show=False) #returns a tuple of (west,east) dfs with yearly stats for SP
	#print(sp_plot)
#print(east_merged)
		#make lists of all the dfs to be merged
		# index=2
		# west_dfs = [get_df_chunk(dry,'west',index,'date',year).groupby(f'huc{huc_level}')['dry_NDSI_Snow_Cover'].median(),
		# get_df_chunk(warm,'west',index,'date',year).groupby(f'huc{huc_level}')['warm_NDSI_Snow_Cover'].median(), 
		# get_df_chunk(warm_dry,'west',index,'date',year).groupby(f'huc{huc_level}')['warm_dry_NDSI_Snow_Cover'].median(),
		# get_df_chunk(total,'west',index,'date',year).groupby(f'huc{huc_level}')['NDSI_Snow_Cover'].median()]#, 
		# #snotel_data[0]]


		# east_dfs = [get_df_chunk(dry,'east',index,'date',year).groupby(f'huc{huc_level}')['dry_NDSI_Snow_Cover'].median(),
		# get_df_chunk(warm,'east',index,'date',year).groupby(f'huc{huc_level}')['warm_NDSI_Snow_Cover'].median(), 
		# get_df_chunk(warm_dry,'east',index,'date',year).groupby(f'huc{huc_level}')['warm_dry_NDSI_Snow_Cover'].median(),
		# get_df_chunk(total,'west',index,'date',year).groupby(f'huc{huc_level}')['NDSI_Snow_Cover'].median()]#, 
		# #snotel_data[1]]

		# west_merged = reduce(lambda  x,y: pd.merge(x,y,on=['huc8'],
  #                                           how='outer'), west_dfs).fillna(np.nan)
		# print('west',west_merged)
		# east_merged = reduce(lambda  x,y: pd.merge(x,y,on=['huc8'],
  #                                           how='outer'), east_dfs).fillna(np.nan)

		# #plot_ratios(west_merged,east_merged,palette)

		# west_yrs.append(west_merged)
		# east_yrs.append(east_merged)
		# print(west_merged)
		# print(east_merged)
		

	# west_dfs = [dry_sp['west'][0].groupby(f'huc{huc_level}')['dry_NDSI_Snow_Cover'].median(),
	# warm_sp['west'][0].groupby(f'huc{huc_level}')['warm_NDSI_Snow_Cover'].median(), 
	# warm_dry_sp['west'][0].groupby(f'huc{huc_level}')['warm_dry_NDSI_Snow_Cover'].median()]

	# 	west = format_df(input_tup=snotel_data,index=0,keep_column='ratio',year=year)
	# 	east = format_df(input_tup=snotel_data,index=1,keep_column='ratio',year=year)
	# 	west_yrs.append(west)
	# 	east_yrs.append(east)

	# #west_all = pd.concat(west_yrs,axis=0)
	# #east_all = pd.concat(east_yrs,axis=0)


	# west_all = pd.concat(west_yrs,axis=0).merge(sca_plot[0],on='year',how='inner')
	# east_all = pd.concat(east_yrs,axis=0).merge(sca_plot[1],on='year',how='inner')
	# print(pd.concat(west_yrs,axis=0).columns)
	
	#west = pd.concat(west_yrs,axis=0)
	#west =west.drop(columns=['huc8'],axis=1)
	#east = pd.concat(east_yrs,axis=0)
	#east = east.drop(columns=['huc8'],axis=1)
	#print(west.columns)
	#print(west.index)
	#plot_ratios(west,east,palette)
	#run_stats(west)