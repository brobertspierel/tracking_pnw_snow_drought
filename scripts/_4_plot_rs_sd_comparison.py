
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
import sys 
import statsmodels.api as sa
import glob 
import scikit_posthocs as sp
import pandas as pd 
import geopandas as gpd 
import numpy as np 
import os 
import _pickle as cPickle
# from _4_process_rs_data import generate_output,combine_rs_snotel_annually,aggregate_dfs,merge_dfs,split_basins,combine_sar_data
import _4_process_rs_data as _4_rs
import matplotlib
from _1_calculate_snow_droughts_mult_sources import FormatData,CalcSnowDroughts



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

def split_dfs_within_winter_season(df,sp=False): 
	"""Splits a single df by date ranges in a winter season."""
	
	early_df = df.loc[(df['date'].dt.month>=11)] 
	mid_df = df.loc[(df['date'].dt.month>=1)&(df['date'].dt.month<=2)]
	late_df = df.loc[(df['date'].dt.month>=3)&(df['date'].dt.month<=4)]
	
	return [early_df,mid_df,late_df]


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
		split_dfs_within_winter_season(input_data,None) #none is a stand in for the region 


def merge_dfs(snotel_data,rs_data,drought_type,huc_level='8',col_of_interest='NDSI_Snow_Cover',resolution=500,**kwargs): #added drought_type arg so the drought type is supplied externally 3/15/2021
	"""Merge snotel snow drought data with RS data."""
	
	if drought_type.lower() == 'total': 
		rs_data.rename(columns={col_of_interest:f'{drought_type}_{col_of_interest}'},inplace=True)
		
		return rs_data

	combined = create_snow_drought_subset(snotel_data,drought_type,huc_level)
	if not f'huc{huc_level}' in combined.columns: 
		combined.rename(columns={'huc_id':f'huc_{huc_level}'},inplace=True)
		combined[f'huc_{huc_level}'] = pd.to_numeric(rs_data['huc'+huc_level])

	combined=combined.merge(rs_data, on=['date',f'huc{huc_level}'], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
	#get the rs data for the time periods of interest for a snow drought type 
	combined.rename(columns={col_of_interest:f'{drought_type}_{col_of_interest}'},inplace=True)
	
	combined = combined.sort_values(f'{drought_type}_{col_of_interest}').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')
	return combined
def get_df_chunk(df,year): 
		"""Helper function."""
		year = int(year)
		#create a date mask to get the range of data we want 
		mask = (df['date']>=pd.to_datetime(f'{year-1}-11-01')) & (df['date'] <= pd.to_datetime(f'{year}-04-30')) #hardcoded
		return df.loc[mask]

def combine_rs_snotel_annually(input_dir,season,pickles,agg_step=12,resolution=500,huc_level='8',col_of_interest='NDSI_Snow_Cover',elev_stat='elev_mean',sp=False,total=False,**kwargs): 
	"""Get RS data for snow drought time steps and return those data split by region."""
	
	combined = []
	optical_files = sorted(glob.glob(input_dir+'*.csv'))
	
	for file in optical_files: 
		year = re.findall('(\d{4})', os.path.split(file)[1])[1] #gets a list with the start and end of the water year, take the second one. expects files to be formatted a specific way from GEE 
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
		optical_data = get_df_chunk(optical_data,year)
		# if 'sar_data' in kwargs: 
		# 	sar_data = input_data.get_sentinel_data('filter')

		#convert pixel counts to area
		if not sp: 
			optical_data=rs_funcs.convert_pixel_count_sq_km(optical_data,col_of_interest,resolution)
			#optical_data['area'] = optical_data[f'huc{huc_level}'].map(kwargs.get('hucs_data'))
			#normalize snow covered area by basin area
			#optical_data[col_of_interest] = optical_data[col_of_interest]/optical_data['area'] #changed 4/9/2021 to update the NDSI_Snow_Cover col in place 
		#optical_data['year'] = optical_data['date'].dt.year

		if not total: 
			#combine the remote sensing and snotel data using the snotel dates of snow droughts to extract rs data 
			merged=merge_dfs(short_term_snow_drought,optical_data,kwargs.get('drought_type')) #snotel_data,rs_data,drought_type
			combined.append(merged)

		else: 
			combined.append(optical_data)
	return pd.concat(combined,axis=0)
			#print('Calculating total with no snow droughts')
		#output = split_dfs_within_winter_season
		
def run_stats(dfs,cols): # list of the dfs you want to compare  
	"""Run Kruskal-Wallis H test. This is analogous to 1 way ANOVA but for non-parametric applications. 
	The conover test is used for post-hoc testing to determine relationship between variables. NOTE that the post hoc tests 
	should only be used when there is a significant result of the omnibus test.""" 

	#deal with cases where all vals in a col are nan 
	#input_df=input_df.dropna(axis=1, how='all')
	#set inf to nan 
	data = [df[col].replace(np.inf,np.nan).to_numpy() for df,col in zip(dfs,cols)]
	#input_df=input_df.replace(np.inf,np.nan)

	# if input_df.isnull().all().all():
	# 	return None
	# #reformat the df cols into arrays to pass to the stats func 
	# data = [input_df[column].to_numpy() for column in input_df.columns if not column=='huc8']
	
	#run the kruskal-wallis 
	H,p = stats.kruskal(*data,nan_policy='omit')
	#return H,p
	#print(H,p)
	try: 
		#run the post-hoc test 
		#conover = sp.posthoc_conover([input_df.dropna().iloc[:,0].values,input_df.dropna().iloc[:,1].values,input_df.dropna().iloc[:,2].values,input_df.dropna().iloc[:,3].values],p_adjust='holm')
		conover = sp.posthoc_conover(data,p_adjust='holm')
		conover.columns = cols
		conover.index = cols
		
		return H,p,conover 
		
	except Exception as e: 
		print('Error is: ', e)
# def map_hucs(huc_shapefile,stats_data):
# 	"""Make a map of the KW outputs."""
# 	gdf = gpd.read_file(huc_shapefile)
# 	gdf['stat'] = gdf['id']
def condense_rs_data(input_df,date_col='date',sort_col='huc8',agg_col='NDSI_Snow_Cover',data_type='sca',resolution=500):

	#add a year col for the annual ones 
	input_df[date_col]= pd.to_datetime(input_df[date_col])

	input_df['year'] = input_df[date_col].dt.year

	if data_type.lower()=='sca': #this is already a sum of the SCA in a given basin so get max extent 
		#convert the SCA pixel count to area 
		input_df[agg_col] = (input_df[agg_col]*resolution*resolution)/1000000
		
		output_df = input_df.groupby([sort_col,'year'])[agg_col].max().reset_index()#.agg({self.swe_c:'max',self.precip:'sum'})
	
	elif data_type.lower()=='sp': 
		pass

	else: 
		print('Your data type for the RS data is neither sp nor sca. Double check what you are doing.')

	#get the long-term means 
	median = output_df.groupby(sort_col)['NDSI_Snow_Cover'].median().reset_index()
	#print('means is: ',means)
	print('asl;ghasldgh;askjh')
	print(median)
	#rename the means cols so when they merge they have distinct names 
	median.rename(columns={'NDSI_Snow_Cover':'median'},inplace=True)
	#print(means)
	print('something worked')
	print(median)
	#merge the means with the summary stats for each year/basin- this can be split for the three processing periods 
	output_df = output_df.merge(median[[sort_col,'median']],how='inner',on=sort_col)

	# return sd_df
	return output_df

def main(sp_data,sca_data,pickles,season,index,data_type,output_dir,daymet_dir,agg_step=12,huc_level='8',resolution=500,start_date='1980-10-01',end_date='2020-09-30',**kwargs):
	"""
	Link the datatypes together and add summary stats. 
	"""
	#read in some modis/viirs data
	rs_early=condense_rs_data(FormatData(glob.glob(sca_data+'*_12_huc8.csv'),drop_cols=['system:index','.geo']).read_in_csvs())
	rs_mid=condense_rs_data(FormatData(glob.glob(sca_data+'*_2_huc8.csv'),drop_cols=['system:index','.geo']).read_in_csvs())
	rs_late=condense_rs_data(FormatData(glob.glob(sca_data+'*_4_huc8.csv'),drop_cols=['system:index','.geo']).read_in_csvs())

	#read in the daymet data 
	early=FormatData(glob.glob(daymet_dir+'*_12_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	mid=FormatData(glob.glob(daymet_dir+'*_2_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	late=FormatData(glob.glob(daymet_dir+'*_4_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	print('example')
	print(rs_early)
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

		#merge the two datasets into one df 
		dfs = snotel_drought.merge(daymet_drought,on=['huc8','year'],how='inner')

		print('outputs:')
		print(snotel_drought)
		print(daymet_drought)
		print(dfs)
		#print('THe combined output looks like: ', dfs)
		#compare each drought type and record the results in a new col 

		#not sure if these comparison cols are actually needed here or not 
		# dfs['dry']=dfs['s_dry']==dfs['d_dry']
		# dfs['warm']=dfs['s_warm']==dfs['d_dry']
		# dfs['warm_dry']=dfs['s_warm_dry']==dfs['d_warm_dry']

		#print(dfs.groupby(['huc8'])['dry','warm','warm_dry'].sum())
		# pd.set_option('display.max_columns', None)
		# pd.set_option('display.max_rows', None)
		# #print(dfs)
		period_list.append(dfs)

	#catch an error before it happens 
	#################################3
	#########commented from here 
	# if (data_type.upper() == 'SP') & (index > 0) & (data_type.upper() != 'SAR'): 
	# 	print('You have specified data type SP but an index for SCA. \n reassigning index to 0.')
	# 	index = 0 
	# #####################################################################################################################
	# #get optical data with first being SP
	# # dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='dry',sp=True),sp=True)
	# # warm_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm',sp=True),sp=True)
	# # warm_dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm_dry',sp=True),sp=True)
	# # total_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,sp=True,total=True),sp=True)

	# #then SCA
	# hucs_df=pd.read_csv(kwargs.get('hucs_data')) 
	# hucs_data = dict(zip(hucs_df.id, hucs_df.area))
	
	
	# dry=combine_rs_snotel_annually(sca_data,season,pickles,drought_type='dry', hucs_data=hucs_data)
	# warm=combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm',hucs_data=hucs_data)
	# warm_dry=combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm_dry',hucs_data=hucs_data)
	# total = combine_rs_snotel_annually(sca_data,season,pickles,total=True,hucs_data=hucs_data)
	
	# # print(dry['dry_NDSI_Snow_Cover'])
	# # print(total)
	# median_extents = (total.groupby('huc8')['NDSI_Snow_Cover'].median()).to_dict()
	# #print(median_extents)
	
	# cols = ['dry_NDSI_Snow_Cover','warm_NDSI_Snow_Cover','warm_dry_NDSI_Snow_Cover','NDSI_Snow_Cover']
	# sca_dfs = [dry,warm,warm_dry,total]
	# #sp_dfs = [dry_sp,warm_sp,warm_dry_sp,total_sp]
	
	# #do a little bit more formatting. Here we get the long term winter median for each basin and divide each 12 day period by that long term median. 
	# for df,col in zip(sca_dfs,cols):
	# 	df['median'] = df['huc8'].map(median_extents) 
	# 	df[col] = df[col]/df['median']
	# # 	print(df.dtypes)

	# # print(sca_dfs[0]['dry_NDSI_Snow_Cover'])

	# #calculate stats 
	# stats_out = {}
	# for huc in set(total['huc8']): 
	# 	#print(huc)
	# 	stats_input = [df.loc[df['huc8']==int(huc)] for df in sca_dfs]
	# 	print(stats_input)
	# 	stats_out.update({huc:run_stats(stats_input,cols)})

	# #print(stats_out)

	# hucs_gdf = gpd.read_file(kwargs.get('huc_shapefile'))
	
	# #print(hucs_gdf.columns)
	# dry_warm = {}
	# dry_warm_dry = {}
	# dry_total = {}
	# warm_warm_dry= {}
	# warm_total={}
	# warm_dry_total={}
	# count = 0
	# for k,v in stats_out.items(): 
	# 	try: 

	# 		# print(v[2].iat[1,0])
	# 		# print(v[2])
	# 		dry_warm.update({k:v[2].iat[1,0]})
	# 		dry_warm_dry.update({k:v[2].iat[2,0]})		
	# 		dry_total.update({k:v[2].iat[3,0]})
	# 		warm_warm_dry.update({k:v[2].iat[2,1]})
	# 		warm_total.update({k:v[2].iat[1,3]})
	# 		warm_dry_total.update({k:v[2].iat[3,2]})

	# 	except Exception as e: 
	# 		print(f'error here was: {e}')
	# 		print(v)
	# 		count +=1 
	# #print(dry_warm)
	# print(f'there were {count} errors in this run ')
	# hucs_gdf['huc8'] = hucs_gdf['huc8'].astype('int')
	
	# hucs_gdf['dry_warm'] = ((hucs_gdf['huc8'].map(dry_warm)) <=0.05).astype('int')
	# hucs_gdf['dry_warm_dry'] = ((hucs_gdf['huc8'].map(dry_warm_dry)) <= 0.05).astype('int')
	# hucs_gdf['dry_total'] = ((hucs_gdf['huc8'].map(dry_total)) <=0.05).astype('int')
	# hucs_gdf['warm_warm_dry'] = ((hucs_gdf['huc8'].map(warm_warm_dry)) <=0.05).astype('int')
	# hucs_gdf['warm_total'] = ((hucs_gdf['huc8'].map(warm_warm_dry)) <=0.05).astype('int')
	# hucs_gdf['warm_dry_total'] = ((hucs_gdf['huc8'].map(warm_dry_total)) <=0.05).astype('int')

	# # print('df looks like')
	# # print(hucs_gdf)
	
	# # hucs_gdf.loc[hucs_gdf['dry_warm'] <=0.05] =1#hucs_gdf['dry_warm'] 
	# # hucs_gdf.loc[hucs_gdf['dry_warm_dry'] <=0.05]=1#hucs_gdf['dry_warm_dry']
	# # hucs_gdf.loc[hucs_gdf['dry_total'] <=0.05]=1#hucs_gdf['dry_total'] 
	# # hucs_gdf.loc[hucs_gdf['warm_warm_dry'] <=0.05]=1#hucs_gdf['warm_warm_dry'] 
	# # hucs_gdf.loc[hucs_gdf['warm_total'] <=0.05]=1#hucs_gdf['warm_total'] 
	# # hucs_gdf.loc[hucs_gdf['warm_dry_total'] <=0.05]=1#hucs_gdf['warm_dry_total'] 

	# #data_df["mean_radius"] = (data_df["mean radius"] <= 12.0).astype(int)
	# #data_df.loc[data_df["mean radius"] > 12.0, "mean radius"] = 0


	# print(hucs_gdf)
	# plot_cols = ['dry_warm','dry_warm_dry','dry_total','warm_warm_dry','warm_total','warm_dry_total']
	# fig,ax = plt.subplots(3,2,sharex=True,sharey=True,figsize=(12,10))	
	# cmap=matplotlib.colors.ListedColormap(['#7f7f7f','#CC7000'])

	# ax = ax.flatten()
	# for i in range(6): 
	# 	hucs_gdf.plot(column=plot_cols[i],ax=ax[i], 
	# 	cmap=cmap,legend=True,#,#,missing_kwds={
	# 	# 		"color": "lightgrey",
	# 	# 		"edgecolor": "red",
	# 	# 		"hatch": "///",
	# 	# 		"label": "Missing values"}, 
	# 	legend_kwds={'label': "Kruskal-Wallis test (alpha 0.05)",
	# 	'orientation': "horizontal"})#,color={'#000000':0,'#ff0000':1})
	# 	#ax[i].legend()
	# 	ax[i].set_title(plot_cols[i])
	# # handles, labels = ax[5].get_legend_handles_labels()
	# # fig.legend(handles, labels, loc='center')
	# plt.tight_layout()
	# plt.show()
	# plt.close('all')


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
		hucs_data = variables['hucs_data']
		huc_shapefile = variables['huc_shapefile']
		stations=variables['stations']
		daymet_dir=variables['daymet_dir']

	hucs=pd.read_csv(stations)

	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	#example function call for just optical data 
	#main(sp_data,sca_data,pickles,season,index=2,data_type='SAR',output_dir=pickles) #note that index can be 0-2 for SCA and only 0 for SP 

	#example call for SAR data included
	main(sp_data,sca_data,pickles,season,-9999,daymet_dir=daymet_dir,data_type='SCA',output_dir=pickles,hucs_data=hucs_data,huc_shapefile=huc_shapefile,hucs=hucs_dict)#,sar_data=sentinel_csv_dir) #note that index can be 0-2 for SCA and only 0 for SP 
