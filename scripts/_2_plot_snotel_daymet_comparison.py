import os 
import sys 
import pandas as pd 
import matplotlib.pyplot as plt 
import json 
import glob
from _1_calculate_revised_snow_drought import FormatData,CalcSnowDroughts
import snow_drought_definition_revision as sd
from snow_drought_definition_revision import DefineClusterCenters
from scipy.stats import pearsonr
from functools import reduce
from scipy import stats
import numpy as np 

def plot_daily_pt_based_comparison(snotel, daymet, **kwargs): #these will come in as lists like: [early, mid, late]
	"""Make regression or scatter plots of the variables used in defining snow droughts for each section of the winter."""
	cols = ['swe','prcp','tavg'] #these are hardcoded and assume you've changed the swe cols to match the daymet cols 

	fig,axs = plt.subplots(3,3,figsize=(8,6),
							sharex='row',
							sharey='row',
							gridspec_kw={'wspace':0.0,'hspace':0.25})

	for x in range(3): #iterate through the cols
		#get a merged df of the two input datasets for that subset of the season 
		df = snotel[x].merge(daymet[x],on=['site_num','date'],how='inner')
		#apply some logical limits- this needs to be amended a bit or figure out where these crazy values are coming from 
		cols_plus = cols + ['s_'+col for col in cols]
		df = df[cols_plus]

		#df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
		print(df)
		print(df.shape)
		df = df.loc[(df['tavg']<=40) & (df['s_tavg']<=40)] #this is maybe a little questionable

		for y in range(3): #iterate through the rows
			
			# df=df.loc[np.abs(df[cols[y]]-df[cols[y]].mean()) <= (3*df[cols[y]].std())]
			# df=df.loc[np.abs(df[f's_{cols[y]}']-df[f's_{cols[y]}'].mean()) <= (3*df[f's_{cols[y]}'].std())]
			# print(df)
			# print(df.shape)
			print(f'the plot is: [{y}][{x}]')

			print('col is: ',cols[y])
			print('snotel')
			print(df[f's_{cols[y]}'].min(),
				df[f's_{cols[y]}'].max())
			print('daymet')
			print(df[cols[y]].min(),
				df[cols[y]].max()
				)
			axs[y][x].scatter(df[f's_{cols[y]}'],
									df[cols[y]],
									s=50,
									facecolors='None',
									edgecolors='black',
									alpha=0.25
									) 
			#add pearson correlation 
			corr, _ = pearsonr(df[f's_{cols[y]}'], df[cols[y]])
			rho, pval = stats.spearmanr(df[f's_{cols[y]}'], df[cols[y]])
			axs[y][x].annotate(f'r = {round(rho,2)}',xy=(0.05,0.85),xycoords='axes fraction',fontsize=12)



	axs[0][0].set_title('Early',fontdict={'fontsize':'large'})
	axs[0][1].set_title('Mid',fontdict={'fontsize':'large'})
	axs[0][2].set_title('Late',fontdict={'fontsize':'large'})
	axs[0][0].set_ylabel('\u03A3 SWE (mm)',fontsize=12)
	axs[1][0].set_ylabel('Daily precip (mm)',fontsize=12)
	axs[2][0].set_ylabel('Tavg (deg C)',fontsize=12)

	fig.text(0.5, 0.04, 'SNOTEL', ha='center',fontsize=12)
	fig.text(0.025, 0.5, 'Daymet', va='center', rotation='vertical',fontsize=12)

	# plt.show()
	# plt.close('all')

	plt.savefig(os.path.join(kwargs.get('fig_dir'),'snotel_daymet_comparison_spearman_draft1.jpg'),
		dpi=500, 
		bbox_inches = 'tight',
    	pad_inches = 0
		)

def main(daymet_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',huc_col = 'site_num', **kwargs):
	"""Compare the agreement of predictor variables from Daymet and SNOTEL used for classifying snow droughts"""
	
	#check if the fig output dir exists, if not make it
	if not os.path.exists(kwargs.get('fig_dir')): 
		os.mkdir(kwargs.get('fig_dir'))

	print(f'The huc col being processed is: {huc_col}')
	################################################################
	#first do the daymet data 
	#read in all the files in this dir and combine them into one df
	early=FormatData(glob.glob(daymet_dir+f'*_12_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	mid=FormatData(glob.glob(daymet_dir+f'*_2_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	late=FormatData(glob.glob(daymet_dir+f'*_4_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	################################################################

	#next do the snotel data 
	output=[]

	#read in some pickled objects, these look like a list of dfs with each being a station for the full time period 
	for item in ['PRCP','TAVG','WTEQ']: #note that PREC is accumulated precip and PRCP is the daily precip- 
		#we use this one to match with the Daymet format and avoid issues with accumulated error in the cumulative data. 
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		df=FormatData(files,drop_cols=['year','month','day']).read_in_pickles()
		output.append(df) #the df here is 365 days x ~30 yrs x 237 stations so these are pretty big dfs
	
	#join the three enviro params 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on=['date','id']), output)
	#convert the temp column from F to C 
	output_df['TAVG'] = (output_df['TAVG']-32)*(5/9) 
	#convert prec and swe cols from inches to mm - note that elsewhere this is cm but daymet is mm. Make sure these are homogenized
	print('before')
	print(output_df.PRCP)
	output_df['PRCP'] = output_df['PRCP']*25.4
	output_df['WTEQ'] = output_df['WTEQ']*25.4
	print('after')
	print(output_df.PRCP)
	#remove rows that have one of the data types missing- this might need to be amended because 
	#it means that there are different numbers of records in some of the periods. 
	output_df=output_df.dropna()
	
	#cast the snotel id col to int to add the hucs 
	output_df['id'] = output_df['id'].astype('int')
	output_df['date'] = pd.to_datetime(output_df['date'])

	#do a quick test of how the variables are calculated 
	# test = output_df.loc[(output_df['id']==867)] #& (output_df.date.dt.year == 1981)]
	# test['year'] = test['date'].dt.year
	# print(test)
	# test = test.loc[test['date'].dt.year == 1990]
	# test = test.loc[test.date.dt.month < 5]
	# pd.set_option("display.max_rows", None, "display.max_columns", None)
	# print(test)
	# ax=test.plot(x='date',y='WTEQ')
	# plt.show()
	#add the as yet nonexistant hucs data to the outputs 
	# hucs = kwargs.get('hucs')
	# output_df[huc_col] = output_df['id'].map(hucs)

	snotel_list = []
	daymet_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
			#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)
		#rename the snotel cols to match daymet
		snotel_chunk.rename(columns={'WTEQ':'s_swe','PRCP':'s_prcp','TAVG':'s_tavg','id':'site_num'},inplace=True) #add s_ to the snotel data so they're not confused after merging
		snotel_list.append(snotel_chunk)
		daymet_list.append(p2)

	plot_daily_pt_based_comparison(snotel_list,daymet_list,**kwargs)
if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		pickles=variables['pickles']
		stations=variables['stations']
		daymet_dir=variables['daymet_dir']
		palette=variables['palette']
		stats_dir = variables['stats_dir']
		fig_dir = variables['fig_dir']
	
	main(daymet_dir,pickles,
		stats_dir=stats_dir,
		fig_dir=fig_dir)