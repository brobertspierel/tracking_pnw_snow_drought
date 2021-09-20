#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt 
import json 
import glob
import datetime
from functools import reduce
from _1_calculate_revised_snow_drought import FormatData,CalcSnowDroughts
import snow_drought_definition_revision as sd
from snow_drought_definition_revision import DefineClusterCenters
from scipy.stats import pearsonr

"""
Make a plot that compares SWE, temp and precip for HUC8 or HUC6 basins. 
Note that this is the summary of pred variables for the basin scale so 
Daymet values are for the full basin and snotel values are the mean of the 
stations in the basin. 
"""

def plot_vars_comparison(df_list,huc_col,**kwargs): 
	"""
	The pred_var arg from kwargs will specify which predictor variable you are interested in plotting. 
	This is a change as of 9/13/2021. 
	""" 
	fig,axs = plt.subplots(3,3, figsize=(8,6),
							sharex=True,
							sharey=True,
							gridspec_kw={'wspace':0,'hspace':0})
	cols=['dry','warm','warm_dry']
	xlabels=['Dry', 'Warm', 'Warm/dry']
	ylabels=['Early','Mid','Late']
# 	output_dict = {}
	count = 0 
	for x in range(3):
		df = df_list[x]
		for y in range(3):  
			print('x is: ',x)
			print('y is: ', y)
			print('drought type is: ', cols[y])
			if kwargs.get('pred_var').lower() == 'precip': 
				d_col = 'd_prcp'
				s_col = 's_PREC'
				label = 'precipitation'
			elif kwargs.get('pred_var').lower() == 'temp': 
				d_col = 'd_tavg'
				s_col = 's_TAVG'
				label = 'Temperature (\u03A3DD)'
			elif kwargs.get('pred_var').lower() == 'swe': 
				d_col = 'd_swe'
				s_col = 's_WTEQ'
				label = '\u03A3\u0394SWE'
			else: 
				print('Make sure you chose the right pred var. You can choose swe, precip or temp')

			df['d_mod'] = np.where((~(df[f'd_{cols[y]}'].isnull()) & ~(df[f's_{cols[y]}'].isnull())), df[d_col], np.nan)
			df['s_mod'] = np.where((~(df[f'd_{cols[y]}'].isnull()) & ~(df[f's_{cols[y]}'].isnull())), df[s_col], np.nan)
			print('input looks like: ')
			print(df)
			print('daymet mean is: ')
			print(df['d_mod'].mean())
			print('snotel mean is: ')
			print(df['s_mod'].mean())
			#df['test'] = np.where((~(df[f'd_{cols[y]}'].isnull()) & ~(df[f's_{cols[y]}'].isnull())), df['d_swe'], np.nan)

			# print('df is: ')
			# print(df[['d_swe_mod','d_dry','d_warm','d_warm_dry','s_warm_dry']])
			#remove nans because its messing with the line of best fit below 
			#df = df.loc[(df['d_swe_mod'].isnull()) & (df['s_swe_mod'].isnull())]
			
			axs[x][y].scatter(df['d_mod'], df['s_mod'], marker='o', facecolors='none', edgecolors='black')
			#add a line of best fit 
			# m, b = np.polyfit(df['d_swe_mod'], df['s_swe_mod'], 1)
			# axs[x][y].plot(df['d_swe_mod'], m*df['d_swe_mod'] + b)
			print('testing')
			print(df['d_mod'])
			print(df['s_mod'])
			#add pearseon/spearman correlation 
			corr, _ = pearsonr(df['d_mod'].dropna(), df['s_mod'].dropna())
			#rho, pval = stats.spearmanr(df[f's_{cols[y]}'], df[cols[y]])
			axs[x][y].annotate(f'r = {round(corr,2)}',xy=(0.05,0.85),xycoords='axes fraction',fontsize=12)

			#set axis labels and annotate 
			#add a subplot letter id 
			axs[x][y].annotate(f'{chr(97+count)}',xy=(0.85,0.9),xycoords='axes fraction',fontsize=10,weight='bold')#f'{chr(97)}'
			axs[0][y].set_title(xlabels[y],fontdict={'fontsize': 10})
			axs[x][0].set_ylabel(ylabels[x],fontsize=10)
			axs[x][y].set_xticks([0.0, 0.5, 1.0])
			axs[x][y].set_yticks([0.0, 0.5, 1.0])
			axs[x][y].tick_params(axis='x', labelsize=10, rotation=45)
			axs[x][y].tick_params(axis='y', labelsize=10)
			count += 1
			#axs[x][y].plot(np.unique(df['d_swe_mod']), np.poly1d(np.polyfit(df['d_swe_mod'], df['s_swe_mod'], 1))(np.unique(df['d_swe_mod'])), color='red', ls='--')
	
	fig.text(0.5, 0.035, f'Daymet scaled {label}', ha='center',fontsize=10)
	fig.text(0.02, 0.5, f'SNOTEL scaled {label}', va='center', rotation='vertical',fontsize=10)

	# fig.text(0.5, 0.04, 'common X', ha='center')
	# fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

	# plt.show()
	# plt.close('all')
	fig_fn = os.path.join(kwargs.get('fig_dir'),f'{huc_col}_daymet_snotel_{kwargs.get("pred_var")}_comparison_draft3.jpg')
	if not os.path.exists(fig_fn): 
		plt.savefig(fig_fn, 
					dpi=500, 
					bbox_inches = 'tight',
					pad_inches = 0, 
					)

def main(daymet_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',huc_col = 'huc8', **kwargs):
	"""Testing improved definitions of snow drought. Original definitions based on Dierauer et al 2019 but trying to introduce some more nuiance into the approach."""
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
	for item in ['PREC','TAVG','WTEQ']:
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		df=FormatData(files,drop_cols=['year','month','day']).read_in_pickles()
		output.append(df) #the df here is 365 days x ~30 yrs x 237 stations so these are pretty big dfs
	
	#join the three enviro params 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on=['date','id']), output)
	
	
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
	output_df[huc_col] = output_df['id'].map(hucs)

	#there are multiple snotel stations in some of the basins, 
	#combine those so there is just one number per basin like the 
	#daymet and RS data. 

	output_df=output_df.groupby([huc_col,'date'])[['PREC','WTEQ','TAVG']].mean().reset_index()

	period_list = []
	snotel_list = []
	daymet_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
			#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)

		##########working below here
		############################
		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991,sort_col=huc_col).prepare_df_cols()
			#print('snotel')
			#print(snotel_drought)
		else: 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',sort_col=huc_col).prepare_df_cols()

		#get cols of interest 
		#snotel_drought=snotel_drought[['huc8','year','dry','warm','warm_dry']]
		#rename cols so they don't get confused when data are merged 
		#snotel_drought.columns=['huc8','year']+['s_'+column for column in snotel_drought.columns if not (column =='huc8') | (column=='year')]
		
		#then do the same for daymet  
		if (p1 == 'mid') | (p1 == 'late'): 
			daymet_drought=CalcSnowDroughts(p2,start_year=1991,sort_col=huc_col).prepare_df_cols()
		else: 
			daymet_drought=CalcSnowDroughts(p2,sort_col=huc_col).prepare_df_cols()
		#print('daymet',daymet_drought)
		#daymet_drought=daymet_drought[['huc8','year','dry','warm','warm_dry']]
		
		#daymet_drought.columns=['huc8','year']+['d_'+column for column in daymet_drought.columns if not (column =='huc8') | (column=='year')]

	##########################################
	
		#run the kmeans with drought types as intiilization conditions (centroids) for the clusters
		
		#these are all of the huc 4 basins in the study area 
		huc4s = ['1708','1801','1710','1711','1709','1701','1702','1705','1703','1601','1707','1706','1712','1704']
		s_output = []
		d_output = []
		for huc4 in huc4s: 
			huc4_s = sd.prep_clusters(snotel_drought,huc4,huc_col=huc_col) #get the subset of the snow drought data for a given huc4
			huc4_d = sd.prep_clusters(daymet_drought,huc4,huc_col=huc_col)
			#make the centroids that serve as the intialization for the kmeans clusters- these are like endmembers (ish)
			s_centroids = DefineClusterCenters(huc4_s,'WTEQ','PREC','TAVG').combine_centroids() #makes a numpy array with four centroids
			d_centroids = DefineClusterCenters(huc4_d,'swe','prcp','tavg').combine_centroids() #makes a numpy array with four centroids

			#clusters should be like: {0:dry, 1:warm, 2:warm_dry, 3:no_drought} 6/8/2021 DOUBLE CHECK
			#run kmeans for the snotel data
			s_clusters = sd.run_kmeans(huc4_s[['WTEQ','PREC','TAVG']].to_numpy(),huc4_s['label'],s_centroids)
			s_clusters = sd.add_drought_cols_to_kmeans_output(s_clusters, huc_col=huc_col) #add a few cols needed for plotting 
			#run kmeans for the daymet data 
			d_clusters = sd.run_kmeans(huc4_d[['swe','prcp','tavg']].to_numpy(),huc4_d['label'],d_centroids)
			d_clusters = sd.add_drought_cols_to_kmeans_output(d_clusters, huc_col=huc_col) #add a few cols needed for plotting 

			s_output.append(s_clusters)
			d_output.append(d_clusters)
		
		s_plot = pd.concat(s_output)
		#select the cols of interest and rename so there's no confusion when dfs are merged 
		s_plot=s_plot[[huc_col,'year','dry','warm','warm_dry']]
		s_plot.columns=[huc_col,'year']+['s_'+column for column in s_plot.columns if not (column == huc_col) | (column=='year')]

		d_plot = pd.concat(d_output)
		d_plot=d_plot[[huc_col,'year','dry','warm','warm_dry']]
		d_plot.columns=[huc_col,'year']+['d_'+column for column in d_plot.columns if not (column == huc_col) | (column=='year')]
		# print(s_plot)
		# print(d_plot)
		

		#merge the two datasets into one df 
		# dfs = s_plot.merge(d_plot,on=[huc_col,'year'],how='inner')
		# period_list.append(dfs)
		# print('final output is: ')
		# print(dfs)
		# print(dfs.columns)
		#try combining the drought classifications and the original input data 
		s_plot[[huc_col,'year']] = s_plot[[huc_col,'year']].astype(int)
		d_plot[[huc_col,'year']] = d_plot[[huc_col,'year']].astype(int)
		daymet_drought[[huc_col,'year']] = daymet_drought[[huc_col,'year']].astype(int)
		snotel_drought[[huc_col,'year']] = snotel_drought[[huc_col,'year']].astype(int)

		daymet_drought.columns=[huc_col,'year']+['d_'+column for column in daymet_drought.columns if not (column == huc_col) | (column=='year')]
		snotel_drought.columns=[huc_col,'year']+['s_'+column for column in snotel_drought.columns if not (column == huc_col) | (column=='year')]
	
		daymet_combined = daymet_drought.merge(d_plot, on=['year', huc_col], how='inner')
		snotel_combined = snotel_drought.merge(s_plot, on=['year', huc_col], how='inner')

		master_df = daymet_combined.merge(snotel_combined, on=['year', huc_col], how='inner')
		master_df = sd.remove_short_dataset_stations(master_df,huc_col)
		period_list.append(master_df)
		print('beast df ', master_df)
	plot_vars_comparison(period_list,huc_col,**kwargs)
	#plot_counts(period_list,kwargs.get('stats_dir'),huc_col=huc_col,**kwargs)

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
	
	hucs=pd.read_csv(stations)
	
	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	print('hucs shape is: ')
	print(hucs.shape)
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	main(daymet_dir,pickles,huc_col='huc8',
							hucs=hucs_dict,
							palette=palette,
							stats_dir=stats_dir,
							fig_dir=fig_dir,
							pred_var='swe'
							)
