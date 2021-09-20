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
from _1_calculate_revised_snow_drought_pt_mode import FormatData,CalcSnowDroughts
import snow_drought_definition_revision_pt_mode as sd
from snow_drought_definition_revision_pt_mode import DefineClusterCenters
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.ticker as ticker
import pymannkendall as mk

def mk_test(input_data): 
	"""Run a version of the Mann-Kendall trend test."""

	trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(input_data)

	return trend, h, p, z, Tau, s, var_s, slope, intercept

def plot_counts(df_list,output_dir,huc_col='huc8',**kwargs): 
	print('Entered the plotting function: ')
	labels=['Snotel','Daymet']
	
	nrow=3
	ncol=3
	fig,axs = plt.subplots(nrow,ncol,sharex=True,sharey=True,
				gridspec_kw={'wspace':0.01,'hspace':0},
                                    #'top':0.95, 'bottom':0.075, 'left':0.05, 'right':0.95},
                figsize=(8,6))
	cols=['dry','warm','warm_dry']
	xlabels=['Dry', 'Warm', 'Warm/dry']
	ylabels=['Early','Mid','Late']
	output_dict = {}
	mk_dict = {}
	count = 0 
	for x in range(3): 
		for y in range(3): 
			
			#produce the count plots 
			s_counts = df_list[x][f's_{cols[y]}'].value_counts().sort_index().astype('int')
			d_counts = df_list[x][f'd_{cols[y]}'].value_counts().sort_index().astype('int')
			# print(s_counts)
			# print(d_counts)
			df = pd.DataFrame({"snotel":s_counts,"daymet":d_counts})
			#reformat a few things in the df 
			df.index=df.index.astype(int)
			df.replace(np.nan,0,inplace=True)

			#there are not droughts in all the years and timeframes but these gaps mess up plotting 
			#so we want to infill them with zeros so all the timeperiods have all of the years. 
			df=df.reindex(np.arange(1990,2021), fill_value=0)
			#run the mann-kendall test to see if these counts are increasing, decreasing or showing no trend over time 
			mk_dict.update({f's_{ylabels[x]}_{xlabels[y]}':mk_test(df.snotel)[0],
				f'd_{ylabels[x]}_{xlabels[y]}':mk_test(df.daymet)[0]})
			#add the counts to a dict so we can output the actual counts and look at them 
			output_dict.update({f's_{ylabels[x]}_{xlabels[y]}':df['snotel'],f'd_{ylabels[x]}_{xlabels[y]}':df['daymet']})

			# calculate Pearson's correlation
			corr, _ = pearsonr(df.snotel, df.daymet)
			print(f'Pearsons correlation: {corr}')

			df.plot.bar(ax=axs[x][y],color=['#D95F0E','#267eab'],width=0.9,legend=False)#,label=['Dry','Warm','Warm/dry']) #need to do something with the color scheme here and maybe error bars? This could be better as a box plot? 
			axs[x][y].grid(axis='both',alpha=0.25)

			#set axis labels and annotate 
			axs[x][y].annotate(f'r = {round(corr,2)}',xy=(0.05,0.9),xycoords='axes fraction',fontsize=10)
			#add a subplot letter id 
			axs[x][y].annotate(f'{chr(97+count)}',xy=(0.85,0.9),xycoords='axes fraction',fontsize=10,weight='bold')#f'{chr(97)}'
			axs[0][y].set_title(xlabels[y],fontdict={'fontsize': 10})
			axs[x][0].set_ylabel(ylabels[x],fontsize=10)
			# ax1.set_ylabel("Snow drought quantities")
			axs[2][2].legend(labels=labels,loc='upper center',prop={'size': 10})
			axs[x][y].xaxis.set_major_locator(ticker.MultipleLocator(5))
			axs[x][y].tick_params(axis='x', labelsize=8)
			count += 1
			#plt.xticks(rotation=90)
	print('The output dict looks like: ')
	print(output_dict)
	print('mk dict is')
	print(mk_dict)
	stats_fn = os.path.join(output_dir,f'{huc_col}_snotel_daymet_snow_drought_counts_w_delta_swe_point_based_draft4.csv')
	output_df = pd.DataFrame(output_dict)
	if not os.path.exists(stats_fn): 
		output_df.to_csv(stats_fn)
	fig_fn = os.path.join(kwargs.get('fig_dir'),f'point_based_counts_snotel_daymet_{huc_col}_w_delta_swe_draft4.jpg')
	if not os.path.exists(fig_fn): 
		plt.savefig(fig_fn, 
			dpi=500, 
			bbox_inches = 'tight',
    		pad_inches = 0.1
    	)
	# plt.show()
	# plt.close('all')

def main(daymet_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',huc_col = 'huc8', **kwargs):
	"""Testing improved definitions of snow drought. Original definitions based on Dierauer et al 2019 but trying to introduce some more nuiance into the approach."""
	id_col = kwargs.get('id_col')
	print(f'The huc col being processed is: {id_col}')
	################################################################
	#first do the daymet data- these are csvs with each csv being a seasonal window for a water year. Stack them up  
	#read in all the files in this dir and combine them into one df
	early=FormatData(glob.glob(daymet_dir+f'*_12_{id_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	mid=FormatData(glob.glob(daymet_dir+f'*_2_{id_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	late=FormatData(glob.glob(daymet_dir+f'*_4_{id_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	################################################################
	#next do the snotel data 
	output=[]

	#read in some pickled objects, these look like a list of dfs with each being a station for the full time period 
	for item in ['PREC','TAVG','WTEQ']:
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		df=FormatData(files,drop_cols=['year','month','day']).read_in_pickles()
		output.append(df) #the df here is 365 days x ~30 yrs x 237 stations so these are pretty big dfs
	
	#join the three predictor vars 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on=['date','id']), output)
	
	#convert the temp column from F to C 
	output_df['TAVG'] = (output_df['TAVG']-32)*(5/9) 
	#there are a couple of erroneous temp values, remove those. 
	#This should maybe be amended further to get rid of very cold values as well. 
	output_df = output_df.loc[output_df['TAVG'] <= 50]

	#convert prec and swe cols from inches to cm 
	output_df['PREC'] = output_df['PREC']*2.54
	output_df['WTEQ'] = output_df['WTEQ']*2.54
	
	#remove rows that have one of the data types missing- this might need to be amended because 
	#it means that there are different numbers of records in some of the periods. 
	output_df=output_df.dropna()
	
	#cast the snotel id col to int to add the hucs 
	output_df['id'] = output_df['id'].astype('int')
	#rename to match daymet 
	output_df.rename(columns={'id':'site_num'},inplace=True)
	
	#this is changed in the pt version- still need to associate the snotel stations with a HUC4 to calculate 
	#kmeans centroids but we won't use them for anything else
	#add the as yet nonexistant hucs data to the outputs 
	hucs = kwargs.get('hucs')

	period_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
		print('season is: ',p1)
		#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)
		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991,sort_col=id_col).prepare_df_cols()
			#hucs disappear in the previous step but are still needed to create the kmeans initialization clusters, add them back in here 
			snotel_drought['huc8'] = snotel_drought['site_num'].map(hucs) #hardcoded for huc8 basins
		else: 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',sort_col=id_col).prepare_df_cols()
			snotel_drought['huc8'] = snotel_drought['site_num'].map(hucs) #hardcoded for huc8 basins
			
		#then do the same for daymet  
		if (p1 == 'mid') | (p1 == 'late'): 
			daymet_drought=CalcSnowDroughts(p2,start_year=1991,sort_col=id_col).prepare_df_cols()
			daymet_drought['huc8'] = daymet_drought['site_num'].map(hucs) #hardcoded for huc8 basins
		else: 
			daymet_drought=CalcSnowDroughts(p2,sort_col=id_col).prepare_df_cols()
			daymet_drought['huc8'] = daymet_drought['site_num'].map(hucs) #hardcoded for huc8 basins
	##########################################
	
		#run the kmeans with drought types as intiilization conditions (centroids) for the clusters
		
		#these are all of the huc 4 basins in the study area 
		huc4s = ['1708','1801','1710','1711','1709','1701','1702','1705','1703','1601','1707','1706','1712','1704']
		s_output = []
		d_output = []
		for huc4 in huc4s: 
			huc4_s = sd.prep_clusters(snotel_drought,huc4,id_col) #get the subset of the snow drought data for a given huc4
			huc4_d = sd.prep_clusters(daymet_drought,huc4,id_col)
			#make the centroids that serve as the intialization for the kmeans clusters- these are like endmembers (ish)
			s_centroids = DefineClusterCenters(huc4_s,'WTEQ','PREC','TAVG').combine_centroids() #makes a numpy array with four centroids
			d_centroids = DefineClusterCenters(huc4_d,'swe','prcp','tavg').combine_centroids() #makes a numpy array with four centroids

			#clusters should be like: {0:dry, 1:warm, 2:warm_dry, 3:no_drought} 6/8/2021 DOUBLE CHECK
			#run kmeans for the snotel data
			s_clusters = sd.run_kmeans(huc4_s[['WTEQ','PREC','TAVG']].to_numpy(),huc4_s['label'],s_centroids)
			s_clusters = sd.add_drought_cols_to_kmeans_output(s_clusters, sort_col=id_col) #add a few cols needed for plotting 
			#run kmeans for the daymet data 
			d_clusters = sd.run_kmeans(huc4_d[['swe','prcp','tavg']].to_numpy(),huc4_d['label'],d_centroids)
			d_clusters = sd.add_drought_cols_to_kmeans_output(d_clusters, sort_col=id_col) #add a few cols needed for plotting 

			s_output.append(s_clusters)
			d_output.append(d_clusters)
		s_plot = pd.concat(s_output)

		#select the cols of interest and rename so there's no confusion when dfs are merged 
		s_plot=s_plot[[id_col,'year','dry','warm','warm_dry']]
		s_plot.columns=[id_col,'year']+['s_'+column for column in s_plot.columns if not (column=='year') | (column == id_col)]

		d_plot = pd.concat(d_output)
		d_plot=d_plot[[id_col,'year','dry','warm','warm_dry']]
		d_plot.columns=[id_col,'year']+['d_'+column for column in d_plot.columns if not (column=='year') | (column == id_col)]
	
		#merge the two datasets into one df 
		dfs = s_plot.merge(d_plot,on=[id_col,'year'],how='inner')
		
		print('The dfs here are: ')
		print(dfs)

		#deal with the scenario that there are basins with less than 30 years of data, remove those here
		dfs = sd.remove_short_dataset_stations(dfs,id_col)
		print('number of stations is: ')
		print(len(list(dfs[id_col].unique())))
		period_list.append(dfs)

	#plot_counts(period_list,kwargs.get('stats_dir'),huc_col=id_col,**kwargs)

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
	
	#still need to use hucs to get the huc4 for snotel stations to calculate the kmeans 
	hucs=pd.read_csv(stations)
	
	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	print('hucs shape is: ')
	print(hucs.shape)
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	main(daymet_dir,pickles,
					id_col='site_num',
					hucs=hucs_dict,
					palette=palette,
					stats_dir=stats_dir,
					fig_dir=fig_dir)
