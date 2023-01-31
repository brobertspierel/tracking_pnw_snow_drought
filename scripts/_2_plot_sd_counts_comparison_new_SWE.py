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
from _1_calculate_revised_snow_drought_new_SWE import FormatData,CalcSnowDroughts
import snow_drought_definition_revision_new_SWE as sd
from snow_drought_definition_revision_new_SWE import DefineClusterCenters
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.ticker as ticker
import pymannkendall as mk
from scipy import stats

def mk_test(input_data): 
	"""Run a version of the Mann-Kendall trend test."""

	trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(input_data)

	return trend, h, p, z, Tau, s, var_s, slope, intercept


def plot_counts(df_list,output_dir,huc_col='huc8',**kwargs): 
	print('Entered the plotting function: ')
	labels=['Snotel','UA SWE']
	
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
			ua_counts = df_list[x][f'ua_{cols[y]}'].value_counts().sort_index().astype('int')
			# print(s_counts)
			# print(ua_counts)
			df = pd.DataFrame({"snotel":s_counts,"ua_swe":ua_counts})
			#reformat a few things in the df 
			df.index=df.index.astype(int)
			df.replace(np.nan,0,inplace=True)

			#there are not droughts in all the years and timeframes but these gaps mess up plotting 
			#so we want to infill them with zeros so all the timeperiods have all of the years. 
			df=df.reindex(np.arange(1991,2021), fill_value=0)
			mk_dict.update({f's_{ylabels[x]}_{xlabels[y]}':mk_test(df.snotel)[0],
				f'ua_{ylabels[x]}_{xlabels[y]}':mk_test(df.ua_swe)[0]})
			#add the counts to a dict so we can output the actual counts and look at them 
			output_dict.update({f's_{ylabels[x]}_{xlabels[y]}':df['snotel'],f'ua_{ylabels[x]}_{xlabels[y]}':df['ua_swe']})

			# calculate Pearson's correlation
			corr, _ = pearsonr(df.snotel, df.ua_swe)
			rho, pval = stats.spearmanr(df.snotel, df.ua_swe)
			print(f'Pearsons correlation: {corr}')

			df.plot.bar(ax=axs[x][y],color=['#D95F0E','#267eab'],width=0.9,legend=False)#,label=['Dry','Warm','Warm/dry']) #need to do something with the color scheme here and maybe error bars? This could be better as a box plot? 
			axs[x][y].grid(axis='both',alpha=0.25)

			#set axis labels and annotate 
			axs[x][y].annotate(f'\u03C1 = {round(rho,2)}',xy=(0.05,0.9),xycoords='axes fraction',fontsize=10)
			#add a subplot letter id 
			axs[x][y].annotate(f'{chr(97+count)}',xy=(0.85,0.9),xycoords='axes fraction',fontsize=10,weight='bold')#f'{chr(97)}'
			axs[0][y].set_title(xlabels[y],fontdict={'fontsize': 10})
			axs[x][0].set_ylabel(ylabels[x],fontsize=10)
			# ax1.set_ylabel("Snow drought quantities")
			axs[2][2].legend(labels=labels,loc='center right',prop={'size': 10})
			axs[x][y].xaxis.set_major_locator(ticker.MultipleLocator(5))
			# for tick in axs[x][y].get_xticklabels():
			# 	tick.set_rotation(90)
			# 	print('tick is: ',tick)
			# 	print(type(tick))
			# 	#tick.label.set_fontsize(14) 
			axs[x][y].tick_params(axis='x', labelsize=8)
			count += 1
	fig.text(0.025, 0.5, 'Snow drought count', va='center', rotation='vertical')

	print('mk dict is')
	print(mk_dict)
	stats_fn = os.path.join(output_dir,f'{huc_col}_ua_swe_snotel_snow_drought_counts_spearman_w_delta_swe_proj_final1.csv')
	output_df = pd.DataFrame(output_dict)
	if not os.path.exists(stats_fn): 
		output_df.to_csv(stats_fn)
	fig_fn = os.path.join(kwargs.get('fig_dir'),f'ua_swe_counts_snotel_{huc_col}_spearman_w_delta_swe_proj_final_new_legend.jpg')
	if not os.path.exists(fig_fn): 
		plt.savefig(fig_fn, 
			dpi=500, 
			bbox_inches = 'tight',
    		pad_inches = 0.1
    	)
	# plt.show()
	# plt.close('all')

def main(model_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',huc_col = 'huc8', **kwargs):
	"""Testing improved definitions of snow drought. Original definitions based on Dierauer et al 2019 but trying to introduce some more nuance into the approach."""
	print(f'The huc col being processed is: {huc_col}')
	################################################################
	#first do the UA swe data - this is now (9/20/2021) in two different files, one from UA SWE and one from PRISM. 
	#These are combined in _0_combine_ua_swe_w_prism.py
	early=FormatData(glob.glob(model_dir+f'*_12_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	mid=FormatData(glob.glob(model_dir+f'*_2_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	late=FormatData(glob.glob(model_dir+f'*_4_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	################################################################
	#next do the snotel data 
	output=[]
	print('early')
	print(early)
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
	#there are a couple of erroneous temp values, remove those 
	output_df = output_df.loc[output_df['TAVG'] < 50]
	output_df = output_df.loc[output_df['TAVG'] > -40]

	#convert prec and swe cols from inches to mm 
	output_df['PREC'] = output_df['PREC']*25.4
	output_df['WTEQ'] = output_df['WTEQ']*25.4
	
	#remove rows that have one of the data types missing- this might need to be amended because 
	#it means that there are different numbers of records in some of the periods. 
	output_df=output_df.dropna() #commented out 9/21/2021
	
	#cast the snotel id col to int to add the hucs 
	output_df['id'] = output_df['id'].astype('int')

	#add the as yet nonexistant hucs data to the outputs 
	hucs = kwargs.get('hucs')
	output_df[huc_col] = output_df['id'].map(hucs)
	#just doing this to get the snotel ids that we're using in later steps before we take the mean below 
	id_df = output_df
	#there are multiple snotel stations in some of the basins, 
	#combine those so there is just one number per basin like the 
	#ua swe and RS data. 

	output_df=output_df.groupby([huc_col,'date'])[['PREC','WTEQ','TAVG']].mean().reset_index()

	period_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
		#get snotel first
		#make a temporal chunk of data- this is all seasonal windows irrespective of the year 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)
		
		#####
		#####
		#new for UA SWE- there are some basins which extend into Canada but PRISM and UA SWE do not. 
		#as of 9/22/2021 these basins are removed because there are no stats for the area north of the border. 
		if huc_col == 'huc8': 
			snotel_chunk = snotel_chunk.loc[~snotel_chunk[huc_col].isin(kwargs.get('remove_ids'))]
			p2 = p2.loc[~p2[huc_col].isin(kwargs.get('remove_ids'))]
		else: 
			pass
		# print('Snotel chunk looks like: ')
		# pd.set_option("display.max_rows", None, "display.max_columns", None)
		# print(snotel_chunk.head(100))
		##########working below here
		############################
		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			print('processing snotel')
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991,sort_col=huc_col).prepare_df_cols()
		else: 
			print('processing snotel')
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',sort_col=huc_col).prepare_df_cols()
		
		#then do the same for ua SWE  
		if (p1 == 'mid') | (p1 == 'late'):
			print('processing uaswe') 
			ua_swe_drought=CalcSnowDroughts(p2,start_year=1991,sort_col=huc_col).prepare_df_cols()
		else: 
			print('processing uaswe')
			ua_swe_drought=CalcSnowDroughts(p2,sort_col=huc_col).prepare_df_cols()

	##########################################
	
		#run the kmeans with drought types as intiilization conditions (centroids) for the clusters
		
		#these are all of the huc 4 basins in the study area 
		huc4s = ['1708','1801','1710','1711','1709','1701','1702','1705','1703','1601','1707','1706','1712','1704']
		#use a subset to run a test for maritime snowpack
		#huc4s = ['1711','1709','1702','1703','1708','1707','1703','1702']
		#use a subset to run an alpine(ish) snowpack
		#huc4s = ['1704','1706','1601','1705']
		s_output = []
		ua_output = []
		for huc4 in huc4s: 
			huc4_s = sd.prep_clusters(snotel_drought,huc4,p1,huc_col=huc_col) #get the subset of the snow drought data for a given huc4
			huc4_ua = sd.prep_clusters(ua_swe_drought,huc4,p1,huc_col=huc_col) #period added 9/21/2021 to make sure these get attributed to the correct water year
			#make the centroids that serve as the intialization for the kmeans clusters- these are like endmembers (ish)
			s_centroids = DefineClusterCenters(huc4_s,'WTEQ','PREC','TAVG').combine_centroids() #makes a numpy array with four centroids
			ua_centroids = DefineClusterCenters(huc4_ua,'swe','ppt','tmean').combine_centroids() #makes a numpy array with four centroids

			#clusters should be like: {0:dry, 1:warm, 2:warm_dry, 3:no_drought} 6/8/2021 DOUBLE CHECK
			#run kmeans for the snotel data
			s_clusters = sd.run_kmeans(huc4_s[['WTEQ','PREC','TAVG']].to_numpy(),huc4_s['label'],s_centroids)
			s_clusters = sd.add_drought_cols_to_kmeans_output(s_clusters, huc_col=huc_col) #add a few cols needed for plotting 
			#run kmeans for the daymet data 
			ua_clusters = sd.run_kmeans(huc4_ua[['swe','ppt','tmean']].to_numpy(),huc4_ua['label'],ua_centroids)
			ua_clusters = sd.add_drought_cols_to_kmeans_output(ua_clusters, huc_col=huc_col) #add a few cols needed for plotting 

			s_output.append(s_clusters)
			ua_output.append(ua_clusters)
		s_plot = pd.concat(s_output)

		#select the cols of interest and rename so there's no confusion when dfs are merged 
		s_plot=s_plot[[huc_col,'year','dry','warm','warm_dry']]
		s_plot.columns=[huc_col,'year']+['s_'+column for column in s_plot.columns if not (column == huc_col) | (column=='year')]

		ua_plot = pd.concat(ua_output)
		ua_plot=ua_plot[[huc_col,'year','dry','warm','warm_dry']]
		ua_plot.columns=[huc_col,'year']+['ua_'+column for column in ua_plot.columns if not (column == huc_col) | (column=='year')]
	
		#merge the two datasets into one df 
		dfs = s_plot.merge(ua_plot,on=[huc_col,'year'],how='inner')
		# print('The final ds looks like: ')
		# print(dfs.head(50))
		#deal with the scenario that there are basins with less than 30 years of data, remove those here
		dfs = sd.remove_short_dataset_stations(dfs,huc_col)
		print()
		print(f'There are now {len(dfs[huc_col].unique())} unique stations in the seasonal window')
		hucs=list(dfs[huc_col].astype(int).unique())
		print(hucs)
		print(len(hucs))
		# test = id_df.loc[id_df['huc8'].isin(dfs['huc8'].astype(int).unique())]['id'].unique()
		# print(test)
		period_list.append(dfs)

	plot_counts(period_list,kwargs.get('stats_dir'),huc_col=huc_col,**kwargs)

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		pickles=variables['pickles']
		stations=variables['stations']
		ua_swe_dir=variables['ua_swe_dir']
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
	remove_ids = [17110004,
				17110001,
				17110005,
				17020006,
				17020002,
				17020001,
				17010216,
				17010215,
				17010104]

	# maritime_test = [17110012,
	# 			17110010,
	# 			17110009,
	# 			17110013,
	# 			17110014,
	# 			17110015,
	# 			17090011,
	# 			17070204,
	# 			17020009,
	# 			17030001,
	# 			17030002,
	# 			17080004,
	# 			17080005,
	# 			17080002,
	# 			17070106,
	# 			17030003,
	# 			17020010,
	# 			17020011,
	# 			17080001,
	# 			17070105,
	# 			17070305,
	# 			17070301,
	# 			17090004,
	# 			17070304,
	# 			17070306,
	# 			17090005,
	# 			17090006]
	# alpine_test = [
	# 			17040207,
	# 			17040205,
	# 			17040208,
	# 			17040218,
	# 			17040217,
	# 			17060201,
	# 			17040214,
	# 			17060204,
	# 			17060203,
	# 			17040204,
	# 			17040202,
	# 			17040209,
	# 			17050120,
	# 			17050111,
	# 			17040219,
	# 			17050113,
	# 			17040220,
	# 			17040221,
	# 			17040212,
	# 			17040213,
	# 			16010202,
	# 			16010204,
	# 			16010201,
	# 			]

	main(ua_swe_dir,pickles,
		huc_col='huc8',
		hucs=hucs_dict,
		stats_dir=stats_dir,
		fig_dir=fig_dir,
		remove_ids=remove_ids
		#test_ids = maritime_test
		)
