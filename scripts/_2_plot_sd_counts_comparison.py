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

		#merge the two datasets into one df 
		dfs = snotel_drought.merge(daymet_drought,on=['huc8','year'],how='inner')
		#print('THe combined output looks like: ', dfs)
		#compare each drought type and record the results in a new col 

		dfs['dry']=dfs['s_dry']==dfs['d_dry']
		dfs['warm']=dfs['s_warm']==dfs['d_dry']
		dfs['warm_dry']=dfs['s_warm_dry']==dfs['d_warm_dry']

		#print(dfs.groupby(['huc8'])['dry','warm','warm_dry'].sum())
		# pd.set_option('display.max_columns', None)
		# pd.set_option('display.max_rows', None)
		# #print(dfs)
		period_list.append(dfs)
		
	labels=['Snotel','Daymet']
	# # ###############################################################
	# print('list is: ', period_list)
	# #df = pd.DataFrame({"dry":dry_counts,"warm":warm_counts,"warm_dry":warm_dry_counts})
	nrow=3
	ncol=3
	fig,axs = plt.subplots(nrow,ncol,sharex=True,sharey=True,
				gridspec_kw={'wspace':0,'hspace':0,
                                    'top':0.95, 'bottom':0.05, 'left':0.05, 'right':0.95},
                figsize=(nrow*2,ncol*2))
	cols=['dry','warm','warm_dry']
	xlabels=['Dry', 'Warm', 'Warm/dry']
	ylabels=['Early','Mid','Late']
	for x in range(3): 
		for y in range(3): 
			print('x is: ',x)
			print('y is: ',y)
			print('col is: ',cols[y])
			#produce the confusion matrices 
			# reference_data = [float(i) for i in list(period_list[x][f's_{cols[y]}'])]
			# predicted_data = [float(i) for i in list(period_list[x][f'd_{cols[y]}'])]
			
			# #ids = zonal_stats_df.index
			# #generate some stats
			# results = confusion_matrix(reference_data, predicted_data)#,zonal_stats_df.index) 
			# print(results)
			
			# #ax=plt.subplot()
			# sns.set(font_scale=2)  # crazy big
			# sns.heatmap(results,annot=True,ax=axs[x][y],fmt='g',cmap='Blues')

			# #axs[x][y].set_xlabel('Predicted labels');axs.set_ylabel('True labels')
			# print(classification_report(reference_data,predicted_data))
			
			#produce the count plots 
			s_counts = period_list[x][f's_{cols[y]}'].value_counts().sort_index().astype('int')
			d_counts = period_list[x][f'd_{cols[y]}'].value_counts().sort_index().astype('int')
			print(s_counts)
			print(d_counts)
			df = pd.DataFrame({"snotel":s_counts,"daymet":d_counts})
			#df = df.astype(int)
			#reformat a few things in the df 
			df.index=df.index.astype(int)
			df.replace(np.nan,0,inplace=True)

			#there are not droughts in all the years and timeframes but these gaps mess up plotting 
			#so we want to infill them with zeros so all the timeperiods have all of the years. 
			df=df.reindex(np.arange(1990,2021), fill_value=0)

			# calculate the Pearson's correlation between two variables
			
			# seed random number generator
			#seed(1)
			# prepare data
			# data1 = 20 * randn(1000) + 100
			# data2 = data1 + (10 * randn(1000) + 50)
			# calculate Pearson's correlation
			corr, _ = pearsonr(df.snotel, df.daymet)
			print(f'Pearsons correlation: {corr}')

			print('df is: ')
			print(df)
			print('########################################')
			# print(s_counts)
			# print(d_counts)
			# s_counts.bar(ax=axs[x][y])
			# d_counts.bar(ax=axs[x][y])
			#set just the horizontal grid lines 
			# axs[x][y].set_axisbelow(True)
			# axs[x][y].yaxis.grid(color='gray', linestyle='dashed')

			df.plot.bar(ax=axs[x][y],color=['#FD9114','#2675a7'],width=0.9,legend=False)#,label=['Dry','Warm','Warm/dry']) #need to do something with the color scheme here and maybe error bars? This could be better as a box plot? 
			axs[x][y].grid(axis='y',alpha=0.5)

			#set axis labels and annotate 
			axs[x][y].annotate(f'r = {round(corr,2)}',xy=(0.05,0.9),xycoords='axes fraction')
			axs[0][y].set_title(xlabels[y])
			axs[x][0].set_ylabel(ylabels[x])
			# ax1.set_ylabel("Snow drought quantities")
			axs[2][2].legend(labels=labels,loc='upper center')
			#axs[x][y].set_xticks(range(1990,2021))
			for tick in axs[x][y].get_xticklabels():
				tick.set_rotation(90)
			#plt.xticks(rotation=90)
	#plt.tight_layout()
	plt.show()
	plt.close('all')
	
if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		pickles=variables['pickles']
		stations=variables['stations']
		daymet_dir=variables['daymet_dir']
		palette=variables['palette']
	
	hucs=pd.read_csv(stations)

	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	main(daymet_dir,pickles,hucs=hucs_dict,palette=palette)