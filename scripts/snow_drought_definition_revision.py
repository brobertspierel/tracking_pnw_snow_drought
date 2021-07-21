import pandas as pd 
import os 
import sys 
import geopandas as gpd 
import json 
import time 
import glob
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from _1_calculate_revised_snow_drought import FormatData,CalcSnowDroughts

#supress the SettingWithCopy warning in pandas
pd.options.mode.chained_assignment = None  # default='warn'

#define a couple of custom exceptions
class CentroidSize(Exception):
    pass

class CheckColNames(Exception): 
	"""Check if correct col names are present."""
	def __init__(self,message="You likely forgot to change the param" 
							" cols to run a different type of data." 
							" Double check and run again."): 
		self.message = message
		super().__init__(self.message)

class DefineClusterCenters(): 
	"""Define methods to get the most extreme examples of snow droughts based on definitions from Dierauer et al. 2019.
	These cluster centroids are based on SWE, temp and precip and are used as the initialization conditions for a kmeans 
	algorithm which outputs each snow drought unit (basin/year/winter chunk) with associated cluster (snow drought type) 
	and a distance to that cluster centroid. Lower values mean the snow drought unit is closer to the extreme (endmember)
	"""

	def __init__(self,df,swe_col,prec_col,temp_col): 
		self.df = df 
		self.swe_col = swe_col
		self.prec_col = prec_col
		self.temp_col = temp_col

	def make_median_centroids(self,df1): 
		"""Testing a scenario where we take the median (or mean) of predictor variables for each 
		drought type and use those to define the initialization centroids for the kmeans algo. 
		"""
		out_df = df1[[self.swe_col,self.prec_col,self.temp_col]].median()
		return out_df


	def check_centroid_arr_size(self,df1): 
		"""Deal with a condition where the three values needed for the cluster centroid 
		have two or three rows. This happens when the scaled data have the same value 
		(e.g. more than one happens to have been the highest or lowest example of that variable which 
		yields 0 or 1 for multiple variables and a tie).
		"""	 
		rows = df1.shape[0]
		while True: 	
			df1 = df1.loc[df1[self.temp_col]==df1[self.temp_col].max()]
			if df1.shape[0] == 1: 
				break 
			elif df1.shape[0] > 1: #condition where df is still more than 1 row
				df1 = df1.loc[df1[self.prec_col]==df1[self.prec_col].min()]
				#there are a few instances that need to be checked where there are multiple zeros in the precip col. 
				#in those cases it means that all the rows are the same and look like [0, 0, 1]. In those instances just take 
				#the first row because they are all the same. 
				df1 = df1.head(1) 
				break
			elif df1.empty: 
				#it looks like when we go from huc8 to huc6 there are some basins that don't have one of the snow drought types ever.
				#in this case we set the initial cluster centroid to 0,0,0 which might not be the best way of doing that. 7/20/2021
				df1 = pd.DataFrame({self.swe_col:[0],self.prec_col:[0],self.temp_col:[0]})
				print('The df was empty and we fill with something that looks like: ')
				print(df1)
				break
		return df1

	def dry_sd_centroid(self): 
		dry = self.df.loc[(self.df[self.swe_col]<self.df[f'mean_{self.swe_col}']) & 
		(self.df[self.prec_col]<self.df[f'mean_{self.prec_col}'])&(self.df[self.temp_col]<=self.df[f'mean_{self.temp_col}'])]
		#dry = dry.loc[dry[self.swe_col]==dry[self.swe_col].min()][[self.swe_col,self.prec_col,self.temp_col]] #.to_numpy() #get the row that has the lowest SWE value and make it into a little numpy array 
		return self.make_median_centroids(dry).to_numpy()#self.check_centroid_arr_size(dry).to_numpy()

	def warm_sd_centroid(self): 
		warm = self.df.loc[(self.df[self.swe_col]<self.df[f'mean_{self.swe_col}']) & 
			(self.df[self.prec_col]>=self.df[f'mean_{self.prec_col}'])]
	
		#warm = warm.loc[warm[self.swe_col]==warm[self.swe_col].min()][[self.swe_col,self.prec_col,self.temp_col]]
		return self.make_median_centroids(warm).to_numpy()#self.check_centroid_arr_size(warm).to_numpy()

	def warm_dry_sd_centroid(self): 
		warm_dry = self.df.loc[(self.df[self.swe_col]<self.df[f'mean_{self.swe_col}']) & 
		(self.df[self.prec_col]<self.df[f'mean_{self.prec_col}'])&(self.df[self.temp_col]>self.df[f'mean_{self.temp_col}'])]
		#warm_dry = warm_dry.loc[warm_dry[self.swe_col]==warm_dry[self.swe_col].min()][[self.swe_col,self.prec_col,self.temp_col]]
		return self.make_median_centroids(warm_dry).to_numpy()#self.check_centroid_arr_size(warm_dry).to_numpy()

	def no_sd_centroid(self): 
		no_drought = self.df.loc[self.df[self.swe_col]==self.df[self.swe_col].max()][[self.swe_col,self.prec_col,self.temp_col]]
		return self.make_median_centroids(no_drought).to_numpy()#self.check_centroid_arr_size(no_drought).to_numpy()

	def combine_centroids(self): 
		"""Take the three drought type centroids and no drought and make them into one numpy arr. 
		This is the input to the kmeans algo.
		"""
		return np.concatenate((self.dry_sd_centroid(),self.warm_sd_centroid(),
			self.warm_dry_sd_centroid(),self.no_sd_centroid()),axis=0)

def run_kmeans(X,y,centroids): 
	"""Run the sci-kit learn kmeans algo to cluster snow drought units (year/season/basin) 
	into different snow drought types. This approaches uses the most 'extreme' case of each snow 
	drought type as initialization cluster centroids. This serves the dual purpose of greatly speeding up 
	processing and making the labeling much easier for the outputs. 
	Inputs: 
	X- array of data to be classified, in this case all years for a station for a season 
	y- label array, this is just the adjacent col in the pandas df to X

	Outputs- 
	Dataframe with the class label and the distance to the cluster centroid. 
	"""
	print('Entered the kmeans function.')
	print('the centroids look like: ')
	print(centroids)
	#make sure an extra centroid didn't sneak through: 
	if centroids.shape[0] > 4: 
		raise CentroidSize(f'The number of centroids must be less than four. You have {centroids.shape[0]}')
	kmeans = KMeans(n_clusters=4, init=centroids, max_iter=300, n_init=1, random_state=10) #centroids.shape[0]
	pred_y = kmeans.fit_predict(X)
	# squared distance to cluster center
	X_dist = kmeans.transform(X)**2

	df = pd.DataFrame(X_dist.sum(axis=1).round(2), columns=['sqdist'])
	
	df['k_label'] = y.values
	
	try: 
		df['drought_clust'] = pred_y
	except Exception as e: 
		print(f'Error here was {e}')
	# print('label df is: ')
	# print(df) #this df is the one we want to use to derive the labels. 
	return df 
	##########################
	##visualize the clusters and centroids 
	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')

	# color_dict = {0:'black',1:'darkblue',2:'purple',3:'orange'}

	# df = pd.DataFrame({'swe': X[:, 0], 'precip': X[:, 1], 'temp': X[:, 2], 'clust_label':pred_y})

	# df['colors'] = df['clust_label'].map(color_dict)
	# print('df')
	# print(df)


	# ax.scatter(X[:,0], X[:,1], X[:,2])
	# ax.scatter(df.swe, df.precip, df.temp, c=df.colors)
	
	# ax.scatter(centroids[:, 0], centroids[:, 1],centroids[:, 2],s=100, c='red')

	# ax.set_xlabel('SWE')
	# ax.set_ylabel('PRECIP')
	# ax.set_zlabel('TAVG')
	# plt.show()

def prep_clusters(df,huc4_str,huc_col='huc8'): 
	"""Prepares data for running k-means algo."""
	#make sure the huc_col is a str 
	
	df[huc_col] = df[huc_col].astype('str')
	basin_df = df.loc[df[huc_col].str.contains(huc4_str)] #get the huc8 or huc6 basins in the huc4 basin 
	basin_df['label'] = basin_df[huc_col] + '_' + basin_df['year'].astype('str') #make a label col so we can attribute pts from the kmeans

	return basin_df

def add_drought_cols_to_kmeans_output(df,huc_col='huc8'): 
	"""Take the output of the kmeans algo, split the label col into separate cols and then 
	add cols with the snow drought years so it can be made into a df for plotting 
	and additional analysis. 
	"""
	# print('The df here is: ')
	# print(df)
	#split that label col back apart so we know the basin id and the year for the cluster
	df['year'] = df['k_label'].str.split('_').str[1].astype(float)
	df[huc_col] = df['k_label'].str.split('_').str[0].astype(float)
	df.drop(columns=['k_label'],inplace=True)
	#now add the drought cols 
	df['dry'] = np.where(df['drought_clust']==0,df['year'],np.nan)
	df['warm'] = np.where(df['drought_clust']==1,df['year'],np.nan)
	df['warm_dry'] = np.where(df['drought_clust']==2,df['year'],np.nan)
	#df['dry'] = np.where(df['drought_clust']==0,df['year'],np.nan) #add the no drought col

	return df 





# print(test.shape)
	# print(test['huc8'].unique())
	# print(test)

	#do dry first- if the conditions evaluate to true add the year and if not just fill with nan 
	# sd_df['dry'] = np.where((sd_df[self.swe_c]<sd_df[f'mean_{self.swe_c}']) & 
	# 	(sd_df[self.precip]<sd_df[f'mean_{self.precip}'])&(sd_df[self.temp]<=sd_df[f'mean_{self.temp}']),sd_df['year'],np.nan)

	# #next do warm
	# sd_df['warm'] = np.where((sd_df[self.swe_c]<sd_df[f'mean_{self.swe_c}']) & 
	# 	(sd_df[self.precip]>=sd_df[f'mean_{self.precip}']),sd_df['year'],np.nan)

	# #then do warm/dry 
	# sd_df['warm_dry'] = np.where((sd_df[self.swe_c]<sd_df[f'mean_{self.swe_c}']) & 
# 	(sd_df[self.precip]<sd_df[f'mean_{self.precip}'])&(sd_df[self.temp]>sd_df[f'mean_{self.temp}']),sd_df['year'],np.nan)




	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')
	# ax.scatter(test['WTEQ'],test['PREC'],test['TAVG'])
	# ax.set_xlabel('SWE')
	# ax.set_ylabel('prec')
	# ax.set_zlabel('temp')
	# plt.show()


	# 	#get cols of interest 
	# 	snotel_drought=snotel_drought[['huc8','year','dry_sd','warm_sd','warm_dry_sd','near_drought']]
	# 	#rename cols so they don't get confused when data are merged 
	# 	snotel_drought.columns=['huc8','year']+['s_'+column for column in snotel_drought.columns if not (column =='huc8') | (column=='year')]

	# 	# print('snotel drought is: ')
	# 	# print(snotel_drought)

	# 	## all working fine, just doing some testing 6/30/2021
	# 	##then do the same for daymet  
	# 	if (p1 == 'mid') | (p1 == 'late'): 
	# 		daymet_drought=CalcSnowDroughts(p2,start_year=1991).calculate_snow_droughts()
	# 	else: 
	# 		daymet_drought=CalcSnowDroughts(p2).calculate_snow_droughts()
	# 	#print('daymet',daymet_drought)
	# 	daymet_drought=daymet_drought[['huc8','year','dry_sd','warm_sd','warm_dry_sd','near_drought']]
		
	# 	daymet_drought.columns=['huc8','year']+['d_'+column for column in daymet_drought.columns if not (column =='huc8') | (column=='year')]
	# 	print(daymet_drought)
	# 	#merge the two datasets into one df 
		

	# 	dfs = snotel_drought.merge(daymet_drought,on=['huc8','year'],how='inner')
	# 	#print('THe combined output looks like: ', dfs)
	# 	#compare each drought type and record the results in a new col 

	# 	dfs['dry']=dfs['s_dry_sd']==dfs['d_dry_sd']
	# 	dfs['warm']=dfs['s_warm_sd']==dfs['d_warm_sd']
	# 	dfs['warm_dry']=dfs['s_warm_dry_sd']==dfs['d_warm_dry_sd']

	# 	print(dfs)
	# 	#print(dfs.groupby(['huc8'])['dry','warm','warm_dry'].sum())
	# 	# pd.set_option('display.max_columns', None)
	# 	# pd.set_option('display.max_rows', None)
	# 	# #print(dfs)
	# 	period_list.append(dfs)


	# #########################################################
	# #test adding a figure 
	# labels=['Snotel','Daymet']
	# ###############################################################
	# # print('list is: ', period_list)
	# # #df = pd.DataFrame({"dry":dry_counts,"warm":warm_counts,"warm_dry":warm_dry_counts})
	# nrow=3
	# ncol=3
	# fig,axs = plt.subplots(nrow,ncol,sharex=True,sharey=True,
	# 			gridspec_kw={'wspace':0,'hspace':0,
 #                                    'top':0.95, 'bottom':0.075, 'left':0.05, 'right':0.95},
 #                figsize=(nrow*2,ncol*2))
	# cols=['dry','warm','warm_dry']
	# xlabels=['Dry', 'Warm', 'Warm/dry']
	# ylabels=['Early','Mid','Late']
	# for x in range(3): 
	# 	for y in range(3): 
	# 		print('x is: ',x)
	# 		print('y is: ',y)
	# 		print('col is: ',cols[y])
	# 		#produce the confusion matrices 
	# 		# reference_data = [float(i) for i in list(period_list[x][f's_{cols[y]}'])]
	# 		# predicted_data = [float(i) for i in list(period_list[x][f'd_{cols[y]}'])]
			
	# 		# #ids = zonal_stats_df.index
	# 		# #generate some stats
	# 		# results = confusion_matrix(reference_data, predicted_data)#,zonal_stats_df.index) 
	# 		# print(results)
			
	# 		# #ax=plt.subplot()
	# 		# sns.set(font_scale=2)  # crazy big
	# 		# sns.heatmap(results,annot=True,ax=axs[x][y],fmt='g',cmap='Blues')

	# 		# #axs[x][y].set_xlabel('Predicted labels');axs.set_ylabel('True labels')
	# 		# print(classification_report(reference_data,predicted_data))
			
	# 		#produce the count plots 
	# 		s_counts = period_list[x][f's_{cols[y]}_sd'].value_counts().sort_index().astype('int')
	# 		d_counts = period_list[x][f'd_{cols[y]}_sd'].value_counts().sort_index().astype('int')
	# 		# print(s_counts)
	# 		# print(d_counts)
	# 		df = pd.DataFrame({"snotel":s_counts,"daymet":d_counts})
	# 		#df = df.astype(int)
	# 		#reformat a few things in the df 
	# 		df.index=df.index.astype(int)
	# 		df.replace(np.nan,0,inplace=True)

	# 		#there are not droughts in all the years and timeframes but these gaps mess up plotting 
	# 		#so we want to infill them with zeros so all the timeperiods have all of the years. 
	# 		df=df.reindex(np.arange(1990,2021), fill_value=0)

	# 		# calculate the Pearson's correlation between two variables
			
	# 		# seed random number generator
	# 		#seed(1)
	# 		# prepare data
	# 		# data1 = 20 * randn(1000) + 100
	# 		# data2 = data1 + (10 * randn(1000) + 50)
	# 		# calculate Pearson's correlation
	# 		corr, _ = pearsonr(df.snotel, df.daymet)
	# 		print(f'Pearsons correlation: {corr}')

	# 		print('df is: ')
	# 		print(df)
	# 		print('########################################')
	# 		# print(s_counts)
	# 		# print(d_counts)
	# 		# s_counts.bar(ax=axs[x][y])
	# 		# d_counts.bar(ax=axs[x][y])
	# 		#set just the horizontal grid lines 
	# 		# axs[x][y].set_axisbelow(True)
	# 		# axs[x][y].yaxis.grid(color='gray', linestyle='dashed')

	# 		df.plot.bar(ax=axs[x][y],color=['#D95F0E','#267eab'],width=0.9,legend=False)#,label=['Dry','Warm','Warm/dry']) #need to do something with the color scheme here and maybe error bars? This could be better as a box plot? 
	# 		axs[x][y].grid(axis='y',alpha=0.5)

	# 		#set axis labels and annotate 
	# 		axs[x][y].annotate(f'r = {round(corr,2)}',xy=(0.05,0.9),xycoords='axes fraction',fontsize=14)
	# 		axs[0][y].set_title(xlabels[y],fontdict={'fontsize': 14})
	# 		axs[x][0].set_ylabel(ylabels[x],fontsize=14)
	# 		# ax1.set_ylabel("Snow drought quantities")
	# 		axs[2][2].legend(labels=labels,loc='upper center')
	# 		#axs[x][y].set_xticklabels(xlabels, Fontsize= )
	# 		#axs[x][y].set_xticks(range(1990,2021,5))
	# 		start, end = axs[x][y].get_xlim()
	# 		print(start,end)
	# 		#ax.xaxis.set_ticks(np.arange(start, end, stepsize))
	# 		for tick in axs[x][y].get_xticklabels():
	# 			tick.set_rotation(90)
	# 			#tick.label.set_fontsize(14) 
	# 		axs[x][y].tick_params(axis='x', labelsize=12)

	# 		#plt.xticks(rotation=90)
	# #plt.tight_layout()
	# plt.show()
	# plt.close('all')