import pandas as pd 
import os 
import sys 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.cluster import KMeans

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
	Input: 
	df - a df for one huc4 for all the years in the record 
	col names- these are specific to the dataset but should be for SWE, precip and temp
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
		out_df = df1[[self.swe_col,self.prec_col,self.temp_col]].mean()
		out_df = out_df.to_frame().T

		if not out_df.empty: 
			#there are some instances where there are no cases of a drought 
			#type in a given basin. In those cases fill the centroid with an arbitrarily 
			#high number so that nothing gets assigned to it. The condition should go to no 
			#drought but we need to keep four distinct centroids or the kmeans won't work. 
			out_df.fillna(9999, inplace=True)
			return out_df

		else: 
			print('Dealing with a circumstance where there are no instances of that drought type in this basin')
			#there are some circumstances where a huc4 basin has no instances of a drought type. 
			return pd.DataFrame({self.swe_col:[9999],self.prec_col:[9999],self.temp_col:[9999]})  

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
		return self.make_median_centroids(dry).to_numpy() #self.check_centroid_arr_size(dry).to_numpy() 

	def warm_sd_centroid(self): 
		warm = self.df.loc[(self.df[self.swe_col]<self.df[f'mean_{self.swe_col}']) & 
			(self.df[self.prec_col]>=self.df[f'mean_{self.prec_col}'])]
		#warm = warm.loc[warm[self.swe_col]==warm[self.swe_col].min()][[self.swe_col,self.prec_col,self.temp_col]]
		return self.make_median_centroids(warm).to_numpy()#self.check_centroid_arr_size(warm).to_numpy() 

	def warm_dry_sd_centroid(self): 
		warm_dry = self.df.loc[(self.df[self.swe_col]<self.df[f'mean_{self.swe_col}']) & 
		(self.df[self.prec_col]<self.df[f'mean_{self.prec_col}'])&(self.df[self.temp_col]>self.df[f'mean_{self.temp_col}'])]
		#warm_dry = warm_dry.loc[warm_dry[self.swe_col]==warm_dry[self.swe_col].min()][[self.swe_col,self.prec_col,self.temp_col]]
		return self.make_median_centroids(warm_dry).to_numpy() #self.check_centroid_arr_size(warm_dry).to_numpy() #self.make_median_centroids(warm_dry).to_numpy()

	def no_sd_centroid(self): 
		no_drought = self.df.loc[self.df[self.swe_col]==self.df[self.swe_col].max()][[self.swe_col,self.prec_col,self.temp_col]]
		return self.make_median_centroids(no_drought).to_numpy()#self.check_centroid_arr_size(no_drought).to_numpy() #self.make_median_centroids(no_drought).to_numpy()

	def combine_centroids(self): 
		"""Take the three drought type centroids and no drought and make them into one numpy arr. 
		This is the input to the kmeans algo.
		"""
		return np.concatenate((self.dry_sd_centroid(),self.warm_sd_centroid(),
			self.warm_dry_sd_centroid(),self.no_sd_centroid()),axis=0)

def run_kmeans(X,y,centroids): 
	"""Run the sci-kit learn kmeans algo to cluster snow drought units (year/season/basin) 
	into different snow drought types. This approaches uses the mean case of each snow 
	drought type as initialization cluster centroids. This serves the dual purpose of greatly speeding up 
	processing and making the labeling much easier for the outputs. 
	Inputs: 
	X- array of data to be classified, in this case all years for a station for a season 
	y- label array, this is just the adjacent col in the pandas df to X

	Outputs- 
	Dataframe with the class label and the distance to the cluster centroid. 
	"""
	#make sure an extra centroid didn't sneak through, they can be sneaky
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
	#this df is the one we want to use to derive the labels. 
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

def prep_clusters(df,huc4_str,sort_col,huc_col='huc8'): 
	"""Prepares data for running k-means algo."""
	#cast these cols as str so they can be concatenated into a unique id
	df[huc_col] = df[huc_col].astype('str')
	df[sort_col] = df[sort_col].astype('str')
	basin_df = df.loc[df[huc_col].str.contains(huc4_str)] #get the huc8 or huc6 basins in the huc4 basin 
	basin_df['label'] = basin_df[sort_col] + '_' + basin_df['year'].astype('str') #make a label col so we can attribute pts from the kmeans
	return basin_df

def add_drought_cols_to_kmeans_output(df,sort_col,huc_col='huc8'): 
	"""Take the output of the kmeans algo, split the label col into separate cols and then 
	add cols with the snow drought years so it can be made into a df for plotting 
	and additional analysis. 
	"""
	#split that label col back apart so we know the basin id and the year for the cluster
	df['year'] = df['k_label'].str.split('_').str[1].astype(float)
	df[sort_col] = df['k_label'].str.split('_').str[0].astype(float)
	#get rid of that col for subsequent analysis 
	df.drop(columns=['k_label'],inplace=True)
	#now add the drought cols from numeric label
	df['dry'] = np.where(df['drought_clust']==0,df['year'],np.nan)
	df['warm'] = np.where(df['drought_clust']==1,df['year'],np.nan)
	df['warm_dry'] = np.where(df['drought_clust']==2,df['year'],np.nan)
	#df['no_drought'] = np.where(df['drought_clust']==0,df['year'],np.nan) #add the no drought col

	return df 


def remove_short_dataset_stations(input_df,sort_col): 
	"""There is a scenario where some of the stations don't have 30 years of data. In 
	these cases the data record starts after the start of the study time period. Remove these
	stations because otherwise we're adding stations throughout the study time period and this could impact results. 
	"""
	input_df[sort_col] = input_df[sort_col].astype(int)
	counts=pd.DataFrame(input_df.groupby([sort_col])['year'].count().reset_index())
	counts = counts.loc[counts['year']>=30]
	return input_df.loc[input_df[sort_col].isin(list(counts[sort_col]))]