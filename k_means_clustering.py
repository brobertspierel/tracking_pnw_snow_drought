

import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler


#code from https://tslearn.readthedocs.io/en/latest/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py
def kmeans(data,clusters,year_of_interest,njobs): 
	#get data
	seed = 5
	np.random.seed(seed)
	#print('data shape is: ',data.shape)
	X_train=data.T
	#print(X_train)
	#print('shape is ', X_train.shape, ' before scaling')
	#X_train = np.expand_dims(np.zeros(X_train.shape),axis=2)
	#print('the shape after expand dims is: ', X_train.shape)
	#print('example before: ',X_train[:,10])
	#np.random.shuffle(X_train)
	#print(X_train.shape)
	#print('x train is: ', X_train)
	#X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
	#print(X_train.shape)
	#X_train[:,:-1]=np.squeeze(TimeSeriesScalerMeanVariance().fit_transform(X_train[:,:-1]),axis=2) #changed from 2 to 1 
	#print('x train is now: ', X_train)
	X_train = np.nan_to_num(X_train)
	#print('example after is: ', X_train)
	#print('X_train shape is ', X_train.shape, ' after scaling')
	#X_train = np.squeeze(X_train, axis=2)
	#X_train_labeled = np.append(X_train,labels)
	sz = X_train.shape[1]

	# Euclidean k-means
	# print("Euclidean k-means")
	# km = TimeSeriesKMeans(n_clusters=clusters, verbose=True, random_state=seed,n_jobs=20)
	# #print('km is ',km)
	# y_pred = km.fit_predict(np.nan_to_num(X_train))#[:,0][0])
	# #print(y_pred)
	# #print('y_pred is',y_pred)
	# cluster_dict = {}
	# cluster_centers = {}
	# plt.figure(figsize=(10,10))
	# #uncomment calls to plot if you want to see the figures
	# for yi in range(clusters):
	#     #print('cluster is: ', yi+1)
	#     time_series = {} #changed from list to dict
	#     plt.subplot(10, 5, yi + 1)
	#     #count = 0 
	#     for xx in X_train[y_pred == yi]:
	#         #time_series.append(xx[-1]) #removed [-1]
	#         time_series.update({xx[-1]:xx})
	#         #print('xxshape is ',xx.shape)
	#         #print('count is: ',count)
	#         #count +=1
	#         #print(xx[1])
	#         plt.plot(xx.ravel(), "darkblue", alpha=.2)
	#     cluster_dict.update({f'cluster_{yi+1}':time_series})
	#     cluster_centers.update({f'cluster_{yi+1}':km.cluster_centers_[yi]})

	#     plt.plot(km.cluster_centers_[yi].ravel(), "r-")
	#     #print(km.cluster_centers_[yi].shape)
	#     plt.xlim(0, sz)
	#     plt.ylim(-10, 10)
	#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
	#              transform=plt.gca().transAxes)
	#     if yi == 1:
	#         plt.title("Euclidean $k$-means")
	        

	#print(cluster_dict)
	#cluster_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in cluster_dict.items() ]))
	#print(cluster_df.iloc[:,0][0])
	# DBA-k-means
	print("DBA k-means")
	dba_km = TimeSeriesKMeans(n_clusters=clusters,
	                          n_init=3,
	                          metric="dtw",
	                          verbose=True,
	                          max_iter_barycenter=10,
	                          random_state=seed,
	                          n_jobs=njobs)
	y_pred = dba_km.fit_predict(np.nan_to_num(X_train))
	cluster_dict = {}
	cluster_centers = {}
	plt.figure(figsize=(10,10))

	for yi in range(clusters):
	    #print('the cluster number is: ', yi+1)
	    time_series = {} #changed from list to dict
	    plt.subplot(10, 5, yi + 1)

	    #plt.subplot(clusters, clusters, (clusters+1) + yi)
	    for xx in X_train[y_pred == yi]:
	        #print('the time series in this cluster look like: ',xx)
	        time_series.update({xx[-1]:xx})
	        plt.plot(xx.ravel(), "darkblue", alpha=.2)
	    cluster_dict.update({f'cluster_{yi+1}':time_series})
	    cluster_centers.update({f'cluster_{yi+1}':dba_km.cluster_centers_[yi]})
	    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
	    plt.xlim(0, sz)
	    plt.ylim(0, 1) #changed from -10,10
	    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
	             transform=plt.gca().transAxes)
	    if yi == 1:
	        plt.title(f"DBA $k$-means {year_of_interest}")

	# #Soft-DTW-k-means
	# print("Soft-DTW k-means")
	# sdtw_km = TimeSeriesKMeans(n_clusters=clusters,
	#                            metric="softdtw",
	#                            metric_params={"gamma": .01},
	#                            verbose=True,
	#                            random_state=seed,
	#                            n_jobs=10)
	# y_pred = sdtw_km.fit_predict(np.nan_to_num(X_train))

	# for yi in range(clusters):
	#     plt.subplot(clusters, clusters, ((clusters*2)+1) + yi)
	#     for xx in X_train[y_pred == yi]:
	#         plt.plot(xx.ravel(), "k-", alpha=.2)
	#     plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
	#     plt.xlim(0, sz)
	#     plt.ylim(-10, 10)
	#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
	#              transform=plt.gca().transAxes)
	#     if yi == 1:
	#         plt.title("Soft-DTW $k$-means")

	plt.tight_layout()
	#plt.show()
	#plt.close(fig)
	plt.clf()
	plt.close('all')
	#print(cluster_centers)
	return cluster_dict,cluster_centers
def clean_headers(input_str): 
	#there is a 0 placeholder at the beginning of the number to deal with some stations being three and some four digits, remove that 
	#print('input STRING IS: ',f'{input_str:.12f}')
	formatted_str = f'{input_str:.12f}'
	if formatted_str[6]=='0': 
		station_id=formatted_str[7:10]
		#print('if station id is now: ',station_id)
	elif formatted_str[6]!='0': 
		station_id=formatted_str[6:10]
		#print('elif station id is now: ', station_id)
	else: 
		print('something is wrong with the inputs')

	year = formatted_str[10:]
	#print('year is ',year)
	#fix years that lost zeros when being converted to float (i.e. 2000, 2010)
	if len(year) ==1: 

		year = year+'000'
	elif len(year)<=3: 
		
		year = year+'0'
	else: 
		pass
	return station_id+'_'+year
def prepare_cluster_outputs(input_dict): 
	
	inter_dict={}

	for k,v in input_dict.items():
		#check that you're getting what you think you're getting
		if not float(next(iter(v.keys()))): 
	# # values_view = a_dict.values()
	# # value_iterator = iter(values_view)
	# first_value = next(value_iterator)
			print('something is wrong with the input')
			raise Exception 
		else: 
			station_dict = {}
			#v is a list of combined ids and years (e.g. .003441998), split it
			#for i,j in v.items(): 
			cluster_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in v.items() ]))
			cluster_df.rename(columns=lambda x: clean_headers(x), inplace=True)

		inter_dict.update({k:cluster_df.iloc[:-1]})
	#print('inter_dict: ',inter_dict)
	return inter_dict

def split_df(df,sites_list): 
	output_dict = {}
	rows_list = []
	#rint(df.head())
	#new_df = pd.DataFrame(columns=['station_id','count','lat','lon'])
	for column in list(df.columns): 
		inter_dict = {}
		#print('COLUMN IS: ', column)
		#print(df[column].count())
		#new_df['station_id'] = 'example'#str(column)
		#new_df['count'] = int(df[column].count())
		count = df[column].count()
		#print ('count is: ',count)
		lat = pd.Series(sites_list.loc[sites_list['site num']==column]['lat']).iloc[0]
		#print ('lat is: ', lat)
		lon = pd.Series(sites_list.loc[sites_list['site num']==column]['lon']).iloc[0]
		#print ('lon is: ', lon)
		#print ('column is: ', column)
		inter_dict.update({'site num':int(str(column)),'count':count,'lat':lat,'lon':lon})
		#print(inter_dict)
		rows_list.append(inter_dict)
			#print(new_df.head())
	output_df=pd.DataFrame(rows_list)#,columns=['station_id','count','lat','lon'])
	
	#print('output_df: ',output_df)
	return output_df


def build_mapping_inputs(input_dict,sites_list): 
	#stations = list(next(iter(input_dict.values())).columns)
	output_dict = {}
	#iterate through the result of prepare_cluster_outputs
	for k,v in input_dict.items(): 
		inter_dict = {}
		stations = list(v.columns) #these are still id_year format
		#from https://stackoverflow.com/questions/5706735/pythonic-way-to-split-string-into-two-lists
		ids,yrs=map(list, zip(*(s.split("_") for s in stations))) #split the col headers
		for i,j in zip(ids,yrs): 
			yrs_list = []
			if not i in inter_dict.keys():
				#add to the list of years for the first time
				yrs_list.append(j) 
				#put the first instance of a id into the keys
				inter_dict.update({i:yrs_list})
			elif i in inter_dict.keys(): 
				#add to the years list
				yrs_list.append(j)
				existing_list = inter_dict[i]
				inter_dict[i] = existing_list+yrs_list
			else: 
				raise Exception('Something was wrong with the inputs')
		cluster_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inter_dict.items() ]))
		output_dict.update({k:cluster_df})
	final_dict = {}
	for m,n in output_dict.items(): 
		inter_df = split_df(n,sites_list)
		#print('M is now: ',m)
		final_dict.update({m:inter_df})
	
	return final_dict