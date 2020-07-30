#this script is the master for calculating snow intermittence from snotel data. it requires snotel_intermittence_V4.py because that has the key functions. 


#import modules and functions from the other intermittence function script
import pandas as pd 
import os
from pathlib import Path
import snotel_intermittence_functions as combine
import multiprocessing as mp
import sys
import numpy as np
from time import time 
import k_means_clustering as kmeans
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cm
import matplotlib as mpl

############################################################################################
############################################################################################
############################################################################################
#assign some global variables
path = Path('/vol/v1/general_files/user_files/ben/')

#uncomment to run other things- just getting the data that we dont need right now
#sites_ids = combine.site_list(path/'oregon_snotel_sites.csv')[1] #this is getting a list of just the snotel site ids. You can grab a list slice to restrict how big results gets below. 
#######################################
#uncomment when running the full archive 
sites_full = combine.site_list(path/'oregon_snotel_sites.csv')[0] #this is getting the full df of oregon (right now) snotel sites
#print(type(sites_full))
station_list = pd.read_csv(path/'stations_of_interest.csv')
station_list = station_list['oregon'].dropna()
station_list = station_list.tolist()
station_list = [str(int(i)) for i in station_list] 
parameter = 'WTEQ' 
new_parameter = parameter+'_scaled'
start_date = "1985-10-01"  
end_date = "2019-09-30" 
state = sites_full['state'][0] #this is just looking into the df created from the sites and grabbing the state id. This is used to query the specific station
change_type='scaled' 
station = "526:OR:SNTL" 
###################
#IMPORTANT
###################
#this is currently set up so that pickle_results is the function that is hitting the snotel API. specifying the True/False argument
#dictates whether it pickles the output or just saves in memory. To get a full archive, more sites etc. that line needs to be 
#uncommented. 
def obtain_data(bool,version,filepath,filename): 
	"""Unpickles pickled snotel data from the snotel API."""
	if bool: 
		pickle_results=combine.snotel_compiler(sites_ids,state,parameter,start_date,end_date,True,version) #this generates a list of dataframes of all of the snotel stations that have data for a given state
		results=combine.pickle_opener(version,state,filepath,filename)
		return results
	else: 
		results=combine.pickle_opener(version,state,filepath,filename)
		#print (len(results))
		return results
def transform_data(input_data,year_of_interest,season): 
	arrs = []
	
	for k,v in input_data.items(): 
		#print('k is: ', k)
		#print('v is: ', v.columns)
		#select (or not) a year of interest 
		#add decimal place
		if len(k) == 3: 
			k = '.00000'+k 
		elif len(k) == 4: 
			k='.0000'+k
		else: 
			print(f'something is wrong with that key which is {k}')
			break 
		#print('v shape is: ', v.shape)
		try: 
			if year_of_interest.isnumeric(): 
				df_slice = pd.DataFrame(v[year_of_interest])
				if not season == 'resample': 
					df = v.append(pd.Series([float(k+str(i)) for i in v.columns],index=df_slice.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
				elif season == 'resample': 
					rows = list(range(53-df_slice.shape[0]))
					df = df_slice.loc[df_slice.index.tolist() + rows]
					df = df.append(pd.Series([float(k+str(i)) for i in df.columns],index=df.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
				else: 	
					print('that is not a valid parameter for "season". Valid params are: core_winter, spring or resample')
					break
				arrs.append(df.to_numpy())
			else: 
				if not season == 'resample': 
					df = v.append(pd.Series([float(k+str(i)) for i in v.columns],index=v.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
				elif season == 'resample': 
					rows = list(range(53-v.shape[0]))
					df = v.loc[v.index.tolist() + rows]
					df = df.append(pd.Series([float(k+str(i)) for i in df.columns],index=df.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
				else: 	
					print('that is not a valid parameter for "season". Valid params are: core_winter, spring or resample')
					break
				arrs.append(df.to_numpy())
		except KeyError: 
			print('that station does not have a year that early (or late)')
			continue
		# if not season.lower() == 'core_winter' or season.lower() == 'spring': 
		# 	print(df.shape)
			
		# 	#df = np.pad(df, ((0,rows),(0,cols)),mode='constant',constant_values=np.nan)
		# 	arrs.append(df.to_numpy())#np.pad(df.to_numpy(), ((0,rows),(0,0)),mode='constant',constant_values=np.nan))
		# else: 
		# 	print(df.shape)
		# 	arrs.append(df.to_numpy())
		
	arr_out= np.concatenate(tuple(arrs),axis=1)
	return arr_out

def main():
	"""Master function for snotel intermittence from SNOTEL and RS data."""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		state_shapefile = variables["state_shapefile"]
		pnw_shapefile = variables["pnw_shapefile"]
		epsg = variables["epsg"]
		output_filepath=variables["output_filepath"]
		cell_size=variables["cell_size"]
		num_clusters=int(variables["num_clusters"])
		season = variables["season"]
		wetter_year = variables["wetter_year"]
		dryer_year = variables["dryer_year"]
		read_from_pickle = variables["read_from_pickle"]
		pickle_it = variables["pickle_it"]
	#run_prep_training = sys.argv[1].lower() == 'true' 
	
	####################
	#uncomment to run the full archive 
	results = obtain_data(False,1,path,f'{state}_snotel_data_list_1')
	#print([i[parameter].max() for i in results])
	#print(results)
	water_years=combine.prepare_clustering(results,parameter,new_parameter,start_date,end_date,season)
	#print(water_years.shape)
	#print(water_years)

	for k,v in water_years[0].items(): 
		#print(v)
		print(v.iloc[:-1,:-1])
		df = v.iloc[:-1,:-1]
		df['average'] = df.mean(axis=1)
		df['anomaly'] = df['average']-water_years[1] #get the full time series mean for the season of interest
		# yr_min=int(v.index.min())
		# yr_max=int(v.index.max())+1
		# year_list = range(yr_min,yr_max,1)
		#get start and end year
		# try:  
		fig,(ax,ax1) = plt.subplots(2,1, figsize=(5,5))
		#plt.figure()
		# 	years = [int(i) for i in v.columns if i.isnumeric()]
		# 	print(years)
		# except:
		# 	print('that one is not a number')
		# 	continue
		# fig,ax = plt.subplots(figsize=(5,5))
		# v[v.columns[len(years):]].plot.line(ax=ax,legend=False)
		#plt.plot(df['anomaly'])


		palette = sns.light_palette('Navy', len(df.T.columns.unique()))
	# for i in range(rows*cols):
	# 	try: 
	# 		count = 0 
	# 		df_slice = df[df['climate_region']==region_list[i]].sort_values('year')
	# 		for j in year_list: 
		# 		df_slice[df_slice['year']==j].sort_values('low_bound').plot.line(x='low_bound',y=variable,ax=axes[i],legend=False,color=list(palette)[count]) #variable denotes first or last day. Can be first_day_mean or last_day_mean
		# 		count +=1

		# 	axes[i].set_title(string.capwords(str(region_list[i]).replace('_',' ')))
		# 	axes[i].xaxis.label.set_visible(False)
			
		# except IndexError: 
		# 	continue
		year_list = sorted(df.T.columns.unique())

		norm = mpl.colors.Normalize(vmin=min(year_list),vmax=max(year_list))
		cmap = sns.light_palette('Navy',len(year_list),as_cmap=True)
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
	
	
		df['anomaly'].plot.line(ax=ax,legend=False,color='darkblue',lw=2)
		v.T.plot.line(ax=ax1,legend=False,lw=0.5)#color=list(palette))

		#v['average'].plot.line(ax=ax,color='red',lw=2)#,color=['darkblue','forestgreen','firebrick','black','peru'])
		#v['plus_one_std'].plot.line(ax=ax,color='darkblue',lw=2)#,color=['violet','purple','darkred',])
		#v['minus_one_std'].plot.line(ax=ax,color = 'forestgreen',lw=2)#,color=['violet','purple','darkred',])
		#v[v.columns[:-3]].plot.line(ax=ax,color='violet',legend=False,alpha=0.5) #exclude the last three cols
		#fig.subplots_adjust(bottom=0.1, top=0.9, left=0.0, right=0.7,
         #           wspace=0.5, hspace=0.02)
	
		# put colorbar at desire position
		#cbar_ax = fig.add_axes([0.95, 0.1, .01, .8])
		#add colorbar
		#fig.colorbar(sm,ticks=np.linspace(int(min(year_list)),int(max(year_list))+1,int(len(year_list))+1),boundaries=np.arange(int(min(year_list)),int(max(year_list))+1,1),cax=cbar_ax)

		index_vals=list(df.index.values)
		ax.set_title(f'SNOTEL station {k} {season} months')
		ax.set_ylabel(f'Anomaly from {min(index_vals)} to {max(index_vals)} mean (inches SWE)')
		ax.set_xlabel('Water year')
		ax1.set_xlabel('Day of winter (Dec-Feb)')
		ax1.set_ylabel('Daily total SWE (in)')
		#ax.set_xticks(year_list)
		#ax.set_xticklabels(year_list)
		plt.tight_layout()
		plt.show()
		plt.close('all')
	#print(water_years)
	# t0=time()
	# #clusters[0] are the data in the clusters and clusters[1] are the cluster centroids 
	# #define file names
	# dy_fn = output_filepath+f'dry_year_{num_clusters}_clusters'
	# wy_fn = output_filepath+f'wet_year_{num_clusters}_clusters'

	# if pickle_it == 'true': 
	# 	if not os.path.exists(dy_fn): 
	# 		dry_year_clusters=kmeans.kmeans(transform_data(water_years,dryer_year,season),num_clusters,dryer_year,30)
	# 		wet_year_clusters=kmeans.kmeans(transform_data(water_years,wetter_year,season),num_clusters,wetter_year,30)	
	# 		pickle_dry_year = pickle.dump(dry_year_clusters, open(dy_fn, 'ab' ))
	# 		pickle_wet_year = pickle.dump(wet_year_clusters, open(wy_fn, 'ab' ))
	# 	else: 
	# 		print('the cluster file already exists')
	# else: 
	# 	print('that version has already been pickled')
	# #depreceated at the moment- used to get all of the stations in one go for clustering 
	# #cluster_data = kmeans.prepare_cluster_outputs(clusters[0])
	# t1=time()
	# print('time to run was ', t1-t0, 'seconds')
	
	# ######################
	# #try the new yearly classification
	# if pickle_it == 'true':
	# 	#pickle_opener(None,None,filepath,filename_in) 
	# 	if not os.path.exists(dy_fn+'_distances_updated'): 
	# 		dry_year_distances = combine.cluster_parallel(pickle.load(open(dy_fn,'rb'))[1],water_years,output_filepath,30)
	# 		wet_year_distances = combine.cluster_parallel(pickle.load(open(wy_fn,'rb'))[1],water_years,output_filepath,30)#combine.cluster_parallel(wet_year_clusters[1],None,water_years,None,None,None,30)
	# 		pickle.dump(dry_year_distances, open(dy_fn+'_distances_updated', 'ab' ))
	# 		pickle.dump(wet_year_distances, open(wy_fn+'_distances_updated', 'ab' ))
	# 	else: 
	# 		print('the distance file already exists')
	# else: 
	# 	print('pickle it is false so reading from pickled version')
	# #print(distances)
	# if read_from_pickle == 'true': 
	# 	combine.plot_distances(pickle.load(open(dy_fn+'_distances_updated','rb')),pickle.load(open(wy_fn+'_distances_updated','rb')),None,int(start_date[:4]),int(end_date[:4]),output_filepath,num_clusters)
	# elif os.path.exists(dy_fn+'_distances_updated') != True: 
	# 	print('the file for the distance plots does not exist')
	# else: 
	# 	print('something else went wrong with the distance plots, please try again')
	#kmeans_output=kmeans.build_mapping_inputs(cluster_data,sites_full)
	#print(kmeans_output)

	#run to make visualization maps
	#for i in range(num_clusters): 
		#print(f'cluster is: {i+1}',kmeans_output[f'cluster_{i+1}'])
	#	combine.grid_mapping(path/'k_means'/'temp_files'/'oregon_state_grid_snotel_joined.shp',kmeans_output[f'cluster_{i+1}'],clusters[1][f'cluster_{i+1}'],i+1)
	#visualize!
	#print(sites_full.head())


	#grid=combine.create_fishnet_vectors(state_shapefile,output_filepath,cell_size)
	#visualize=combine.point_mapping(station_list,station_csv,version,state,label,site_label,par_dir,colors,color_dict,diffs,from_year,to_year)

if __name__ == '__main__':
    main()

