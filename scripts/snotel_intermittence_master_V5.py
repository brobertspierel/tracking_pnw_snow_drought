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
		input_csv = variables["input_csv"]
	#run_prep_training = sys.argv[1].lower() == 'true' 
	
	####################
	#uncomment to run the full archive 
	results = obtain_data(False,1,path,f'{state}_snotel_data_list_1')
	#print([i[parameter].max() for i in results])
	#print(results)
	water_years=combine.DataCleaning(results,parameter,new_parameter,start_date,end_date,season)
	#print(water_years.shape)
	#print(water_years.prepare_data())
	viz = combine.SentinelViz(water_years.prepare_data()[0],input_csv) #get the dictionary of water years not the df of all the years which would be index 0
	#print(viz.clean_gee_data())
	print(viz.simple_lin_reg_plot())

	
if __name__ == '__main__':
    main()

