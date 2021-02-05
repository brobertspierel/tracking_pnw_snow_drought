import os
import snotel_functions as combine
import sys
import json
import pickle
import pandas as pd
import glob
import geopandas as gpd
import pyParz
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors



def create_basin_scale_datasets(state,output_filepath,anom_start_date,anom_end_date,season,anom_bool):
	"""Helper function to take the pickled files that store all the state level snotel data and then make it into something useable for next steps."""
	try: 
		# #params currently considered, either update here or include as a param 
		# param_list = ['WTEQ','PREC','TAVG','SNWD']
		# param_dict = {}

		#read the dfs of each snotel param into a dict for further processing 
		wteq_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_WTEQ_{anom_start_date}_{anom_end_date}_snotel_data_list'),
			'WTEQ',season)
		prec_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_PREC_{anom_start_date}_{anom_end_date}_snotel_data_list'),
			'PREC',season)
		tavg_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_TAVG_{anom_start_date}_{anom_end_date}_snotel_data_list'),
			'TAVG',season)
		snwd_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_SNWD_{anom_start_date}_{anom_end_date}_snotel_data_list'),
			'SNWD',season)
		
	except FileNotFoundError: 
		raise FileNotFoundError('Something is wrong with the pickled file you requested, double check that file exists and run again')
	
	#format snotel data by paramater
	param_dict = {'wteq':wteq_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state),'prec':prec_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state),
	'tavg':tavg_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state),'snwd':snwd_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state)}
	return param_dict

def main():
	"""Master function for snotel intermittence from SNOTEL 
	Requirements: 
	snotel_intermittence_functions.py - this is where most of the actual work takes place, this script just relies on code in that script 
	snotel_intermittence_master.txt - this is the param file for running all of the snotel/rs functions outlined in the functions script
	run this in a python 3 conda environment and make sure that the climata package is installed

	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		season = variables["season"]
		stations = variables["stations"]
		anom_start_date = variables["anom_start_date"]
		anom_end_date = variables["anom_end_date"]
		write_out = variables["write_out"]
		pickles = variables["pickles"]
		anom_bool = variables["anom_bool"]
		time_step = variables['time_step']
		state = variables['state'] 

	#get some of the params needed
	stations_df = pd.read_csv(stations)
	sites = combine.make_site_list(stations_df,'huc_08',8,state)
	sites_full = sites[0] #this is getting the full df of oregon (right now) snotel sites. Changed 10/28/2020 so its just getting all of the site ids
	#sites_ids = sites[1] #this is getting a list of just the snotel site ids. You can grab a list slice to restrict how big results gets below. 
	huc_dict = sites[2] #this gets a dict of format {site_id:huc12 id}
	huc_list = list(set(i for i in huc_dict.values())) 
	#create the input data
	if write_out.lower() == 'true': 
		for param in ['WTEQ','PREC','TAVG','SNWD','PRCP','PRCPSA']: #PRCP
			for st in ['OR','WA','ID']: 
				sites_ids = combine.make_site_list(stations_df,'huc_08',8,st)[1] #get a list of stations for the given state 	
				print('current param is: ', param)
				input_data = combine.CollectData(param,anom_start_date,anom_end_date,st,sites_ids,write_out, pickles) #changed state variable to None 2/2/2021
				#get new data
				pickle_results=input_data.snotel_compiler() #this generates a list of dataframes of all of the snotel stations that have data for a given state
	
	##########################################################################
	filename = pickles+f'master_dict_all_states_all_years_all_params_dict_{anom_start_date}_{anom_end_date}_{season}_updated'
	if not os.path.exists(filename): 
		states_list = []
		for i in ['OR','WA','ID']: 
			print('the state is now: ',i)

			state_ds = create_basin_scale_datasets(i,pickles,anom_start_date,anom_end_date,season,'true') #these are in the form {param:{station:df}}
			#print('the dataset looks like: ', state_ds)
			states_list.append(state_ds)
		master_dict = {}
		for j in ['wteq','tavg','prec','snwd']: 
			vals0 = states_list[0][j] #get the param dict
			vals1 = states_list[1][j]
			vals2 = states_list[2][j]
			new_dict = {**vals0,**vals1,**vals2} #concat the param dicts
			master_dict.update({j:new_dict})
		#master_dict = {**states_list[0],**states_list[1],**states_list[2]}
		pickle_data=pickle.dump(master_dict, open(filename,'ab'))
	
	else: 
		print('That data file already exists, working from pickled version')

	# ##########################################################################
	



if __name__ == '__main__':
    main()