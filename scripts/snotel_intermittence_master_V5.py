#author broberts

#import modules and functions from the other intermittence function script
import os
import snotel_intermittence_functions as combine
import sys
import json
import pickle
import pandas as pd
import glob
import pyParz

def run_model(args):
	station_id,param_dict,start_date,end_date,sentinel_dict = args 
	station_ls = []
	for k,v in param_dict.items(): #go through the possible snotel params  
			station_ls.append(v[station_id])
	station_df = pd.concat(station_ls,axis=1) #make a new df that has the years of interest and the snotel params 
	for year in range(int(start_date[0:4])+1,int(end_date[0:4])+1): #run analysis annually- this will likely need to be changed
		#select the params for that year 
		try: 
			snotel_year = station_df.loc[:, station_df.columns.str.contains(str(year))]
			analysis_df = combine.PrepPlottingData(station_df,None,int(station_id),sentinel_dict[str(year)]).make_plot_dfs(str(year))
			#print(analysis_df.head())
		except KeyError: 
			print('That file may not exist')
			continue 

		mlr = combine.LinearRegression(analysis_df).multiple_lin_reg()
		print(mlr)
def main():
	"""Master function for snotel intermittence from SNOTEL and RS data. 
	Requirements: 
	snotel_intermittence_functions.py - this is where most of the actual work takes place, this script just relies on code in that script 
	snotel_intermittence_master.txt - this is the param file for running all of the snotel/rs functions outlined in the functions script
	run this in a python 3 conda environment and make sure that the climata package is installed

	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		state_shapefile = variables["state_shapefile"]
		pnw_shapefile = variables["pnw_shapefile"]
		epsg = variables["epsg"]
		output_filepath=variables["output_filepath"]
		season = variables["season"]
		csv_dir = variables["csv_dir"]
		stations = variables["stations"]
		parameter = variables["parameter"]
		start_date = variables["start_date"]
		end_date = variables["end_date"]
		write_out = variables["write_out"]
		pickles = variables["pickles"]
	#run_prep_training = sys.argv[1].lower() == 'true' 
	#get some of the params needed- TO DO- the source csv needs to be updated with the precise snotel site locations 
	sites = combine.make_site_list(stations)
	sites_full = sites[0] #this is getting the full df of oregon (right now) snotel sites
	sites_ids = sites[1] #this is getting a list of just the snotel site ids. You can grab a list slice to restrict how big results gets below. 
	new_parameter = parameter+'_scaled'
	state = sites_full['state'][0] #this is just looking into the df created from the sites and grabbing the state id. This is used to query the specific station
	
	#create the input data 
	input_data = combine.CollectData(stations,parameter,start_date,end_date,state,sites_ids, write_out,output_filepath)
	if write_out.lower() == 'true': 
		#get new data- this requires that the user specify the param
		pickle_results=input_data.snotel_compiler() #this generates a list of dataframes of all of the snotel stations that have data for a given state
		results=input_data.pickle_opener()
		# return results
	else: 
		#use pickled data 
		try: 
			# #params currently considered, either update here or include as a param 
			# param_list = ['WTEQ','PREC','TAVG','SNWD']
			# param_dict = {}
			#read the dfs of each snotel param into a dict for further processing 
			wteq_wy = combine.StationDataCleaning(input_data.pickle_opener(
				output_filepath+f'{state}_WTEQ_{start_date}_{end_date}_snotel_data_list'),
				'WTEQ',new_parameter,start_date,end_date,season)
			prec_wy = combine.StationDataCleaning(input_data.pickle_opener(
				output_filepath+f'{state}_PREC_{start_date}_{end_date}_snotel_data_list'),
				'PREC',new_parameter,start_date,end_date,season)
			tavg_wy = combine.StationDataCleaning(input_data.pickle_opener(
				output_filepath+f'{state}_TAVG_{start_date}_{end_date}_snotel_data_list'),
				'TAVG',new_parameter,start_date,end_date,season)
			snwd_wy = combine.StationDataCleaning(input_data.pickle_opener(
				output_filepath+f'{state}_SNWD_{start_date}_{end_date}_snotel_data_list'),
				'SNWD',new_parameter,start_date,end_date,season)
			
		except: 
			raise FileNotFound('Something is wrong with the pickled file you requested, double check that file exists and run again')
	#make outputs 
	#water_years=combine.StationDataCleaning(results,parameter,new_parameter,start_date,end_date,season)
	#generate a dictionary with each value being a dict of site_id:df of years of param
	param_dict = {'wteq':wteq_wy.prepare_data('WTEQ'),'prec':prec_wy.prepare_data('PREC'),
	'tavg':tavg_wy.prepare_data('TAVG'),'snwd':snwd_wy.prepare_data('SNWD')}
	station_dict = {}
	#pickle_files = [i for i in glob.glob(pickles+'*2014*')] #just pick something that all of the subset file names have 
	sentinel_dict = {}
	#read in the sentinel csvs and get the water year
	for file in glob.glob(csv_dir+'*.csv'): 
		sentinel_data = combine.PrepPlottingData(None,file,None,None).csv_to_df()
		sentinel_dict.update({str(sentinel_data[1]):sentinel_data[0]})
	#go through the station ids and do the regression
	# 	ef clustering(args):
	#     """Compare station time series against reference dataset and determine time series type. 
	#     """
	#     key,x,y = args #x is the station dataframe and y is the dictionary of training dataframes
	#     return [key,x[key].apply(lambda column: faster(y,column),axis=0).reset_index()]

	# def cluster_parallel(train_dict,classify_data,filepath,njobs): 
	#     """Paralellizes the clustering function above."""
	#     #get data from pickled dictionary of dataframes
	#     # if from_pickle: 
	#     #     df_dict = pickle_opener(1,None,filepath,filename_in)
	#     # else: 
	#     df_dict = classify_data
	#     print('working...')
	#     #run wrapper pyParz for multiprocessing pool function 
	#     results_classes=pyParz.foreach(list(df_dict.keys()),clustering,args=[df_dict,train_dict],numThreads=njobs)
	output = pyParz.foreach(sites_ids,run_model,args=[param_dict,start_date,end_date,sentinel_dict],numThreads=20)
	# for station_id in sites_ids[:2]: #make parallel
	# 	station_ls = []
	# 	for k,v in param_dict.items(): #go through the possible snotel params  
	# 		station_ls.append(v[station_id])
	# 	station_df = pd.concat(station_ls,axis=1) #make a new df that has the years of interest and the snotel params 
	# 	for year in range(int(start_date[0:4])+1,int(end_date[0:4])+1): #run analysis annually- this will likely need to be changed
	# 		#select the params for that year 
	# 		try: 
	# 			snotel_year = station_df.loc[:, station_df.columns.str.contains(str(year))]
	# 			analysis_df = combine.PrepPlottingData(station_df,None,int(station_id),sentinel_dict[str(year)]).make_plot_dfs(str(year))
	# 			#print(analysis_df.head())
	# 		except KeyError: 
	# 			print('That file may not exist')
	# 			continue 

	# 		mlr = combine.LinearRegression(analysis_df).multiple_lin_reg()
			#print(mlr)
		#station_dict.update({station_id:station_df}) #put new combined df into a dictionary format like station id: df of all params combined
	#print(station_dict)
    # for station_id in sites_ids[:2]: 

   	# 	for year in range(int(start_date[0:4])+1,int(end_date[0:4])+1):
   		
   	# 		for k,v in station_dict.items(): 

   	# 			combine.PrepPlottingData(station_dict,input_csv) 
		#print(station_df)
	#viz = combine.PrepPlottingData(water_years.prepare_data(),input_csv) #get the dictionary of water years not the df of all the years which would be index 0
	# # #print(viz.clean_gee_data())
	# print(viz.simple_lin_reg_plot())

	
if __name__ == '__main__':
    main()

# wteq_results=input_data.pickle_opener(output_filepath+f'{state}_WTEQ_{start_date}_{end_date}_snotel_data_list')
			# prec_results=input_data.pickle_opener(output_filepath+f'{state}_PREC_{start_date}_{end_date}_snotel_data_list')
			# tavg_results=input_data.pickle_opener(output_filepath+f'{state}_TAVG_{start_date}_{end_date}_snotel_data_list')
			# snwd_results=input_data.pickle_opener(output_filepath+f'{state}_SNWD_{start_date}_{end_date}_snotel_data_list')