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
import matplotlib.pyplot as plt

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
def combine_dfs(sites_ids,param_dict,start_date,end_date): 
	#station_id,param_dict,start_date,end_date = args
	dry = {}
	warm = {}
	warm_dry = {}
	for station_id in sites_ids: 
		print(f'the station id is: {station_id}')
		combined_df = pd.concat([param_dict['wteq'][station_id],param_dict['prec'][station_id],param_dict['tavg'][station_id]], axis=0).T #right now the col headers are the index because we were calculating anomalies, transpose that
		small_df = combined_df.tail(1)
		
		# dry = small_df.filter(like='WTEQ').columns
		# dry = dry[dry<=0]
		# print(small_df)
		
		for year in range(int(start_date[0:4])+1,int(end_date[0:4])+1):
			#print(f'year is {year}')
			yearly = small_df.filter(like=str(year))
			#print(yearly)
			try: 
				if (yearly[f'WTEQ_{year}'][0] < 0) and (yearly[f'PREC_{year}'][0] < 0) and (yearly[f'TAVG_{year}'][0] < 0): 
					dry.update({station_id:year})
				elif (yearly[f'WTEQ_{year}'][0] < 0) and (yearly[f'PREC_{year}'][0] > 0): 
					warm.update({station_id:year})
				elif (yearly[f'WTEQ_{year}'][0] < 0) and (yearly[f'PREC_{year}'][0]) < 0 and (yearly[f'TAVG_{year}'][0]) > 0: 
					warm_dry.update({station_id:year})
				else: 
					print(f'station {station_id} for {year} was normal or above average. The swe anom was: {yearly[f"WTEQ_{year}"][0]}')
			except KeyError:
				continue 

	#print(dry,warm,warm_dry)
	# 	small_df=
	#inter_list = combined_df.index[df['BoolCol'] == True].tolist()
	#df.filter(like='spike').columns
	output = {'dry':dry,'warm':warm,'warm_dry':warm_dry}
	return output

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
	if write_out.lower() == 'true': 
		for param in ['WTEQ','PREC','TAVG','SNWD']: 
			print('current param is: ', param)
			input_data = combine.CollectData(stations,param,start_date,end_date,state,sites_ids, write_out,output_filepath)
			#get new data
			pickle_results=input_data.snotel_compiler() #this generates a list of dataframes of all of the snotel stations that have data for a given state
		#results=input_data.pickle_opener()
		# return results
	#use pickled data 
	try: 
		# #params currently considered, either update here or include as a param 
		# param_list = ['WTEQ','PREC','TAVG','SNWD']
		# param_dict = {}

		#read the dfs of each snotel param into a dict for further processing 
		wteq_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_WTEQ_{start_date}_{end_date}_snotel_data_list'),
			'WTEQ',season)
		prec_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_PREC_{start_date}_{end_date}_snotel_data_list'),
			'PREC',season)
		tavg_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_TAVG_{start_date}_{end_date}_snotel_data_list'),
			'TAVG',season)
		snwd_wy = combine.StationDataCleaning(combine.pickle_opener(
			output_filepath+f'{state}_SNWD_{start_date}_{end_date}_snotel_data_list'),
			'SNWD',season)
		
	except FileNotFoundError: 
		raise FileNotFoundError('Something is wrong with the pickled file you requested, double check that file exists and run again')
	#make outputs 
	#water_years=combine.StationDataCleaning(results,parameter,new_parameter,start_date,end_date,season)
	#generate a dictionary with each value being a dict of site_id:df of years of param
	param_dict = {'wteq':wteq_wy.prepare_data('true',start_date,end_date),'prec':prec_wy.prepare_data('true',start_date,end_date),
	'tavg':tavg_wy.prepare_data('true',start_date,end_date),'snwd':snwd_wy.prepare_data('true',start_date,end_date)}
	output=combine_dfs(sites_ids,param_dict,start_date,end_date)
	print(output)
	plt.bar(output['dry'].values(), list(output['dry'].keys()), color='g')
	plt.show()
	#print(param_dict['wteq'])
	#print(param_dict)
	#output = pyParz.foreach(sites_ids,combine_dfs,args=[param_dict,start_date,end_date],numThreads=25) 
	#print(output[1])
		#combined_df = pd.concat([param_dict['wteq'][station],param_dict['prec'][station],param_dict['tavg'][station]], axis=0)
		# wteq = 
		# prec = param_dict['prec'][station]
		# tavg = param_dict['tavg'][station]

	#uncomment to run regressions 
	# station_dict = {}
	# #pickle_files = [i for i in glob.glob(pickles+'*2014*')] #just pick something that all of the subset file names have 
	# sentinel_dict = {}
	# #read in the sentinel csvs and get the water year
	# for file in glob.glob(csv_dir+'*.csv'): 
	# 	sentinel_data = combine.PrepPlottingData(None,file,None,None).csv_to_df()
	# 	sentinel_dict.update({str(sentinel_data[1]):sentinel_data[0]})

	# output = pyParz.foreach(sites_ids,run_model,args=[param_dict,start_date,end_date,sentinel_dict],numThreads=20)
	

	
if __name__ == '__main__':
    main()

# wteq_results=input_data.pickle_opener(output_filepath+f'{state}_WTEQ_{start_date}_{end_date}_snotel_data_list')
			# prec_results=input_data.pickle_opener(output_filepath+f'{state}_PREC_{start_date}_{end_date}_snotel_data_list')
			# tavg_results=input_data.pickle_opener(output_filepath+f'{state}_TAVG_{start_date}_{end_date}_snotel_data_list')
			# snwd_results=input_data.pickle_opener(output_filepath+f'{state}_SNWD_{start_date}_{end_date}_snotel_data_list')