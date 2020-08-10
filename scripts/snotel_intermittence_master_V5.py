#author broberts

#import modules and functions from the other intermittence function script
import os
import snotel_intermittence_functions as combine
import sys
import json
import pickle

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
		input_csv = variables["input_csv"]
		stations = variables["stations"]
		parameter = variables["parameter"]
		start_date = variables["start_date"]
		end_date = variables["end_date"]
		write_out = variables["write_out"]
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
		#get new data
		pickle_results=input_data.snotel_compiler()#combine.snotel_compiler(sites_ids,state,parameter,start_date,end_date,True) #this generates a list of dataframes of all of the snotel stations that have data for a given state
		results=input_data.pickle_opener()#combine.pickle_opener(state,filepath)
		# return results
	else: 
		#use pickled data 
		results=input_data.pickle_opener()#combine.pickle_opener(state,filepath)
	
	print(results)
	
	#make outputs 
	water_years=combine.DataCleaning(results,parameter,new_parameter,start_date,end_date,season)
	# #print(water_years.shape)
	viz = combine.SentinelViz(water_years.prepare_data(),input_csv) #get the dictionary of water years not the df of all the years which would be index 0
	# #print(viz.clean_gee_data())
	print(viz.simple_lin_reg_plot())

	
if __name__ == '__main__':
    main()

