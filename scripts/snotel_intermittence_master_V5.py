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



def run_model(huc_id,snotel_param_dict,start_date,end_date,sentinel_dict):
	#station_id,param_dict,start_date,end_date,sentinel_dict = args 
	station_ls = []
	for k,v in snotel_param_dict.items(): #go through the possible snotel params  
		station_ls.append(v[huc_id])
		#station_ls.append(v[station_id]) #dict like {'station id:df of years for a param'}
	station_df = pd.concat(station_ls,axis=1) #make a new df that has the years of interest and the snotel params 
	#print(station_df)
	#station_df = station_df.fillna(0)
	for year in range(int(start_date[0:4])+1,int(end_date[0:4])+1): #run analysis annually- this will likely need to be changed
		#select the params for that year 
		try: 
			#print(station_df)
			snotel_year = station_df.loc[:, station_df.columns.str.contains(str(year))]
			#print(snotel_year)
			analysis_df = combine.PrepPlottingData(station_df,None,None,sentinel_dict[str(year)][huc_id]).make_plot_dfs(str(year))
			print(analysis_df)
			for column in analysis_df.columns: 
				if str(year) in column:
					param = column.split('_') #cols are formatted like param_year or stat_param
					analysis_df[f'plus_std_{param[0]}'] = analysis_df[column]+analysis_df[column].std(axis=0)
					analysis_df[f'minus_std_{param[0]}'] = analysis_df[column]-analysis_df[column].std(axis=0)
				elif 'filter' in column: 
					analysis_df[f'plus_std_filter'] = analysis_df[column]+analysis_df[column].std(axis=0)
					analysis_df[f'minus_std_filter'] = analysis_df[column]-analysis_df[column].std(axis=0)
				else: 
					continue
				print(analysis_df)
		except KeyError: 
			print('That file may not exist')
			continue 
		#modify the df with anomalies 

		param_list = ['WTEQ','PRCP','TAVG','SNWD'] #this is what dictates the plots created in the next line. Can be more automated. 
		
		vis = combine.LinearRegression(analysis_df,param_list).vis_relationship(year,huc_id)
		#mlr = combine.LinearRegression(analysis_df).multiple_lin_reg()
		#print(mlr)
def get_years(input_df): 
	years = input_df.columns[:-1] #drop one for the stats column
	years = [int(i.split('_')[1]) for i in years]
	years_min = min(years)
	years_max = max(years)
	return years_min,years_max

def combine_dfs(sites_ids,param_dict,start_date,end_date): 
	#station_id,param_dict,start_date,end_date = args
	#here the param_dict is like {param:{station_id:df for param}}
	dry = {}
	warm = {}
	warm_dry = {}
	for station_id in sites_ids: 

		try:
			#print(f'the station id is: {station_id}')
			#combined_df = pd.concat([param_dict['wteq'][station_id],param_dict['prcp'][station_id],param_dict['tavg'][station_id]], axis=0) #combine params for use in the next step
			wteq_df = param_dict['wteq'][station_id]
			prcp_df = param_dict['prcp'][station_id]
			tavg_df = param_dict['tavg'][station_id]
			print(wteq_df)
			print(prcp_df)
			print(tavg_df)
			start_year = max([get_years(wteq_df)[0],get_years(prcp_df)[0],get_years(tavg_df)[0]]) #get the min year from each of these dataframes and then get the most recent one so there is data for all years
			end_year = min([get_years(wteq_df)[1],get_years(prcp_df)[1],get_years(tavg_df)[1]]) #get the max year from each of the dataframes and then take the min so you get the earliest year that has data for all years
			#print('combined df is: ', combined_df)
			# print('tav_df is ', tavg_df)
			# print(f'tav_df shape is {tavg_df.shape}')
			# test = tavg_df['TAVG_2005'][tavg_df[f'TAVG_2005']>0]
			# print(test)
			# print(f'test.shape is: {test.shape}')
			# print(test.shape[0])
		except KeyError: 
			print('that station is missing')
			continue
		#small_df = combined_df.tail(1)
		#max_values = pd.DataFrame(combined_df.max()).T
		for year in range(start_year,end_year+1):#range(int(start_date[0:4])+1,int(end_date[0:4])+1): the first year will actually be the start of the water year, not the water year itself 
			print(f'the year is {year}')
			try: 
				if (wteq_df[f'WTEQ_{year}'].max() < wteq_df[f'stat_WTEQ'][0]) and (prcp_df[f'PREC_{year}'].max()<prcp_df['stat_PREC'][0]) and ((tavg_df[f'TAVG_{year}'][tavg_df[f'TAVG_{year}']>0].shape[0])<tavg_df[f'stat_TAVG'][0]): #(tavg_df[f'TAVG_{year}'].mean()<tavg_df[f'stat_TAVG'][0]):
					dry.update({station_id:year})
				elif (wteq_df[f'WTEQ_{year}'].max() < wteq_df[f'stat_WTEQ'][0]) and (prcp_df[f'PREC_{year}'].max()>=prcp_df['stat_PREC'][0]): 
					warm.update({station_id:year})
				elif (wteq_df[f'WTEQ_{year}'].max() < wteq_df[f'stat_WTEQ'][0]) and (prcp_df[f'PREC_{year}'].max()<prcp_df['stat_PREC'][0]) and ((tavg_df[f'TAVG_{year}'][tavg_df[f'TAVG_{year}']>0].shape[0])>tavg_df[f'stat_TAVG'][0]): 
					warm_dry.update({station_id:year})
				else: 
					print(f'station {station_id} for {year} was normal or above average. The swe value was: {wteq_df[f"WTEQ_{year}"].max()} and the long term mean max value was: {wteq_df[f"stat_WTEQ"][0]}')
			#print(f'year is {year}')
			#yearly = max_values.filter(like=str(year))
			#l = [str(year),'stat']
			#regstr = '|'.join(l)
			#yearly = max_values.loc[:,max_values.columns.str.contains(regstr)]
			#print(yearly)
			# try: #in these if/else statements the 'stat' col is the long term max. I've taken the max values from there in line with Dierauer et. al 2019
			# 	if (yearly[f'WTEQ_{year}'][0] < yearly['stat_WTEQ'][0]) and (yearly[f'PRCP_{year}'][0] < yearly['stat_PRCP'][0]) and (yearly[f'TAVG_{year}'][0] < yearly['stat_TAVG'][0]): 
			# 		dry.update({station_id:year})
			# 	elif (yearly[f'WTEQ_{year}'][0] < yearly['stat_WTEQ'][0]) and (yearly[f'PRCP_{year}'][0] > yearly['stat_PRCP'][0]): 
			# 		warm.update({station_id:year})
			# 	elif (yearly[f'WTEQ_{year}'][0] < yearly['stat_WTEQ'][0]) and (yearly[f'PRCP_{year}'][0] < yearly['stat_PRCP'][0]) and (yearly[f'TAVG_{year}'][0] > yearly['stat_TAVG'][0]) : 
			# 		warm_dry.update({station_id:year})
			# 	else: 
			# 		print(f'station {station_id} for {year} was normal or above average. The swe value was: {yearly[f"WTEQ_{year}"][0]} and the long term mean was: {yearly[f"stat_WTEQ"][0]}')
			except Exception as e:
				print(f'error is: {e}')
				#print(f'The year {year} does not exist in the df whose first year is: {wteq_df.columns[0]}')
				
				continue	
	#print(dry,warm,warm_dry)
	# 	small_df=
	#inter_list = combined_df.index[df['BoolCol'] == True].tolist()
	#df.filter(like='spike').columns
	output = {'dry':dry,'warm':warm,'warm_dry':warm_dry}
	print(output)
	return output

def plot_anoms(input_dict,anom_start_date,anom_end_date): 

	width = 1.0     # gives histogram aspect to the bar diagram
	plot_dict = {}
	for k,v in input_dict.items(): 
		count_dict = {}
		vals = list(v.values())
		for i in set(vals): #get the unique years
			counts = vals.count(i) #get a count of each year
			count_dict.update({i:counts})
		plot_dict.update({k:count_dict})
	fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
	ax = ax.flatten()
	count = 0
	for k,v in plot_dict.items(): 
		ax[count].bar(list(v.keys()), v.values(), color='g')
		#ax[count].set_xticks(range(int(anom_start_date[0:4])+1,int(anom_end_date[0:4])+1))
		ax[count].set_title(f'{k} snow drought')
		count +=1 
	# plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()
	plt.close('all')
		
	return plot_dict

def format_dict(input_dict,modifier): 
		return {f'{modifier}_'+k:v for k,v in input_dict.items()}

def organize_plots(input_dir,season): 
	fig,ax = plt.subplots(3,3,figsize=(10,10),sharex=True,sharey=True)
	#ax = ax.flatten()
	files = glob.glob(input_dir+'*.csv')
	for i in range(3): 
		df = pd.read_csv(files[i]) 
		for j in range(3): 
			ax[i][j].bar(df['year'],df.iloc[:,j+1],color='darkblue') #iterate through the df columns 
			ax[i][j].set_title(os.path.split(files[i])[1][:-4].split('_')[0]+f' {" ".join(season.split("_"))} '+df.columns[j+1])
			ax[i][j].set_ylabel('Station count')
	plt.show()
	plt.close('all')



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
		anom_start_date = variables["anom_start_date"]
		anom_end_date = variables["anom_end_date"]
		write_out = variables["write_out"]
		pickles = variables["pickles"]
		anom_bool = variables["anom_bool"]
		state_abbr = variables['state_abbr']
	#run_prep_training = sys.argv[1].lower() == 'true' 
	#get some of the params needed
	stations_df = pd.read_csv(stations)
	stations_df = stations_df[stations_df['state']==state_abbr] #this might not be the best way to run this, right now it will require changing the state you want 
	sites = combine.make_site_list(stations_df)
	sites_full = sites[0] #this is getting the full df of oregon (right now) snotel sites
	sites_ids = sites[1] #this is getting a list of just the snotel site ids. You can grab a list slice to restrict how big results gets below. 
	huc_dict = sites[2] #this gets a dict of format {site_id:huc12 id}
	new_parameter = parameter+'_scaled'
	state = sites_full['state'].iloc[0] #this is just looking into the df created from the sites and grabbing the state id. This is used to query the specific station
	#print('huc_dict is: ', huc_dict)
	#make a csv for sentinel data
	#sites_full.to_csv(f"/vol/v1/general_files/user_files/ben/excel_files/{state}_snotel_data.csv")
	huc_list = list(set(i for i in huc_dict.values())) 
	
	#create the input data
	if write_out.lower() == 'true': 
		for param in ['WTEQ','PREC','TAVG','SNWD','PRCP','PRCPSA']: #PRCP
			
			print('current param is: ', param)
			input_data = combine.CollectData(stations,param,anom_start_date,anom_end_date,state,sites_ids, write_out,output_filepath)
			#get new data
			pickle_results=input_data.snotel_compiler() #this generates a list of dataframes of all of the snotel stations that have data for a given state
		
	try: 
		# #params currently considered, either update here or include as a param 
		# param_list = ['WTEQ','PREC','TAVG','SNWD']
		# param_dict = {}

		#read the dfs of each snotel param into a dict for further processing 
		pass
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
	#make outputs 
	#water_years=combine.StationDataCleaning(results,parameter,new_parameter,start_date,end_date,season)
	#generate a dictionary with each value being a dict of site_id:df of years of param
	#organize_plots('/vol/v1/general_files/user_files/ben/excel_files/snow_drought_outputs/',season)

	#format snotel data by paramater
	param_dict = {'wteq':wteq_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state),'prcp':prec_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state),
	'tavg':tavg_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state),'snwd':snwd_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state)}
	# #test={'tavg':tavg_wy.prepare_data(anom_bool,anom_start_date,anom_end_date,state)}
	# #print(test)
	# #print(param_dict['tavg'])
	# ##########################################################################
	#this is working and generates the snow drought years
	#generate anomolies 
	snow_droughts=combine_dfs(sites_ids,param_dict,anom_start_date,anom_end_date)
	#uncomment to make plot of anomaly years 

	year_counts=plot_anoms(snow_droughts,anom_start_date,anom_end_date)
	#print(anomolies)
	#collect some data on these results
	#anom_dict = {'num_stations':len(sites_ids),'count_of_years':year_counts}
	#print('anom dict is: ', anom_dict)
	anom_df = pd.DataFrame.from_dict(year_counts)
	#if season.lower() == ''
	anom_df.to_csv(f'/vol/v1/general_files/user_files/ben/excel_files/{state}_anom_outputs_{season}_degree_days.csv')
	#########################################################################
	#this might be working and should be used to generate the multiple linear regressions
	#print(param_dict['wteq'])
	# if anom_bool == 'true': 
	# 	#uncomment to run regressions 
	# 	station_dict = {}
	# 	#pickle_files = [i for i in glob.glob(pickles+'*2014*')] #just pick something that all of the subset file names have 
	
	################################################################################
	#below here working and the correct sequence to generate figures of snotel params against sentinel 1 by year and huc basin
	
	# sentinel_dict = {}
	# # 	#read in the sentinel csvs and get the water year
	# for file in glob.glob(csv_dir+'*.csv'): 
	# 	sentinel_data = combine.PrepPlottingData(None,file,None,None).csv_to_df()
	# 	sentinel_dict.update({str(sentinel_data[1]):sentinel_data[0]})
	# #print(sentinel_dict)
	# #testing old method for making figures
	# # for i in sites_ids: #use to make sentinel/snotel figures 
	# # 	run_model(i,param_dict,start_date,end_date,sentinel_dict)
	
	# #reformat snotel data so that it is organized by huc code 
	# #######################################################################################
	# snotel_output_dict = {} #should look like {param:{huc_id:[list of dfs]}}
	# for k,v in param_dict.items(): #k is param and v is dict of {'station id':df}
	# 	huc_out = {i:list() for i in huc_list} #use to get the station id and huc code 
	# 	for k1,v1 in huc_dict.items(): 
	# 		inter_df = v[k1]
	# 		for k2,v2 in huc_out.items():
				
	# 			if v1 == k2: 
	# 				huc_out[k2].append(inter_df)
	# 			else: 
	# 				pass
	# 				#print('that is not the right id')
	# 	snotel_output_dict.update({k:huc_out})
	
	# #reformat to concat lists of df to one df
	# for k,v in snotel_output_dict.items(): 
	# 	for k1,v1 in v.items(): #k1 is huc id and v1 is a list of dataframes for that param 
	# 		output_df = pd.concat(v1).groupby(level=0).mean() #concats correctly but then the mean isn't workingdf_concat.groupby(level=0).mean()
	# 		snotel_output_dict[k][k1] = output_df #overwrite the list of dfs with df. Now in the form {param:{huc_id:df of years avg for the basin}}
	# #######################################################################################
	# #reformat the sentinel data
	# sentinel_dict_out = {}
	# for k,v in sentinel_dict.items(): #this is like {year:df of sentinel data} 
	# 	#print('v is : ',v)
	# 	huc_out = {i:list() for i in huc_list} #use to get the station id and huc code 

	# 	for i in sites_ids: 
	# 		try: 
	# 			inter_df = combine.PrepPlottingData(None,None,int(i),v).clean_gee_data()#.make_plot_dfs(str(year))
	# 			huc_id = huc_dict[i]
	# 			huc_out[huc_id].append(inter_df)
	# 			#sentinel_dict_out.update({i:inter_df})
	# 		except:
	# 			print(f'id {i} for year {k} does not appear to exist')
	# 			continue
	# 	sentinel_dict_out.update({k:huc_out})
	# #print(sentinel_dict_out)
	# for k,v in sentinel_dict_out.items(): 
	# 	for k1,v1 in v.items(): 
	# 		# for df in v1: 
	# 		# 	df = df.set_index('week_of_year')
	# 		#In [4]: df.groupby('StationID', as_index=False)['BiasTemp'].mean()
	# 		#combine the list of dfs and get the mean of filter by matching weeks of year (they are not all the same size df)
	# 		output_df = pd.concat(v1).groupby('week_of_year',as_index=False)['filter'].mean()
	# 		sentinel_dict_out[k][k1] = output_df
	# #print(sentinel_dict_out)
	# #make some figures
	# for i in huc_list: 
	# 	run_model(i,snotel_output_dict,start_date,end_date,sentinel_dict_out)
	
	######################################################################
	# station_ls = []
	# for k,v in snotel_param_dict.items(): #go through the possible snotel params  
	# 	for k1,v1 in v.items(): #{huc_id:df of years for that param}
	# 		station_ls.append(v[station_id]) #dict like {'station id:df of years for a param'}
	# station_df = pd.concat(station_ls,axis=1) #make a new df that has the years of interest and the snotel params 
	# print(station_df)

	# for k,v in sentinel_dict_out.items(): #use to make sentinel/snotel figures k is years and v is dict like: {huc_id:df of basin avg for that year}
	# 	inter_dict = {}
	# 	inter_list = []
	# 	for k1,v1 in v.items(): 
	# 		for i in ['wteq','prec','tavg','snwd']: 
	# 			snotel_output_dict[i]

		# for k1,v1 in v.items(): #k1 is huc id and v1 is df of avg for that year 
		# 	for k2,v2 in snotel_output_dict.items(): 
		# 		input_data = PrepPlottingData(snotel_output_dict,None,None,v) #here we just pass v as the gee_data 
		# 		run_model(snotel_output_dict,start_date,end_date,v)

	
	# output = pyParz.foreach(sites_ids,run_model,args=[param_dict,start_date,end_date,sentinel_dict],numThreads=20)

	
if __name__ == '__main__':
    main()

#currently working
# def run_model(station_id,param_dict,start_date,end_date,sentinel_dict):
# 	#station_id,param_dict,start_date,end_date,sentinel_dict = args 
# 	station_ls = []
# 	for k,v in param_dict.items(): #go through the possible snotel params  
# 		station_ls.append(v[station_id]) #dict like {'station id:df of years for a param'}
# 	station_df = pd.concat(station_ls,axis=1) #make a new df that has the years of interest and the snotel params 
# 	print(station_df)
# 	#station_df = station_df.fillna(0)
# 	for year in range(int(start_date[0:4])+1,int(end_date[0:4])+1): #run analysis annually- this will likely need to be changed
# 		#select the params for that year 
# 		try: 
# 			#print(station_df)
# 			snotel_year = station_df.loc[:, station_df.columns.str.contains(str(year))]
# 			print(snotel_year)
# 			analysis_df = combine.PrepPlottingData(station_df,None,int(station_id),sentinel_dict[str(year)]).make_plot_dfs(str(year))
# 			#print(analysis_df)
# 		except KeyError: 
# 			print('That file may not exist')
# 			continue 
# 		#modify the df with anomalies 

# 		param_list = ['WTEQ','PREC','TAVG','SNWD'] #this is what dictates the plots created in the next line. Can be more automated. 
# 		#vis = combine.LinearRegression(analysis_df,param_list).vis_relationship(year,station_id)
# 		#mlr = combine.LinearRegression(analysis_df).multiple_lin_reg()
# 		#print(mlr)

