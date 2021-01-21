# -*- coding: utf-8 -*-


#author broberts

#import modules and functions from the other intermittence function script
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
import numpy as np 
from collections import defaultdict
#import geoplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
import calendar

def get_years(input_df): 
	"""Helper function."""
	years = input_df.columns[:-1] #drop one for the stats column
	years = [int(i.split('_')[1]) for i in years]
	years_min = min(years)
	years_max = max(years)
	return years_min,years_max
def season_window_dates(season): 
	"""Helper function."""

	if season.lower() == 'core_winter': 
		start_month = '12'
		end_month = '02'

	elif season.lower() == 'extended_winter': 
		start_month = '11'
		end_month = '04'

	#select spring months
	elif season.lower() == 'spring': 
		start_month = '03'
		end_month = '05'

	elif season.lower() == 'full_season': 
		start_month = '10'
		end_month = '06'
	return start_month,end_month


def add_dates_to_df(input_df,season,year_of_interest): 
	"""Helper function."""
	#construct start and end dates
	if not season.lower() == 'spring': #all seasons besides spring start on one calendar year and end on the next  
		start_year = year_of_interest-1
	else: 
		pass

	start_date = f'{start_year}-{season_window_dates(season)[0]}-06' #calendar.monthrange gets (week start, days in month) index for days in month 
	end_date = f'{year_of_interest}-{season_window_dates(season)[0]}-{calendar.monthrange(year_of_interest,int(season_window_dates(season)[0]))[1]}'

	#add a date column
	dates_list = pd.date_range(start_date,end_date)
	dates_list = dates_list.to_pydatetime() #change those timestamp objects to dates
	days_list = range(len(dates_list))
	days_to_dates = dict(zip(days_list,dates_list))
	input_df['unique_id'] = np.arange(input_df.shape[0])
	input_df['date'] = input_df.unique_id.map(days_to_dates)
	input_df['date'] = pd.to_datetime(input_df['date'])

	return input_df


def calculate_long_term_snow_drought(input_dict,start_date,end_date,year_of_interest,hucs): 
	"""Calculate long term snow drought for all years of snotel data collected."""
		#for station_id in sites_ids: 
	swe_dict = input_dict['wteq']
	precip_dict = input_dict['prec'] #this could change depending on which variable is used
	temp_dict = input_dict['tavg']
	
	output_df = pd.DataFrame(columns=['station_id','huc_id','dry','warm','warm_dry'])
	for k,v in swe_dict.items(): #iterate a dict that looks like {'station_id':df of years}

		try: 
			precip_df = precip_dict[k]
			temp_df = temp_dict[k]
			start_year = max([get_years(v)[0],get_years(precip_df)[0],get_years(temp_df)[0]]) #get the min year from each of these dataframes and then get the most recent one so there is data for all years
			end_year = min([get_years(v)[1],get_years(precip_df)[1],get_years(temp_df)[1]]) #get the max year from each of the dataframes and then take the min so you get the earliest year that has data for all years
		except Exception as e: 
			print(f'The error at df splitting was {e}')

		for year in range(start_year,end_year+1): 
			
			if (v[f'WTEQ_{year}'].max() < v[f'stat_WTEQ'][0]) and (precip_df[f'PREC_{year}'].max()<precip_df['stat_PREC'][0]) and ((temp_df[f'TAVG_{year}'][temp_df[f'TAVG_{year}']>0].count())<temp_df['stat_TAVG'][0]): #(temp_df[f'TAVG_{year}'].mean()<temp_df[f'stat_TAVG'][0]):
				#dry.append(year)#dry.update({station_id:year})
				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':year,'warm':np.nan,'warm_dry':np.nan},ignore_index=True)

			elif (v[f'WTEQ_{year}'].max() < v[f'stat_WTEQ'][0]) and (precip_df[f'PREC_{year}'].max()>=precip_df['stat_PREC'][0]): 
				#warm.append(year)#warm.update({station_id:year})
				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':year,'warm_dry':np.nan},ignore_index=True)

			elif (v[f'WTEQ_{year}'].max() < v[f'stat_WTEQ'][0]) and (precip_df[f'PREC_{year}'].max()<precip_df['stat_PREC'][0]) and ((temp_df[f'TAVG_{year}'][temp_df[f'TAVG_{year}']>0].count())>temp_df['stat_TAVG'][0]): 
				#warm_dry.append(year)#warm_dry.update({station_id:year})
				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':np.nan,'warm_dry':year},ignore_index=True)

			else: 
				pass
				print(f'station {k} for {year} was normal or above average. The swe value was: {v[f"WTEQ_{year}"].max()} and the long term mean max value was: {v[f"stat_WTEQ"][0]}')
				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':np.nan,'warm_dry':np.nan},ignore_index=True)

	print(output_df)
	return output_df


def calculate_short_term_snow_drought(input_dict,year_of_interest,agg_step,hucs,season): 
	"""Calculate the weekly snow drought for a year of interest."""
	
	swe_dict = input_dict['wteq']
	precip_dict = input_dict['prec'] #this could change depending on which variable is used
	temp_dict = input_dict['tavg']

	output_df = pd.DataFrame(columns=['station_id','huc_id','dry','warm','warm_dry'])
	print(swe_dict['2029'])
	print(hucs['2029'])
	for k,v in swe_dict.items(): #iterate a dict that looks like {'station_id':df of years}
		print('the station id is: ', k)
		if (k == '2029') or (k ==2029): #something weird is happening with this station and I am not sure what it is. We're going to pass it for now and come back
			pass
		else: 

			try: 
				precip_df = precip_dict[k]
				temp_df = temp_dict[k]
				wteq_df = v
			except Exception as e: 
				print(f'The error at df splitting was {e}')
			
			wteq_df = add_dates_to_df(wteq_df,season,year_of_interest)
			precip_df = add_dates_to_df(precip_df,season,year_of_interest)
			temp_df = add_dates_to_df(temp_df,season,year_of_interest)
			
			for i in range(0,wteq_df.shape[0],agg_step): #get the number of rows in the dataframe which is the season length. 
				print('i is ', i)
		
				try: 
					wteq_df_sub = wteq_df.iloc[i:i+agg_step]
					precip_df_sub = precip_df.iloc[i:i+agg_step]
					temp_df_sub = temp_df.iloc[i:i+agg_step]

				except Exception as e: 
					raise
					print('final week')
					wteq_df_sub = wteq_df.iloc[i:]
					precip_df_sub = precip_df.iloc[i:]
					temp_df_sub = temp_df.iloc[i:]
			
				wteq_df_sub.drop(columns='stat_WTEQ',inplace=True)
				
				precip_df_sub.drop(columns='stat_PREC',inplace=True)
				
				temp_df_sub.drop(columns='stat_TAVG',inplace=True)
				
				wteq_df_sub['week_wteq'] = (wteq_df_sub[wteq_df_sub.columns.difference(['date'])].max()).mean() #df["C"] = df[["A", "B"]].max(axis=1)
				
				precip_df_sub['week_prcp'] = (precip_df_sub[precip_df_sub.columns.difference(['date'])].max()).mean()
				
				temp_df_sub['week_tavg'] = (temp_df_sub[temp_df_sub.columns.difference(['date'])][temp_df_sub[temp_df_sub.columns.difference(['date'])] > 0 ].count()).mean() 
				
				current_date_start = wteq_df_sub.loc[wteq_df_sub['unique_id']==i]['date'].iloc[0] #get the time step start date, select the first item of the series (date) and make sure its a date object 
				try: 
					if (wteq_df_sub[f'WTEQ_{year_of_interest}'].max() < wteq_df_sub[f'week_wteq'].iloc[0]) and (precip_df_sub[f'PREC_{year_of_interest}'].max()<precip_df_sub['week_prcp'].iloc[0]) and ((temp_df_sub[f'TAVG_{year_of_interest}'][temp_df_sub[f'TAVG_{year_of_interest}']>0].count())<temp_df_sub[f'week_tavg'].iloc[0]): 
						#print(f"{wteq_df_sub[f'WTEQ_{year_of_interest}'].max()} < {wteq_df_sub[f'week_wteq'].iloc[0]}")
						output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':current_date_start,'warm':np.nan,'warm_dry':np.nan},ignore_index=True)

					elif (wteq_df_sub[f'WTEQ_{year_of_interest}'].max() < wteq_df_sub[f'week_wteq'].iloc[0]) and (precip_df_sub[f'PREC_{year_of_interest}'].max()>=precip_df_sub['week_prcp'].iloc[0]): 
						output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':current_date_start,'warm_dry':np.nan},ignore_index=True)

					elif (wteq_df_sub[f'WTEQ_{year_of_interest}'].max() < wteq_df_sub[f'week_wteq'].iloc[0]) and (precip_df_sub[f'PREC_{year_of_interest}'].max()<precip_df_sub['week_prcp'].iloc[0]) and ((temp_df_sub[f'TAVG_{year_of_interest}'][temp_df_sub[f'TAVG_{year_of_interest}']>0].count())>temp_df_sub[f'week_tavg'].iloc[0]): 
						output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':np.nan,'warm_dry':current_date_start},ignore_index=True)
					
					else:
						print('passed')
				except Exception as e: 
					print(f'Error was {e}')
					continue
	print(output_df)
	return output_df



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
		output_filepath=variables["output_filepath"]
		season = variables["season"]
		csv_dir = variables["csv_dir"]
		stations = variables["stations"]
		start_date = variables["start_date"]
		end_date = variables["end_date"]
		anom_start_date = variables["anom_start_date"]
		anom_end_date = variables["anom_end_date"]
		pickles = variables["pickles"]
		time_step = variables['time_step']
		agg_step = variables["agg_step"]
		year_of_interest = int(variables["year_of_interest"])
	#get some of the params needed
	stations_df = pd.read_csv(stations)
	sites = combine.make_site_list(stations_df,'huc_08',8)
	sites_full = sites[0] #this is getting the full df of oregon (right now) snotel sites. Changed 10/28/2020 so its just getting all of the site ids
	sites_ids = sites[1] #this is getting a list of just the snotel site ids. You can grab a list slice to restrict how big results gets below. 
	huc_dict = sites[2] #this gets a dict of format {site_id:huc12 id}

	huc_dict = {str(k):str(v) for k,v in huc_dict.items()}
	huc_list = list(set(i for i in huc_dict.values())) 
	
	##########################################################################
	#read in the SNOTEL station data
	filename = pickles+f'master_dict_all_states_all_years_all_params_dict_correctly_combined_{season}_updated'

	if os.path.exists(filename): 
		master_param_dict=combine.pickle_opener(filename)
		pd.set_option("display.max_rows", None, "display.max_columns", None) #change to print an entire df

		#print(master_param_dict['tavg']['304'].iloc[:,0])
	else: 
		print(f'The filename {filename} you supplied for the snotel data is incorrect. Please check the file path and try again.')

	# ########################################################################
	#this is working and generates the snow drought years or weeks

	long_term_snow_drought_filename = pickles+f'long_term_snow_drought_{season}_w_hucs'
	short_term_snow_drought_filename = pickles+f'short_term_snow_drought_{season}_{agg_step}_day_time_step_w_hucs'

	if not os.path.exists(long_term_snow_drought_filename): 
		long_term_snow_drought=calculate_long_term_snow_drought(master_param_dict,anom_start_date,anom_end_date,year_of_interest,huc_dict)
		pickle.dump(long_term_snow_drought, open(long_term_snow_drought_filename,'ab'))
	else: 
		print('Long term snow drought exists, passing')

	if not os.path.exists(short_term_snow_drought_filename): 
		short_term_snow_drought = calculate_short_term_snow_drought(master_param_dict,year_of_interest,int(agg_step),huc_dict,season)
		pickle.dump(short_term_snow_drought, open(short_term_snow_drought_filename,'ab'))
	else: 
		print('Short term snow drought already exists, passing')
if __name__ == '__main__':
    main()

