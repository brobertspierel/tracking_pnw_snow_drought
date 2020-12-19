import os
import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import sentinel_data_analysis as sentinel 
import snotel_data_analysis as snotel 
import sp_data_analysis as optical 
import snotel_intermittence_functions as snotel_functions
import json
from datetime import date, timedelta,datetime
import scipy as sp
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.lines as mlines


class AcquireData(): 
	def __init__(self,water_year_start,water_year_end,huc_level): 
		self.water_year_start=water_year_start
		self.water_year_end=water_year_end
		self.huc_level=huc_level

	def get_sentinel_data(self,csv_dir,orbit):
		"""Get cleaned and plottable output from sentinel 1. These data are processed in the sentinel_data_analysis.py script."""
		sentinel_data=sentinel.get_sentinel_data(csv_dir,self.huc_level,orbit,self.water_year_start,self.water_year_end)
		return sentinel_data

	def get_optical_data(self,csv_dir): 
		"""Get cleaned and plottable output from optical data, either Landsat or MODIS/VIIRS. This relies on the sp_data_analysis.py script."""
		optical_data = optical.get_sp_data(csv_dir)
		return optical_data
	def get_snotel_data(self,pickled_file): 	
		snow_droughts=snotel_functions.pickle_opener(pickled_file)
		return snow_droughts
def clean_huc_snotel_data(input_dict): 
	#for k,v in input_dict.items(): 
		#for k1,v1 in v.items(): 
			#print float(sum(d['value'] for d in v)) / len(v)
	output_dict = {}
	for k1,v1 in input_dict.items(): #each of these are now formatted as the huc id is k and then each v1 is a dict of warm,dry,warm dry with each of those a dictionary of stations/years or weeks
		#print('k1 is: ',k1)
		
		#print('v1 is: ', v1)
		dry_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in v1['dry'].items()])).mean(axis=1) #changed from input_dict 
		warm_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in v1['warm'].items()])).mean(axis=1)
		warm_dry_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in v1['warm_dry'].items()])).mean(axis=1)
		#print(type(dry_df))
		mean_ratios = {'dry':dry_df[0],'warm':warm_df[0],'warm_dry':warm_dry_df[0]}
		output_dict.update({k1:mean_ratios})

	return output_dict
def clean_weekly_counts_snotel_data(input_dict,start_date,end_date): 
	"""Get a dictionary of weeks and counts of stations in drought and make plottable. Dict is formatted like: {dry:df[time,counts],etc} """
	#generate a list/df of dates to match with the dfs
	dates_list = pd.date_range(start_date,end_date)
	dates_list = dates_list.to_pydatetime() #change those timestamp objects to dates
	#dates_list = [i.strftime('%Y-%m-%d') for i in dates_list] #change that date to a string so its readable- this might need to be amended
	#print(dates_list)
	days_list = range(len(dates_list))
	days_to_dates = dict(zip(days_list,dates_list))
	for k,v in input_dict.items(): #these are now hucs as k 
		for k1,v1 in v.items(): 
			v1['date'] = v1['time'].map(days_to_dates)
			v1['date'] = pd.to_datetime(v1['date'])
			v1['week_of_year'] = v1['date'].dt.week 
			#add a month col
			v1['month'] = v1['date'].dt.month
			#get the week of year where October 1 falls for the year of interest 
			base_week = datetime(v1['date'].iloc[0].to_pydatetime().year,10,1).isocalendar()[1]
			v1.loc[v1.month >= 10,'week_of_year'] = v1.week_of_year-base_week
			#adjust values that are after the first of the calendar year 
			v1.loc[v1.month < 10, 'week_of_year'] = v1.week_of_year + 12
			#print(v)
	#print(days_to_dates)
	#print(input_dict)
	return input_dict 
def select_low_elevation_snotel(input_csv): 
	df = pd.read_csv(input_csv)
	print(df)
	upper_limit = df['elev'].quantile(.25) #get the 25% quantile of elevation
	df = df[df['elev']<=upper_limit] 
	df['huc_id'] = df['huc_id'].astype('float')
	#df['huc_id'] = df['huc_id'].astype('str')

	print(df['huc_id'].iloc[0])

	return df
	#low, high = df.B.quantile([0.25,0.75])
	#df.query('{low}<B<{high}'.format(low=low,high=high))
def check_optical_scale(station_csv,optical_data,elev_col): 
	# station_df = pd.read_csv(station_csv)
	# print(optical_data.head())
	# print(list(set(optical_data.huc8)))
	# stations_subset = station_df.loc[station_df['huc_id'].isin(list(set(optical_data.huc8)))]
	# print(stations_subset)
	hucs = []
	elevs_stats = []
	pct_changes = []
	for huc in list(set(optical_data.huc8)): 
		df = optical_data[optical_data['huc8']==huc]
		df['pct_change'] = df['NDSI_Snow_Cover'].pct_change()
		pct_change_std = df['pct_change'].mean()
		elev_stat = df[elev_col].iloc[0] #get the first value for mean elev, these should all be the same for that hucID
		hucs.append(huc)
		elevs_stats.append(elev_stat)
		pct_changes.append(pct_change_std)

	output_df = pd.DataFrame({'huc_id':hucs,elev_col:elevs_stats,'pct_change_std':pct_changes})
	#print(output_df)
	return output_df
def plot_optical_scale(input_df,elev_col): 
	print(input_df)
	fig,ax=plt.subplots()
	ax.scatter(input_df[elev_col],input_df['pct_change_std']) #plot a simple regression 
	linreg = sp.stats.linregress(input_df[elev_col],input_df.pct_change_std)
	#The regression line can then be added to your plot: -


	#df['anomaly'].plot.line(ax=ax,legend=False,color='darkblue',lw=2)
	#ax.scatter(df[year],df['filter'])
	ax.plot(np.unique(input_df[elev_col]), np.poly1d(np.polyfit(input_df[elev_col], input_df.pct_change_std, 1))(np.unique(input_df[elev_col])))
	ax.set_xlabel('Mean basin elevation (m)')
	ax.set_ylabel('Standard deviation of \n cumulative percent change')
	ax.set_title('Mean elevation vs MODIS \n snow persistence cumalitve percent change')
	X2 = sm.add_constant(input_df[elev_col])
	est = sm.OLS(input_df.pct_change_std, X2)
	print(est.fit().f_pvalue)
	#Similarly the r-squared value: -
	ax.annotate(f'r2 = {round(linreg.rvalue,2)}',xy=(0.7,0.75),xycoords='figure fraction')

	#plt.show()
	# fig,ax = plt.subplots(10,10)
	# ax=ax.flatten()
	# count=0
	# for i in list(set(input_df['huc_id'])):
	# 	plot_df = input_df[input_df['huc_id']==i] 
	# 	ax[count].scatter(plot_df['mean_elev'],plot_df['pct_change_std'])
	# 	count +=1
	plt.show()
	plt.close('all')

def combine_data_and_plot(sentinel_data,optical_data,snotel_data): 
	fig,ax = plt.subplots(4,4,figsize=(10,10))#,sharex=True,sharey=True)
	ax = ax.flatten()
	hucs = list(set(list(sentinel_data['huc4'])))
	#hucs = [str(i) for i in hucs] #convert all the list elements to string
	
	for x in range(len(hucs)): 
		sentinel_df = sentinel_data[sentinel_data['huc4']==hucs[x]]
		sentinel_df['scaled'] = (sentinel_df['filter']-sentinel_df['filter'].min())/(sentinel_df['filter'].max()-sentinel_df['filter'].min()) #scale data to 0-1
		sentinel_df['pct_change'] = sentinel_df['scaled'].pct_change()#.rolling(2).std()
		sentinel_df['change_from_mean'] = sentinel_df['scaled']-sentinel_df['scaled'].mean()
		optical_df = optical_data[optical_data['huc4']==hucs[x]]
		optical_df['pct_change'] = optical_df['NDSI_Snow_Cover'].pct_change()#.rolling(2).std()
		#optical_df['sentinel_date'] = sentinel_df['date_time']
		optical_df['change_from_mean'] = optical_df['NDSI_Snow_Cover']-optical_df['NDSI_Snow_Cover'].mean()
		dry_df = snotel_data[str(hucs[x])]['dry'].sort_values(by=['date']) #get the key:pair from the dictionary with that huc id then get the dry dict from there and then get the df associated with that drought type 
		warm_df = snotel_data[str(hucs[x])]['warm'].sort_values(by=['date'])
		warm_dry_df = snotel_data[str(hucs[x])]['warm_dry'].sort_values(by=['date'])
		weeks_of_interest = list(dry_df['week_of_year']) #get the weeks that we have snotel data for 
		sentinel_df = sentinel_df[sentinel_df['week_of_year'].isin(weeks_of_interest)] #get the sentinel data that for the weeks that align with the snotel data 
		#optical_df['sentinel_date'] = sentinel_df['date_time']
		#print(optical_df)
		#print(sentinel_df)
		#print(dry_df)
		plot_df = pd.merge(sentinel_df,warm_dry_df,on='week_of_year', how='left')
		print(plot_df)
		#print(snotel_data[int(hucs[x])])
		#print(optical_df)
		#regress optical and sar
		ax[x].scatter(plot_df['scaled'],plot_df['counts'])#.iloc[:sentinel_df.shape[0]])
		#ax[x].scatter(sentinel_df['pct_change'],optical_df['pct_change'].iloc[:sentinel_df.shape[0]])
		#plot everything individually
		#ax[x].plot(sentinel_df['date_time'],sentinel_df['scaled'],color='darkgreen',label='Sentinel') #plot sentinel data
		#ax2 = ax[x].twinx()
		#ax2.plot(dry_df['date'],dry_df['counts'],color='darkred',label='SNOTEL')
		#ax[x].plot(optical_df['date'],optical_df['pct_change'],color='darkblue',label='Optical') #plot optical data
		# ax[x].axhline(y=snotel_data[str(hucs[x])]['dry'], color='r', linestyle='-',label='Snotel dry')
		# ax[x].axhline(y=snotel_data[str(hucs[x])]['warm'], color='black', linestyle='-',label='Snotel warm')
		# ax[x].axhline(y=snotel_data[str(hucs[x])]['warm_dry'], color='orange', linestyle='-',label='Snotel warm_dry')


		# ax[x].plot(snotel_data[str(hucs[x])]['dry'],label='dry')
		# ax[x].plot(snotel_data[str(hucs[x])]['warm'],label='warm')
		# ax[x].plot(snotel_data[str(hucs[x])]['warm_dry'],label='warm_dry')
		ax[x].tick_params(axis='x',labelrotation=90)
		ax[x].set_title(f'HUC4 ID {hucs[x]}')
		ax[4].set_xlabel('Date')
		ax[13].set_ylabel('Percent change from \n previous observation')
		ax[0].legend()
	#plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()
	plt.close('all')
def main():  
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		sentinel_csv_dir=variables['sentinel_csv_dir']
		huc_level=variables['huc_level']
		orbit=variables['orbit']
		water_year_start=variables['water_year_start']
		water_year_end=variables['water_year_end']	
		optical_csv_dir=variables['optical_csv_dir']
		pickles=variables['pickles']
		stations=variables['stations']

	sentinel_data = AcquireData(water_year_start,water_year_end,huc_level).get_sentinel_data(sentinel_csv_dir,orbit)
	optical_data = AcquireData(water_year_start,water_year_end,huc_level).get_optical_data(optical_csv_dir)
	snotel_data = AcquireData(None,None,None).get_snotel_data(pickles+f'2018_counts_of_stations_by_week_and_by_huc') #hardcoded, should be changed when running another year#pickles+'drought_by_basin_dict')
	low_elev_snotel = select_low_elevation_snotel(stations)
	cleaned_optical=check_optical_scale(stations,optical_data,'elev_max')
	plot_optical_scale(cleaned_optical,'elev_max')
	#print(snotel_data)
	#cleaned_snotel_data = clean_weekly_counts_snotel_data(snotel_data,'2017-10-01','2018-04-30')
	#print(cleaned_snotel_data)
	#combine_data_and_plot(sentinel_data,optical_data,cleaned_snotel_data)
	 
	
	
	#print(optical_data)
if __name__ == '__main__':
	main()