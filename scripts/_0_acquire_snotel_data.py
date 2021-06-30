import pandas as pd 
import os
import sys
import numpy as np 
from climata.snotel import StationDailyDataIO 
from pathlib import Path
import pickle 
import datetime
import zeep
from datetime import datetime, timedelta
import json

class CollectData(): 
	#these functions are used to get snotel data, clean and organize. 

	def __init__(self,station,parameter,start_date,end_date,state):#,site_list,write_out,output_filepath): 
		#self.station_csv = station_csv
		#self.station = station
		self.parameter = parameter
		self.start_date = start_date
		self.end_date = end_date
		self.state = state
		self.station = station 
		# self.site_list = site_list
		# self.output_filepath = output_filepath
		# self.write_out = write_out #a boolean dictating whether to pickle the collected data 

	def format_station(self): 
		return f'{self.station}:{self.state}:SNTL'

	def get_snotel_data(self):#station, parameter,start_date,end_date): #create a function that pulls down snotel data
		"""Collect snotel data from NRCS API. The guts of the following code block comes from: 
		https://pypi.org/project/climata/."""

		data = StationDailyDataIO(
		station=self.format_station(), #station id
		start_date=self.start_date, 
		end_date=self.end_date,
		parameter=self.parameter #assign parameter
		)
		#Display site information and time series data

		for series in data: 
			#print(series.data)
			snow_var=pd.DataFrame([row.value for row in series.data]) #create a dataframe of the variable of interest
		

			date=pd.DataFrame([row.date for row in series.data]) #get a dataframe of dates
		
		try: 
			df=pd.concat([date,snow_var],axis=1) #concat the dataframes
			df.columns=[f'{self.parameter}_date',f'{self.parameter}'] #rename columns
			df['site_id']=str(self.station) #create an id column from input station 
			return df 
		except UnboundLocalError: 
			print(f'Did not get any data for station {self.station}')

		

def main(snotel_data,pickle_dir): 
	
	output = {}
	snotel_df = pd.read_csv(snotel_data)
	
	output_fn = os.path.join(pickle_dir,'snotel_data_ref_basins.p')

	if not os.path.exists(output_fn):
		print('working...')

		for station,st in zip(snotel_df['ntwk_sta_i'],snotel_df['st_cd']): 
			print(station)
			count = 0
			dfs = []
			for var in ['WTEQ','PREC','TAVG']: 
				var_df = CollectData(station, var, '1980-10-01', '2020-09-30', st).get_snotel_data() #dates are hardcoded 
				if count > 0: 
					var_df.drop(columns='site_id',inplace=True) #after the first iteration drop redundant cols 
				dfs.append(var_df)
				count += 1 
				try: 
					out_df = pd.concat(dfs,axis=1)
				except ValueError: 
					print('Those dfs are empty.')
					continue 

			output.update({station:out_df})
		pickle.dump(output,open(output_fn,'wb'))

	else: 
		print('snotel dict exists.')
if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f: 
		variables = json.load(f)
		pickles = variables['pickles']
		snotel = variables['snotel']
	main(snotel,pickles)

#####################################################################################################
#this generally works but has some bugs, hold in case climata stops working 

# class CollectData(): 
# 	#these functions are used to get snotel data, clean and organize. 

# 	def __init__(self,station,parameter,start_date,end_date,state):#,site_list,write_out,output_filepath): 
# 		#self.station_csv = station_csv
# 		#self.station = station
# 		self.parameter = parameter
# 		self.start_date = start_date
# 		self.end_date = end_date
# 		self.state = state
# 		self.station = station 
# 		#set the NRCS defaults 
# 		self.ordinal = 1
# 		self.flags = 'true'
# 		self.Feb29 = 'true'
# 		self.duration = 'DAILY'
# 		self.wsdl = 'https://wcc.sc.egov.usda.gov/awdbWebService/services?WSDL'
# 		self.dates = 'true'


# 	def format_station(self): 
# 		return f'{self.station}:{self.state}:SNTL'

# 	def get_snotel_data(self): 
# 		"""Uses the zeep library to access a station's SNOTEL data from NRCS SOAP web service."""

# 		try: 
# 			client = zeep.Client(wsdl=self.wsdl)

# 			data=client.service.getData(stationTriplets=self.format_station(),
# 				elementCd=self.parameter,
# 				beginDate=self.start_date,
# 				endDate=self.end_date,
# 				ordinal=self.ordinal,
# 				duration=self.duration,
# 				getFlags=self.flags, 
# 				alwaysReturnDailyFeb29=self.Feb29,
# 				)

# 			return data 

# 		except zeep.exceptions.Fault as fault:
# 			parsed_fault_detail = client.wsdl.types.deserialize(fault.detail[0])
# 			print('That station had an error which was: ')
# 			print(parsed_fault_detail)


# class FormatData(): 

# 	def __init__(self,input_list, parameter): 
# 		self.input_list = input_list
# 		self.parameter = parameter

# 	def parse_data(self,dict_key):
# 		"""The output of the NRCS data is a list with a dictionary as the only item. 
# 		Pull that out to get the keys we want."""
# 		return self.input_list[0][dict_key]

# 	def date_from_str(self,date): 
# 		try: 
# 			return datetime.strptime(date,'%Y-%m-%d').date()
# 		except ValueError:
# 			raise
# 			# print('The date is not formatted like: yyyy-mm-dd, trying yyyy/mm/dd') 
# 			# return datetime.strptime(date,'%y/%m/%d')

# 	def make_date_range(self,start_date,end_date): 
# 		"""Take two dates and create a list of dates inbetween."""
# 		return pd.date_range(self.date_from_str(start_date), 
# 			self.date_from_str(end_date),freq='d') #-timedelta(days=1)

# 	def make_dfs(self): 
# 		"""Take lists from the NRCS dictionary and make them into a dataframe."""

# 		data = [{
# 			'date': date,
# 			'value': val,
# 			} for date, val in zip(self.make_date_range(self.parse_data('beginDate'),self.parse_data('endDate')),
# 			 self.parse_data('values'))]

# 		print(len(data))

# 		return data 
# 		# try: 
# 		# 	data = {'dates':self.make_date_range(), f'{self.parameter}':self.parse_data('values')}

# 		# 	df = pd.DataFrame.from_dict(data)
# 		# except ValueError: 
# 		# 	print('There was a len mismatch between the dates and the values.') 

# 		# return df 


# 	#pd.date_range(sdate,edate-timedelta(days=1),freq='d')

# parameter = 'WTEQ'
# start_date = '1980-10-01'
# end_date = '2020-09-01'
# state = 'AZ'

# test=CollectData(1140, parameter, start_date, end_date, state).get_snotel_data()

# print(test[0]['beginDate'])

# out = FormatData(test,parameter).make_dfs()

