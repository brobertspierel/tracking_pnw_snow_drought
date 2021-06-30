Acquire SNOTEL data from NRCS
=============================

Example using the Python Climata library to acquire data for a time series for a selection of SNOTEL stations from NRCS web service. 

Dependencies: 
		* _0_acquire_snotel_data.py 


Code Example:: 
	
	class CollectData(): 
	#these functions are used to get snotel data, clean and organize. 

	def __init__(self,station,parameter,start_date,end_date,state):#,site_list,write_out,output_filepath): 
		self.parameter = parameter
		self.start_date = start_date
		self.end_date = end_date
		self.state = state
		self.station = station 

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

The main args you might supply to the paramater variable are: 
		* WTEQ - SWE (inches)
		* TAVG - daily average temperature (deg F)
		* PREC - cumulative precipitation (in)
		* PRCP - daily precipitation (in)
		* additional variables can be found at the `NRCS Web Service <https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#elementCodes>`_

This process can be carried out by iterating through a list of stations of interest and taking the output of the CollectData class (one pandas dataframe/station/variable) and appending to a list, dictionary etc. 

