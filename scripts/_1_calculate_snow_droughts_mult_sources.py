import pandas as pd 
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt 
import geopandas as gpd 
import json 
import glob
import datetime

"""Note that this was set up to format data which was acquired and pickled with the original version of the snotel data obtaining script. 
This original script formatted data as a dictionary."""

class FormatData(): 
	def __init__(self,input_files,time_period=None,drop_cols=None,date_col='date'): 
		self.input_files=input_files
		self.drop_cols=drop_cols
		self.date_col=date_col
		self.time_period=time_period


	def read_in_csvs(self): 
		files = []
		for file in self.input_files: 
			df = pd.read_csv(file)
			files.append(df)
		output = pd.concat(files)
		#make sure the date col is cast as date type 
		output[self.date_col]= pd.to_datetime(output[self.date_col])
		try: 
			output=output.drop(columns=self.drop_cols,axis=1) 
			return output
		except Exception as e: 
			return output


	def read_in_pickles(self): 
		lists=[]
		for file in self.input_files: 
			lists.append(pd.read_pickle(file))
		flat_list = [item for sublist in lists for item in sublist]
		output=pd.concat(flat_list,ignore_index=True)

		#make sure the date col is cast as a date type 
		output[self.date_col] = pd.to_datetime(output[self.date_col])

		try: 
			output=output.drop(columns=self.drop_cols,axis=1) 
			return output
		except AttributeError as e: 
			return output


	def split_yearly_data(self,df): 
		"""Make chunks of data from a year."""
		#cast the date col to date 
		while True: 
			try: 
				df[self.date_col]= pd.to_datetime(df[self.date_col])
				break
			except KeyError as e:
				print('Your date column has a different name than date. The columns in this df are: ')
				print(df.columns)
				date_col = input('Which one would you like to use?')

		while True: 
			if self.time_period.lower() == 'early': 
				output = df.loc[(df[self.date_col].dt.month>=11)&(df[self.date_col].dt.month<=12)]
				break 
			elif self.time_period.lower() == 'mid': 
				output = df.loc[(df[self.date_col].dt.month>=1)&(df[self.date_col].dt.month<=2)]
				break
			elif self.time_period.lower() == 'late': 
				output = df.loc[(df[self.date_col].dt.month>=3)&(df[self.date_col].dt.month<=4)]
				break 
			elif self.time_period.lower() == 'year': #added 5/26/2021 to try running for a full winter season
				output = df.loc[(df[self.date_col].dt.month>=11)&(df[self.date_col].dt.month<=4)] 
				break
			else: 
				self.time_period = input('Your choice of time_period is invalid.\nYou can choose one of early, mid, or late. \nType your choice and hit enter.')
		return output

class CalcSnowDroughts(): 
	"""Calculate snow droughts from any data that has SWE, avg temp and precip. This approach is adopted from Dierauer et al 2019
	and is predicated on daily temporal resolution data. The approach assumes you are working with temporal chunks of some kind with the 
	default being three winter periods like: Nov-Dec, Feb-Jan, Mar-Apr.
	Inputs should look like: 
	input_df- this is a df that has date and basin (huc) cols as well as a col for swe, precip,  and temp. This df should be for the period in the year
	you want to calculate and include the full study time period. 
	the other args are the cols for the params you are using. They default to the cols for daymet but can be overriden in the class/function call. 
	"""
	def __init__(self,input_df,swe_c='swe',precip='prcp',temp='tavg',date_col='date',sort_col='huc8',start_year=1990): 
		self.input_df=input_df
		self.swe_c=swe_c
		self.precip=precip
		self.temp=temp
		self.date_col=date_col
		self.sort_col=sort_col
		self.start_year=start_year

	def prepare_df_cols(self):

		#add a year col for the annual ones 

		self.input_df['year'] = self.input_df[self.date_col].dt.year
		
		#restrict the dfs to 1990-doing this because some of the earlier years don't have enough data for the snotel stations 
		self.input_df=self.input_df.loc[self.input_df['year']>=self.start_year]
		
		if self.precip.upper() == 'PREC':
			#process for snotel- these are both cumulative variables so we want to take the max 
			#get agg stats for swe and precip
			print(f'Processing a precip col called PREC and swe called {self.swe_c}')
			swe_prcp = self.input_df.groupby([self.sort_col,'year'])[self.swe_c,self.precip].max().reset_index()#.agg({self.swe_c:'max',self.precip:'sum'})
		
		elif self.precip.lower() == 'prcp': 
			#process for daymet- precip is the daily sum and swe is cumulative
			print(f'Processing a precip col called prcp and a swe col called {self.swe_c}')
			swe_prcp = self.input_df.groupby([self.sort_col,'year']).agg({self.swe_c:'max',self.precip:'sum'}).reset_index() 
		
		else: 
			#deal with an instance where those cols are something else
			print('Did not find a column called PREC or prcp.')
			precip_c = input('Please input the name of your precip column (case sensitive): ')
			swe_c = input('Please input the name of your swe column (case sensitive): ')
			precip_stat = input('Please input the summary stat for precip: ')
			swe_stat = input('Please input the summary stat for swe: ')

			swe_prcp = self.input_df.groupby([self.sort_col,'year']).agg({swe_c:swe_stat,precip_c:precip_stat}).reset_index() 

		#get agg stats for temp, this is a little harder because its a degree day model. 
		#NOTE it is critical that temperatures are in deg C and not in deg F 
		self.input_df.loc[self.input_df[self.temp] < 0, self.temp] = 0 
		temp_df = self.input_df.groupby([self.sort_col,'year'])[self.temp].sum().reset_index()
		#combine the three vars 
		sd_df = swe_prcp.merge(temp_df,how='inner',on=[self.sort_col,'year'])
		#get the long-term means 
		means = sd_df.groupby(self.sort_col).agg({self.swe_c:'mean',self.precip:'mean',self.temp:'mean'})
		#rename the means cols so when they merge they have distinct names 
		means.columns = ['mean_'+column for column in means.columns]
		#print(means)
		
		#merge the means with the summary stats for each year/basin- this can be split for the three processing periods 
		sd_df = sd_df.merge(means,how='inner',on=self.sort_col)

		return sd_df

	def calculate_snow_droughts(self): 
		#define some stats for if/else to get snow droughts.
		#get a df of data from the previous function
		sd_df = self.prepare_df_cols()

		#do dry first- if the conditions evaluate to true add the year and if not just fill with nan 
		sd_df['dry'] = np.where((sd_df[self.swe_c]<sd_df[f'mean_{self.swe_c}']) & 
			(sd_df[self.precip]<sd_df[f'mean_{self.precip}'])&(sd_df[self.temp]<=sd_df[f'mean_{self.temp}']),sd_df['year'],np.nan)
		
		#next do warm
		sd_df['warm'] = np.where((sd_df[self.swe_c]<sd_df[f'mean_{self.swe_c}']) & 
			(sd_df[self.precip]>=sd_df[f'mean_{self.precip}']),sd_df['year'],np.nan)
		
		#then do warm/dry 
		sd_df['warm_dry'] = np.where((sd_df[self.swe_c]<sd_df[f'mean_{self.swe_c}']) & 
			(sd_df[self.precip]<sd_df[f'mean_{self.precip}'])&(sd_df[self.temp]>sd_df[f'mean_{self.temp}']),sd_df['year'],np.nan)

		return sd_df
