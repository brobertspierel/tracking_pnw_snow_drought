import os
import sys
import pandas as pd
import remote_sensing_functions as rs_funcs
import snotel_functions 


#define a class that is used to read in formatted data from snotel, sentinel 1 and MODIS/VIIRS

class AcquireData(): 

	def __init__(self,sentinel_data,optical_data,snotel_data,hucs_data,huc_level,resolution): 
		self.sentinel_data = sentinel_data
		self.optical_data = optical_data
		self.snotel_data = snotel_data
		self.hucs_data = hucs_data
		self.huc_level=huc_level
		self.resolution = resolution

	def get_sentinel_data(self,col_of_interest):
		"""Get cleaned and plottable output from sentinel 1."""
		sentinel_data=rs_funcs.combine_hucs_w_sentinel(self.hucs_data,self.sentinel_data,self.huc_level,self.resolution,col_of_interest) #for sentinel 1 the default col of interest is 'filter'
		return sentinel_data

	def get_optical_data(self): 
		"""Get cleaned and plottable output from optical data, either Landsat or MODIS/VIIRS."""
		optical_data = rs_funcs.read_csv(self.optical_data,'optical')
		return optical_data

	def get_snotel_data(self): 	
		"""Get cleaned snotel data ready for analysis."""
		snow_droughts=snotel_functions.pickle_opener(self.snotel_data)
		return snow_droughts