import os 
import sys
import glob
import pandas as pd 
import numpy as np 
import geopandas as gpd
import json 
import matplotlib.pyplot as plt  
import seaborn as sns 
import remote_sensing_functions as rs_funcs
import _4a_calculate_remote_sensing_snow_droughts as _4a_rs
import _3_obtain_all_data as obtain_data
import _4bv1_calculate_long_term_sp as _4b_rs 
import re
import math 
from scipy import stats




def main():
	"""
	Link the datatypes together and add summary stats. 
	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		season = variables["season"]
		pickles = variables["pickles"]
		csv_dir = variables["csv_dir"]
		palette = variables["palette"]

		
		dry=_4b_rs.generate_output(_4b_rs.combine_rs_snotel_annually(csv_dir,season,pickles,'dry'),'dry')
		warm=_4b_rs.generate_output(_4b_rs.combine_rs_snotel_annually(csv_dir,season,pickles,'warm'),'warm')
		warm_dry=_4b_rs.generate_output(_4b_rs.combine_rs_snotel_annually(csv_dir,season,pickles,'warm_dry'),'warm_dry') 
		print(dry)
if __name__ == '__main__':
    main()