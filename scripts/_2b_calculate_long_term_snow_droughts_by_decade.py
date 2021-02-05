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
from collections import defaultdict
import numpy as np 
#import geoplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import _2a_plot_snow_drought_long_term_trend as sd_trend


def create_long_term_snow_drought_counts(input_df,col_of_interest,grouping_col): 
	"""Calculates the long term percent of time a basin should be classified as in snow drought."""

	nans = input_df.fillna(-9999).groupby([grouping_col])[col_of_interest].count().to_frame().reset_index()

	valid_count = (input_df.groupby([grouping_col])[col_of_interest].count()).to_frame().reset_index()

	output_df = pd.merge(nans, valid_count, on=grouping_col)

	output_df['pct_drought'] = (output_df[col_of_interest+'_y']/output_df[col_of_interest+'_x'])*100

	return output_df

def main():
	"""
	Plot the long term snow drought types and trends. 
	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		pnw_shapefile = variables["pnw_shapefile"]
		huc_shapefile = variables['huc_shapefile']
		us_boundary = variables['us_boundary']
		epsg = variables["epsg"]
		output_filepath=variables["output_filepath"]
		season = variables["season"]
		csv_dir = variables["csv_dir"]
		stations = variables["stations"]		
		pickles = variables["pickles"]
		year_of_interest = int(variables["year_of_interest"])

		plot_func = 'recentness'
		try: 
			long_term_snow_drought = combine.pickle_opener(pickles+f'long_term_snow_drought_{season}_w_hucs')
		except Exception as e: 
			print('File {e} does not exist. Make sure the snow drought files are correct and re-run')

		# pct_drought = create_long_term_snow_drought_counts(long_term_snow_drought,'warm_dry','huc_id')
		# print(pct_drought)
		counts = sd_trend.define_snow_drought_recentness(long_term_snow_drought,'warm_dry','huc_id','counts')
		# for k,v in counts.items(): 
		# 	v['dry_pct_change'] = v['dry'].pct_change()
		# 	v['warm_pct_change'] = v['warm'].pct_change()
		# 	v['warm_dry_pct_change'] = v['warm_dry'].pct_change()
		print(counts)

if __name__ == '__main__':
    main()