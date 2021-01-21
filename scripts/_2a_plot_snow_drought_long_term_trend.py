
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


def create_long_term_snow_drought_counts(input_df,col_of_interest,grouping_col): 
	"""Helper function."""
	nans = input_df.fillna(-9999).groupby([grouping_col])[col_of_interest].count().to_frame().reset_index()

	valid_count = (input_df.groupby([grouping_col])[col_of_interest].count()).to_frame().reset_index()#.agg(mean=('returns', 'count'), sum=('returns', 'mode'))

	output_df = pd.merge(nans, valid_count, on=grouping_col)

	output_df['pct_drought'] = (output_df[col_of_interest+'_y']/output_df[col_of_interest+'_x'])*100

	return output_df

def define_snow_drought_recentness(input_df,col_of_interest,grouping_col):
	"""Classify how recently the snow droughts are happening in a given basin."""

	#we want to get periods of 1985-1990, 1991-2000, 2001-2010 and 2011-2020

	output_dict = {}
	for item in input_df[grouping_col].unique(): 
		df_subset = input_df[input_df[grouping_col]==item]
		filter_values = pd.IntervalIndex.from_tuples([(1985, 1989), (1990, 1999), (2000, 2009),(2010,2019)],closed='both')#[1985, 1989, 1999, 2009, 2019]   
		out = df_subset[['dry','warm','warm_dry']].apply(pd.cut,bins=filter_values)#(df_subset[['dry','warm','warm_dry']], bins=filter_values)
		counts = out.apply(pd.Series.value_counts)
		counts=counts.idxmax()
	
		output_dict.update({int(item):counts})

	return output_dict

def main():
	"""
	Plot the long term snow drought types and trends. 
	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		state_shapefile = variables["state_shapefile"]
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

		try: 
			long_term_snow_drought = combine.pickle_opener(pickles+f'long_term_snow_drought_{season}_w_hucs')
			#short_term_snow_drought = combine.pickle_opener(pickles+f'short_term_snow_drought_{season}_{agg_step}_time_step_w_hucs')			
			#print(long_term_snow_drought)
		except Exception as e: 
			print('File {e} does not exist. Make sure the snow drought files are correct and re-run')
		
		#plot the long term 'recentness' of snotel snow drought
		recentness=define_snow_drought_recentness(long_term_snow_drought,'warm_dry','huc_id')
		dry = {}
		warm = {}
		warm_dry = {}

		for k,v in recentness.items(): 

			dry.update({k:int(v.at['dry'].left)})
			warm.update({k:int(v.at['warm'].left)})
			warm_dry.update({k:int(v.at['warm_dry'].left)})

	
		hucs_shp = gpd.read_file(huc_shapefile)
		us_bounds = gpd.read_file(us_boundary)
		hucs_shp['huc8'] = hucs_shp['huc8'].astype('int32')
		pd.set_option("display.max_rows", None, "display.max_columns", None) #change to print an entire df
	
		hucs_shp['dry_colors'] = hucs_shp.huc8.map(dry)
		hucs_shp['warm_colors'] = hucs_shp.huc8.map(warm)
		hucs_shp['warm_dry_colors'] = hucs_shp.huc8.map(warm_dry)
		print(hucs_shp)
		# print(hucs_shp)
		# hucs_shp['epoch_color_dry'] = hucs_shp['huc8'].map(recentness.values().at('dry'))
		
		# hucs['snow_droughts'] = hucs['huc8'].map(basin_droughts_by_year)
		# print(hucs.snow_droughts.iloc[15])

		# #the column with the id here is called 'huc8'
		# colors = ['xkcd:pumpkin', "xkcd:bright sky blue", 'xkcd:light green', 
  #         'salmon', 'grey', 'xkcd:pale grey']
		colors = ['#fee5d9','#fcae91','#fb6a4a','#cb181d']
		bounds = [1980,1990,2000,2010,2020]
		cmap = ListedColormap(colors)
		fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, sharex=True, sharey=True,figsize=(15,15))
		minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds

		#plot dry snow drought
		us_bounds.plot(ax=ax1,color='white', edgecolor='black')
		hucs_shp.plot(ax=ax1,color='gray',edgecolor='darkgray')
		hucs_shp.plot(ax=ax1,column='dry_colors',cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
		ax1.set_title('Occurance of dry snow drought')
		ax1.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		ax1.set_ylim(miny - 1, maxy + 1)

		#ctx.add_basemap(ax1)
		#hucs.plot(ax=axins,color='red', edgecolor='black')
		#plot warm snow drought
		us_bounds.plot(ax=ax2,color='white', edgecolor='black')
		hucs_shp.plot(ax=ax2,color='gray',edgecolor='darkgray')
		hucs_shp.plot(ax=ax2, column='warm_colors',cmap=cmap,vmin=1980,vmax=2019)
		ax2.set_title('Occurance of warm snow drought')
		ax2.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		ax2.set_ylim(miny - 1, maxy + 1)
		
		#ctx.add_basemap(ax2)
		#define things for colorbar 
		divider = make_axes_locatable(ax3)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		us_bounds.plot(ax=ax3,color='white', edgecolor='black')
		hucs_shp.plot(ax=ax3,color='gray',edgecolor='darkgray')
		hucs_shp.plot(ax=ax3, column='warm_dry_colors', cmap=cmap,vmin=1980,vmax=2019)#,legend=True,cax=cax)#,vmin=1985,vmax=2019)
		ax3.set_title('Occurance of warm/dry snow drought')
		ax3.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		ax3.set_ylim(miny - 1, maxy + 1)
		#ctx.add_basemap(ax3)
		#fig.subplots_adjust(bottom=0.5)

		#cmap = mpl.colors.ListedColormap(colors)
		#cmap.set_over('0.25')
		#cmap.set_under('0.75')

		#bounds = [1980, 1990, 2000, , 8]
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		cb2 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
		                                norm=norm,
		                                boundaries=bounds,#[0] + bounds + [13]
		                                ticks=bounds,
		                                spacing='proportional',
		                                orientation='vertical')
		#cb2.set_label('Discrete intervals, some other units')
		#fig.show()
		plt.tight_layout()
		plt.show()
		plt.close('all')

if __name__ == '__main__':
    main()