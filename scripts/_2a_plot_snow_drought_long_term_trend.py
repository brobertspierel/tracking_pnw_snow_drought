
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
	"""Calculates the long term percent of time a basin should be classified as in snow drought."""
	
	nans = input_df.fillna(-9999).groupby([grouping_col])[col_of_interest].count().to_frame().reset_index()
	valid_count = (input_df.groupby([grouping_col])[col_of_interest].count()).to_frame().reset_index()
	output_df = pd.merge(nans, valid_count, on=grouping_col)

	output_df['pct_drought'] = (output_df[col_of_interest+'_y']/output_df[col_of_interest+'_x'])*100

	return output_df
def snow_drought_counts_by_decade(input_df,col_of_interest,grouping_col): 
	"""Gets the count of snow droughts in a decade. Basin-wise operation."""


def define_snow_drought_recentness(input_df,col_of_interest,grouping_col,output_var):
	"""Classify how recently the snow droughts are happening in a given basin."""

	#we want to get periods of 1985-1990, 1991-2000, 2001-2010 and 2011-2020

	output_dict = {}
	print('The input df is: ', input_df)
	for item in input_df[grouping_col].unique(): 
		df_subset = input_df[input_df[grouping_col]==item]
		filter_values = pd.IntervalIndex.from_tuples([(1980, 1989), (1990, 1999), (2000, 2009),(2010,2019)],closed='both')
		out = df_subset[['dry','warm','warm_dry']].apply(pd.cut,bins=filter_values)
		counts = out.apply(pd.Series.value_counts)
		if not output_var: #default here is to get the max value which is for recentness plot. If output_var evaluates to True (ie not None) it will not send the max but just the counts 
			counts=counts.idxmax()
		else: 
			pass
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
		pnw_shapefile = variables["pnw_shapefile"]
		huc_shapefile = variables['huc_shapefile']
		us_boundary = variables['us_boundary']
		epsg = variables["epsg"]
		output_filepath=variables["output_filepath"]
		anom_start_date = variables["anom_start_date"]
		anom_end_date = variables["anom_end_date"]
		season = variables["season"]
		csv_dir = variables["csv_dir"]
		stations = variables["stations"]		
		pickles = variables["pickles"]
		year_of_interest = int(variables["year_of_interest"])
		palette = variables["palette"]

		palette = list(palette.values())
		labels=['Dry', 'Warm', 'Warm/dry', 'No drought']
		plot_func = 'recentness' #can be one of count, year_totals, recentness
		try: 
			long_term_snow_drought = combine.pickle_opener(pickles+f'long_term_snow_drought_{anom_start_date}_{anom_end_date}_{season}_w_hucs') #	long_term_snow_drought_filename = pickles+f'long_term_snow_drought_{anom_start_date}_{anom_end_date}_{season}_w_hucs'

			#short_term_snow_drought = combine.pickle_opener(pickles+f'short_term_snow_drought_{season}_{agg_step}_time_step_w_hucs')			
			print(long_term_snow_drought)
		except Exception as e: 
			print('File {e} does not exist. Make sure the snow drought files are correct and re-run')
		
		if plot_func.lower() == 'count': #plot the long term percent of time a basin is in drought 
			counts = create_long_term_snow_drought_counts(long_term_snow_drought,'warm_dry','huc_id')
			print(counts)

		elif plot_func.lower() == 'year_totals': 
			dry_counts = long_term_snow_drought['dry'].value_counts()
			warm_counts = long_term_snow_drought['warm'].value_counts()
			warm_dry_counts = long_term_snow_drought['warm_dry'].value_counts()
			
			print('dry')
			print(dry_counts)
			print('warm')
			print(warm_counts)
			print('warm_dry')
			print(warm_dry_counts)
			plt.rcParams.update({'font.size': 18})


			df = pd.DataFrame({"dry":dry_counts,"warm":warm_counts,"warm_dry":warm_dry_counts})
			#fig,ax = plt.subplots()
			
			ax = df.plot.bar(color=palette[:-1], rot=0)#,label=['Dry','Warm','Warm/dry']) #need to do something with the color scheme here and maybe error bars? This could be better as a box plot? 
			ax.grid()
			ax.set_xlabel(" ")
			ax.set_ylabel("Snow drought quantities")
			ax.legend(labels=labels)

			plt.xticks(rotation=45)
			plt.show()
			plt.close('all')
		
		elif plot_func.lower() == 'recentness': 
			#plot_func the long term 'recentness' of snotel snow drought
		
			recentness=define_snow_drought_recentness(long_term_snow_drought,'warm_dry','huc_id',None)
		
			dry = {}
			warm = {}
			warm_dry = {}


			for k,v in recentness.items(): 
				try: 
					#get the start year of the decade for each drought type 
					dry.update({k:int(v.at['dry'].left)})
					warm.update({k:int(v.at['warm'].left)})
					warm_dry.update({k:int(v.at['warm_dry'].left)})
				except Exception as e: 
					print('The error here was {e} and is likely the result of not running in max mode above.')
			
			#read in shapefiles
			hucs_shp = gpd.read_file(huc_shapefile)
			us_bounds = gpd.read_file(us_boundary)
			pnw_states = gpd.read_file(pnw_shapefile)
			hucs_shp['huc8'] = hucs_shp['huc8'].astype('int32')
			pd.set_option("display.max_rows", None, "display.max_columns", None) #change to print an entire df
		
			hucs_shp['dry_colors'] = hucs_shp.huc8.map(dry)
			hucs_shp['warm_colors'] = hucs_shp.huc8.map(warm)
			hucs_shp['warm_dry_colors'] = hucs_shp.huc8.map(warm_dry)
			#print(hucs_shp)
			print(hucs_shp['dry_colors'].value_counts())
			print(hucs_shp['warm_colors'].value_counts())
			print(hucs_shp['warm_dry_colors'].value_counts())

			# hucs_shp['epoch_color_dry'] = hucs_shp['huc8'].map(recentness.values().at('dry'))
			
			# hucs['snow_droughts'] = hucs['huc8'].map(basin_droughts_by_year)
			# print(hucs.snow_droughts.iloc[15])

			#the column with the id here is called 'huc8'
			
			#colors = ['#fee5d9','#fcae91','#fb6a4a','#cb181d']
			bounds = [1980,1990,2000,2010,2020]
			cmap = ListedColormap(palette)
			fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True,figsize=(18,14))
			minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds

			#plot dry snow drought
			us_bounds.plot(ax=ax1,color='white', edgecolor='black')
			pnw_states.plot(ax=ax1,color='white',edgecolor='darkgray')
			#hucs_shp.plot(ax=ax1,color='gray',edgecolor='darkgray')
			hucs_shp.plot(ax=ax1,column='dry_colors',cmap=cmap,vmin=1980,vmax=2020)#, column='Value1')
			ax1.set_title('Dry snow drought')
			ax1.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
			ax1.set_ylim(miny - 1, maxy + 1)

			#ctx.add_basemap(ax1)
			#hucs.plot(ax=axins,color='red', edgecolor='black')
			#plot warm snow drought
			us_bounds.plot(ax=ax2,color='white', edgecolor='black')
			pnw_states.plot(ax=ax2,color='white',edgecolor='darkgray')
			hucs_shp.plot(ax=ax2,color='gray',edgecolor='darkgray')
			hucs_shp.plot(ax=ax2, column='warm_colors',cmap=cmap,vmin=1980,vmax=2020)
			ax2.set_title('Warm snow drought')
			ax2.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
			ax2.set_ylim(miny - 1, maxy + 1)
			
			#ctx.add_basemap(ax2)
			#define things for colorbar 
			divider = make_axes_locatable(ax3)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			us_bounds.plot(ax=ax3,color='white', edgecolor='black')
			pnw_states.plot(ax=ax3,color='white',edgecolor='darkgray')
			hucs_shp.plot(ax=ax3,color='gray',edgecolor='darkgray')
			hucs_shp.plot(ax=ax3, column='warm_dry_colors', cmap=cmap,vmin=1980,vmax=2020)#,legend=True,cax=cax)#,vmin=1985,vmax=2019)
			ax3.set_title('Warm/dry snow drought')
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
			
			#turn off the box in the fourth plot. This is where we'll put the caption. Might want to annotate it in here as well. 
			ax4.axis('off')
			
			plt.tight_layout()#rect=[0, 0.03, 1, 0.95])
			plt.show()
			plt.close('all')
		else: 
			print('doing neither recentness nor counts')
if __name__ == '__main__':
    main()