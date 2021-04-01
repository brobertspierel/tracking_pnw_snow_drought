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
#import geoplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
import _3_obtain_all_data as obtain_data

def plot_snow_drought_ratios(input_data,pts_shapefile,pnw_shapefile,us_boundary,year_of_interest,**kwargs):
	pts_shapefile = pd.read_csv(pts_shapefile)
	gdf = gpd.GeoDataFrame(pts_shapefile, geometry=gpd.points_from_xy(pts_shapefile.lon, pts_shapefile.lat)) 
	

	try: 
		df = gdf.merge(input_data, on=['site_num'], how='inner')
	except ValueError as e: 
		gdf['site_num'] = gdf['site_num'].astype('int')
		input_data['site_num'] = input_data['site_num'].astype('int')
		df = gdf.merge(input_data, on=['site_num'], how='inner')
	print(df)
	type_list = ['dry_ratio','warm_ratio','warm_dry_ratio']

	pnw = gpd.read_file(pnw_shapefile)
	us_bounds = gpd.read_file(us_boundary)
	#df["B"] = df["A"].map(equiv)
	fig, ax = plt.subplots(1, 3,figsize=(18,18))
	#ax = ax.flatten()
	if 'palette' in kwargs: 
		palette = kwargs.get('palette').values()
	else: 
		palette = 'Reds'
	for x in range(3):  
		minx, miny, maxx, maxy = pnw.geometry.total_bounds

		#hucs.plot(ax=ax[x],color='lightgray', edgecolor='black')
		us_bounds.plot(ax=ax[x],color='#a6a6a6',edgecolor='black')
		pnw.plot(ax=ax[x],color='lightgray',edgecolor='black')
		
		if x < 2:  
			pcm=df.plot(column=type_list[x],ax=ax[x],legend=False,cmap=palette,vmin=0,vmax=1,edgecolor='black',markersize=45)#,norm=norm)
		else: 
			divider = make_axes_locatable(ax[x])
			cax = divider.append_axes('right', size='5%', pad=0.05)
			pcm=df.plot(column=type_list[x],ax=ax[x],legend=True,cax=cax,cmap=palette,vmin=0,vmax=1,edgecolor='black',markersize=45)#,norm=norm)

		ax[x].set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		ax[x].set_ylim(miny - 1, maxy + 1)

		if '_' in type_list[x]: 
			drought_type=" ".join(type_list[x].split("_")).capitalize()
			ax[x].set_title(f'{drought_type} snow drought \n {year_of_interest} water year')
		else: 
			ax[x].set_title(f'{type_list[x].capitalize()} snow drought \n {year_of_interest} water year')
		# for idx, row in hucs.iterrows():
		# 	ax[x].annotate(s=row['huc4'], xy=row['coords'],horizontalalignment='center')
		
		#add a context map
		# axins = inset_axes(ax[0], width="30%", height="40%", loc=4)#,bbox_to_anchor=(.1, .5, .5, .5),bbox_transform=ax[x].transAxes)
		# axins.tick_params(labelleft=False, labelbottom=False)
		# us_bounds.plot(ax=axins,color='darkgray', edgecolor='black')
		#hucs.plot(ax=axins,color='red', edgecolor='black')

	plt.tight_layout()
	plt.show()
	plt.close('all')

def main(year_of_interest,season,pickles,agg_step=12,huc_level='8'): 
	

	snotel_data = pickles+f'short_term_snow_drought_{year_of_interest}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'

	if season == 'extended_winter': 
		date_range = list(pd.date_range(start=f"{int(year_of_interest)-1}-11-01",end=f"{year_of_interest}-04-30"))
		num_periods = len(date_range)/int(agg_step)

	input_data = obtain_data.AcquireData(None,None,snotel_data,None,huc_level,None)	
	short_term_snow_drought = input_data.get_snotel_data()

	df = short_term_snow_drought.groupby(['station_id','huc_id']).agg({'dry': 'count','warm':'count','warm_dry':'count'}).reset_index()

	df= df.groupby('huc_id').agg({'dry':'median','warm':'median','warm_dry':'median'}).reset_index()
	#df1 = short_term_snow_drought.groupby(['station_id']).agg({'dry': 'count','warm':'count','warm_dry':'count'}).reset_index()

	
	#print(df1)
	for column in ['dry','warm','warm_dry']: 
		df[f'{column}_ratio'] = df[column] /num_periods
	df.rename(columns={'station_id':'site_num'},inplace=True)
	
	#plot_snow_drought_ratios(df,stations_no_hucs,pnw_shapefile,us_boundary,year_of_interest)
	return df 

if __name__ == '__main__':
   

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		season = variables['season']
		agg_step = variables['agg_step']
		huc_level = variables['huc_level']
		year_of_interest = variables['year_of_interest']
		pickles = variables['pickles']
		stations = variables['stations']
		stations_no_hucs = variables['stations_no_hucs']
		us_boundary = variables['us_boundary']
		pnw_shapefile = variables['pnw_shapefile']
		palette = variables['palette']

	main(year_of_interest,season,pickles)