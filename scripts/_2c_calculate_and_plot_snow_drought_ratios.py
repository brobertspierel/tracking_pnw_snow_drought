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


def snow_drought_ratios(input_dict,num_weeks): 
	output_dict = input_dict
	for k,v in input_dict.items(): #this is the top level dict of hucs
		for k1,v1 in v.items(): #this is the dict of weeks or years in dry, warm etc for each huc
				v.update({k1:len(v1)/int(num_weeks)})
	#print('output dict is: ')
	#print(output_dict)
	#print('num weeks is: ',num_weeks)
	#.rename(columns=['station_id','weekly_ratio']) 
	# print(warm_dry_df)
	# print(warm_dry_df.shape)

	return output_dict
def reformat_dict(input_dict): 
	return {int(k):round(v,2) for k,v in input_dict.items()}
def plot_snow_drought_ratios(input_dict,pnw_shapefile,huc_shapefile,us_boundary,input_pts_data,year_of_interest):
	#dry_df = pd.DataFrame(input_dict['dry'],index=['weekly_ratio']).T.reset_index().rename(columns={'index':'station_id'})
	#warm_df=pd.DataFrame(input_dict['warm'],index=['weekly_ratio']).T.reset_index().rename(columns={'index':'station_id'}) 
	#warm_dry_df=pd.DataFrame(input_dict['warm_dry'],index=['weekly_ratio']).T.reset_index().rename(columns={'index':'station_id'})
	gdf = gpd.GeoDataFrame(input_pts_data, geometry=gpd.points_from_xy(input_pts_data.lon, input_pts_data.lat)) 
	#print(type(input_dict['dry']))
	#print(input_dict)
	#print(type(gdf['site_num'].iloc[0]))

	gdf['dry'] = gdf['site_num'].map(reformat_dict(input_dict['dry'])) 
	gdf['warm'] = gdf['site_num'].map(reformat_dict(input_dict['warm']))
	gdf['warm_dry'] = gdf['site_num'].map(reformat_dict(input_dict['warm_dry']))
	type_list = ['dry','warm','warm_dry']
	print('gdf is: ')
	print(gdf.dry)
	print(gdf.warm_dry)
	#print(gdf)countries_gdf = geopandas.read_file("package.gpkg", layer='countries')
	#get background shapefiles
	hucs=gpd.read_file(huc_shapefile)
	hucs['coords'] = hucs['geometry'].apply(lambda x: x.representative_point().coords[:]) #add label column to gpd
	hucs['coords'] = [coords[0] for coords in hucs['coords']]

	pnw = gpd.read_file(pnw_shapefile)
	us_bounds = gpd.read_file(us_boundary)
	#df["B"] = df["A"].map(equiv)
	fig, ax = plt.subplots(1, 3,figsize=(18,18))
	#ax = ax.flatten()
	for x in range(3):  
		#divider = make_axes_locatable(ax[x])
		#cax = divider.append_axes("right", size="5%", pad=0.1)
		#cmap = 'Reds'#colors.ListedColormap(['b','g','y','r'])
		#bounds=[0,.25,.5,.75,1]
		#cmap = colors.ListedColormap(['b','g','y','r'])#
		#norm = colors.BoundaryNorm(bounds, cmap)
		divider = make_axes_locatable(ax[x])
		cax = divider.append_axes('right', size='5%', pad=0.05)
		hucs.plot(ax=ax[x],color='lightgray', edgecolor='black')
		pcm=gdf.plot(column=type_list[x],ax=ax[x],legend=True,cax=cax,cmap='Reds',vmin=0,vmax=1)#,norm=norm)
		#fig.colorbar(pcm, cax=cax, orientation='vertical')


		#divider = make_axes_locatable(ax[x])
		#cax = divider.append_axes("right", size="5%", pad=0.1)
		#cbar=fig.colorbar(pcm,cax=cax)
		#ax[x].set_clim(vmin=0, vmax=1)

		if '_' in type_list[x]: 
			drought_type=" ".join(type_list[x].split("_")).capitalize()
			ax[x].set_title(f'Proportion of weeks classified as {drought_type} snow drought \n {year_of_interest} water year')
		else: 
			ax[x].set_title(f'{type_list[x].capitalize()} snow drought \n {year_of_interest} water year')
		for idx, row in hucs.iterrows():
			ax[x].annotate(s=row['huc4'], xy=row['coords'],horizontalalignment='center')
		#add a context map
		axins = inset_axes(ax[0], width="30%", height="40%", loc=4)#,bbox_to_anchor=(.1, .5, .5, .5),bbox_transform=ax[x].transAxes)
		axins.tick_params(labelleft=False, labelbottom=False)
		us_bounds.plot(ax=axins,color='darkgray', edgecolor='black')
		hucs.plot(ax=axins,color='red', edgecolor='black')

	plt.tight_layout()
	plt.show()
	plt.close('all')