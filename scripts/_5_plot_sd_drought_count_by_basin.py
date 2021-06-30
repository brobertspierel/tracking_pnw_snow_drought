
import geopandas as gpd
import json 
import matplotlib.pyplot as plt  
import seaborn as sns 
import remote_sensing_functions as rs_funcs
import _3_obtain_all_data as obtain_data
import _4bv1_calculate_long_term_sp as _4b_rs 
import re
import math 
from scipy import stats
from functools import reduce
import sys 
import statsmodels.api as sa
import glob 
import scikit_posthocs as sp
import pandas as pd 
import geopandas as gpd 
import numpy as np 
import os 
import _pickle as cPickle
# from _4_process_rs_data import generate_output,combine_rs_snotel_annually,aggregate_dfs,merge_dfs,split_basins,combine_sar_data
import _4_process_rs_data as _4_rs
import matplotlib
from _1_calculate_snow_droughts_mult_sources import FormatData,CalcSnowDroughts
from sklearn.preprocessing import MinMaxScaler
from _4_plot_rs_sd_comparison import condense_rs_data,format_snotel_data,add_drought_cols_to_df
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors


def main(sp_data,sca_data,pickles,season,index,data_type,output_dir,daymet_dir,agg_step=12,huc_level='8',resolution=500,**kwargs):
	"""
	Link the datatypes together and add summary stats. 
	"""
	# #read in some modis/viirs data
	# rs_early=condense_rs_data(FormatData(glob.glob(sca_data+'*_12_huc8_no_forest_thresh.csv'),drop_cols=['system:index','.geo']).read_in_csvs())
	# rs_mid=condense_rs_data(FormatData(glob.glob(sca_data+'*_2_huc8_no_forest_thresh.csv'),drop_cols=['system:index','.geo']).read_in_csvs())
	# rs_late=condense_rs_data(FormatData(glob.glob(sca_data+'*_4_huc8_no_forest_thresh.csv'),drop_cols=['system:index','.geo']).read_in_csvs())

	#read in the daymet data 
	early=FormatData(glob.glob(daymet_dir+'*_12_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	mid=FormatData(glob.glob(daymet_dir+'*_2_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	late=FormatData(glob.glob(daymet_dir+'*_4_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	
	################################################################
	#next get the snotel data
	snotel_data=format_snotel_data(pickles,**kwargs)

	period_list = []
	snotel_periods=[]
	daymet_periods=[]
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
		print(f'Doing the {p1} period')
		#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(snotel_data)
		
		#make a rs chunk of the data- will be one df with all years and the full winter 
		rs_chunk = FormatData(glob.glob(sca_data+'*.csv'),drop_cols=['system:index','.geo']).read_in_csvs()
		#split that df into the season to match other data 
		rs_chunk = condense_rs_data(FormatData(None,time_period=p1).split_yearly_data(rs_chunk))

		# print('chunk rs')
		# print(rs_chunk)

		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991).calculate_snow_droughts()
			#print('snotel')
			#print(snotel_drought)
		else: 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG').calculate_snow_droughts()
		
		#get cols of interest 
		snotel_drought=snotel_drought[['huc8','year','dry','warm','warm_dry']]
		#rename cols so they don't get confused when data are merged 
		snotel_drought.columns=['huc8','year']+['s_'+column for column in snotel_drought.columns if not (column =='huc8') | (column=='year')]
		
		#then do the same for daymet  
		if (p1 == 'mid') | (p1 == 'late'): 
			daymet_drought=CalcSnowDroughts(p2,start_year=1991).calculate_snow_droughts()
		else: 
			daymet_drought=CalcSnowDroughts(p2).calculate_snow_droughts()
		#print('daymet',daymet_drought)
		daymet_drought=daymet_drought[['huc8','year','dry','warm','warm_dry']]
		
		daymet_drought.columns=['huc8','year']+['d_'+column for column in daymet_drought.columns if not (column =='huc8') | (column=='year')]

		#join the snotel, daymet and rs data and add a few cols for plotting 
		snotel_drought=add_drought_cols_to_df(snotel_drought,rs_chunk)
		daymet_drought=add_drought_cols_to_df(daymet_drought,rs_chunk)

		print(snotel_drought)
		#merge the two datasets into one df 
		#dfs = snotel_drought.merge(daymet_drought,on=['huc8','year'],how='inner')
		snotel_periods.append(snotel_drought)
		daymet_periods.append(daymet_drought)
	

	cols = ['rs_dry','rs_warm','rs_warm_dry']
		
	#print(snotel_periods[0].groupby('huc8')[cols].count()).reset_index()
	#print(type(snotel_periods[0][cols].count()))		
	# print(snotel_periods)
	# print(daymet_periods)

	us_bounds = gpd.read_file(kwargs.get('us_boundary'))
	pnw_states = gpd.read_file(kwargs.get("pnw_shapefile"))
	canada = gpd.read_file(kwargs.get("canada"))
	

	nrow=3
	ncol=3

	s_colors = ['#ccc596','#e4b047','#D95F0E','#666666']
	d_colors = ['#d4cfd9','#95aac5','#267eab','#666666']

	#define the overall figure- numbers are hardcoded
	fig,axs = plt.subplots(nrow,ncol,sharex=True,sharey=True,gridspec_kw={'wspace':0,'hspace':0,
                                    'top':0.95, 'bottom':0.05, 'left':0.05, 'right':0.95},
                       figsize=(nrow*2,ncol*2))  # <- adjust figsize but keep ratio n/m#constrained_layout=True,gridspec_kw=dict(wspace=0.0, hspace=0.0))#figsize=(ncol + 1, nrow + 1),gridspec_kw=dict(wspace=0.0, hspace=0.0))
				
	#iterate down rows (each row is a time period)
	for x in range(3): 
		
		#iterate through cols (across a row) with each col being a drought type. 
		plot_cols=['dry_colors','warm_colors','warm_dry_colors']
		xlabels=['Dry', 'Warm', 'Warm/dry']
		ylabels=['Early','Mid','Late']

		hucs_shp = gpd.read_file(kwargs.get('huc_shapefile'))
		hucs_shp['huc8'] = hucs_shp['huc8'].astype('int32')
		df = snotel_periods[x].groupby('huc8')[cols].count().reset_index()
		#join the count data to the shapefile for plotting 
		hucs_shp = hucs_shp.merge(df,how='inner',on='huc8')

		for y in range(3): 
			print(f'row is: {x} and col is {y}')
			bounds = [0,3,6,9,12]
			cmap = ListedColormap(['#dedede','#a3a3a3','#525252','#0b0b0b'])
			#cmap = ListedColormap(palette)
			#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True,figsize=(18,14))
			minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds

			# print('shapefile')
			# print(hucs_shp.columns)
			# print(hucs_shp[cols].max())
			# print(hucs_shp[cols])
			#plot dry snow drought
			#ax=plt.subplot(gs[x,y])
			canada.plot(ax=axs[x][y],color='#f2f2f2', edgecolor='darkgray',lw=0.5)
			us_bounds.plot(ax=axs[x][y],color='#f2f2f2', edgecolor='black',lw=0.5)
			pnw_states.plot(ax=axs[x][y],color='#f2f2f2',edgecolor='darkgray',lw=0.5)
			#hucs_shp.plot(ax=ax1,color='gray',edgecolor='darkgray')
			if not (x==nrow-1) & (y==ncol-1): 
				hucs_shp.plot(ax=axs[x][y],column=cols[y],cmap=cmap,vmin=0,vmax=12)#, column='Value1')
			else: 
				# divider = make_axes_locatable(axs[x][y])
				# cax = divider.append_axes('right', size='5%', pad=0.05)
				hucs_shp.plot(ax=axs[x][y],column=cols[y],cmap=cmap,vmin=0,vmax=12)#, column='Value1')
			#axs[x][y].set_title('Dry snow drought')
			axs[x][y].set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
			axs[x][y].set_ylim(miny - 1, maxy + 1)
			axs[0][y].set_title(xlabels[y])
			axs[x][0].set_ylabel(ylabels[x])
			#del hucs_shp
		hucs_shp.drop(columns=cols,inplace=True)
	# bounds = [0, 2, 4, 6,8,10,12]
	# cmap = ListedColormap('winter')
	#print(cmap)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	#print(norm)

			#pcm=df.plot(column=type_list[x],ax=ax[x],legend=True,cax=cax,cmap=palette,vmin=0,vmax=1,edgecolor='black',markersize=45)#,norm=norm)

	# im = plt.gca().get_children()[-1]
	fig.subplots_adjust(right=0.6)
	cbar_ax = fig.add_axes([0.94, 0.2, 0.02, 0.7])
	cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cbar_ax)#mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cbar_ax)

	#cbar = fig.colorbar(im, cax=cb_ax)

 	#set the colorbar ticks and tick labels
	# cbar.set_ticks(np.arange(bounds))
	# cbar.set_ticklabels(bounds)

	# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
	# plt.colorbar(cax=cax)
	plt.show()
	plt.close('all')


if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)		
		#construct variables from param file
		sp_data = variables['sp_data']
		sca_data = variables['sca_data']
		pickles = variables['pickles']
		season = variables['season']
		palette = variables['palette'] #"no_drought":"#cbbdb1",
		hucs_data = variables['hucs_data']
		huc_shapefile = variables['huc_shapefile']
		stations=variables['stations']
		daymet_dir=variables['daymet_dir']
		pnw_shapefile = variables["pnw_shapefile"]
		huc_shapefile = variables['huc_shapefile']
		us_boundary = variables['us_boundary']
		output_dir=variables['output_dir']
		canada=variables['canada']
	hucs=pd.read_csv(stations)

	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	#example function call for just optical data 
	#main(sp_data,sca_data,pickles,season,index=2,data_type='SAR',output_dir=pickles) #note that index can be 0-2 for SCA and only 0 for SP 

	#example call for SAR data included
	main(sp_data,sca_data,pickles,season,-9999,daymet_dir=daymet_dir,data_type='SCA',
	output_dir=pickles,hucs_data=hucs_data,huc_shapefile=huc_shapefile,hucs=hucs_dict,palette=palette,us_boundary=us_boundary,canada=canada,pnw_shapefile=pnw_shapefile)#,sar_data=sentinel_csv_dir) #note that index can be 0-2 for SCA and only 0 for SP 
