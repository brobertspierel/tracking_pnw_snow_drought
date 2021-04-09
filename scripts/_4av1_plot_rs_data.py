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
import _3_obtain_all_data as obtain_data
import re
import math 
from scipy import stats
from functools import reduce
import _pickle as cPickle
#import _4_process_rs_data as _4_rs_process
from _4_process_rs_data import generate_output,combine_rs_snotel_annually,aggregate_dfs,merge_dfs,split_basins,combine_sar_data
import _4_process_rs_data as _4_rs


def plot_sp_sca(dry,warm,warm_dry,total,ylabel,palette,nrows,ncols,fig_dir,stats_output=None,sp=False,show=True): 

	#plot the SCA data 
	font = {'family' : 'normal',
	'weight' : 'normal',
	'size'   : 16}
	
	plt.rc('font', **font)
	#if ncols > 1: 
	fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex='col',sharey='row',squeeze=False,figsize=(13,8)) #2,3 for nrows,ncols for sca 
	# else: 
	# 	fig,axes = plt.subplots(nrows,sharex=True,sharey='row') #2,3 for nrows,ncols for sca 
	width = 0.8
	for i in range(nrows): 
		for j in range(ncols): 
			print('indices ',i,' ',j)
			
			dfs = [dry,warm,warm_dry]
			types = ['dry_','warm_','warm_dry_']
			west_dfs = [aggregate_dfs(df,j,'west',drought_type,sp) for df,drought_type in zip(dfs,types)]
			east_dfs = [aggregate_dfs(df,j,'east',drought_type,sp) for df,drought_type in zip(dfs,types)]

			total_west = aggregate_dfs(total,j,'west','',sp)
			total_east = aggregate_dfs(total,j,'east','',sp)
			
			west = reduce(lambda x,y: pd.merge(x,y, on='year', how='outer'), west_dfs).sort_values('year')
			east = reduce(lambda x,y: pd.merge(x,y, on='year', how='outer'), east_dfs).sort_values('year') #[dry_east, warm_east, warm_dry_east]
			#print(west)
			#print(east)
			# west = get_anom_col(west,'NDSI_Snow_Cover')
			# east = get_anom_col(east,'NDSI_Snow_Cover')

			#get stats outputs as input 
			if stats_output: 
				if sp: 
					stats_file = glob.glob(stats_output+f'*SP_time_{j}_results.pickle')[0]
					print('processing sp')
				else: 
					stats_file = glob.glob(stats_output+f'*SCA_time_{j}_results.pickle')[0]	
				
				with open(stats_file, "rb") as input_file:
					stats_input = cPickle.load(input_file)

			if i < 1: #plot the west region on the top row 
				axes[i][j].set_axisbelow(True)
				#plot it 
				west.plot.bar(x='year',ax=axes[i][j],legend=False,color=palette.values(),width=width,stacked=True)#,grid={'color':'gray','linestyle':'-','axis':'y'}) #grid(color='r', linestyle='-', linewidth=2)
				#add grid lines to the y axis 
				axes[i][j].yaxis.grid(color='gray', linestyle='dashed',alpha=0.5)
				#add the sig levels 
				west['stats_input'] = west['year'].map(stats_input['west'])
				
				#get info about the bars to place the sig levels below 
				rects = axes[i][j].patches
		
				for rect,sig in zip(rects,list(west['stats_input'])):
					if sig > 0.0: 
						height = rect.get_height()
						axes[i][j].text(rect.get_x() + rect.get_width() / 2, -.01, '*',
						ha='center',va='top')

				#print the total on another axis 
				ax1=axes[i][j].twiny()
				total_west.plot.line(x='year',y='NDSI_Snow_Cover',ax=ax1,color='black',linewidth=2,linestyle='--',legend=False)#axes[i][j])
				ax1.xaxis.get_major_formatter().set_useOffset(False)
				# remove upper axis ticklabels
				ax1.set_xticklabels([])
				# set the limits of the upper axis to match the lower axis ones
				ax1.set_xlim(west['year'].min(),west['year'].max())
				ax1.set_xlabel(' ')

				
			else: #plot the east region on the bottom row 
				axes[i][j].set_axisbelow(True)
				east.plot.bar(x='year',ax=axes[i][j],legend=False,color=palette.values(),width=width,stacked=True)
				axes[i][j].yaxis.grid(color='gray', linestyle='dashed',alpha=0.5)
				east['stats_input'] = east['year'].map(stats_input['east'])
				print(east)
				#get info about the bars to place the sig levels below 
				rects = axes[i][j].patches
		
				for rect,sig in zip(rects,list(east['stats_input'])):
					if sig > 0.0: 
						height = rect.get_height()
						axes[i][j].text(rect.get_x() + rect.get_width() / 2, -.01, '*',
						ha='center',va='top')
					
				ax2=axes[i][j].twiny()
				total_east.plot.line(x='year',y='NDSI_Snow_Cover',ax=ax2,color='black',linewidth=2,linestyle='--',legend=False)#axes[i][j])
				ax2.xaxis.get_major_formatter().set_useOffset(False)
				# remove upper axis ticklabels
				ax2.set_xticklabels([])
				# set the limits of the upper axis to match the lower axis ones
				ax2.set_xlim(east['year'].min(),east['year'].max())
				ax2.set_xlabel(' ')
			#set titles and legend
			axes[i][j].set_xlabel(' ')
			
			if ylabel: 
				axes[i][j].set_ylabel(ylabel)
			
	palette = {k.title():v for k,v in palette.items()}

	if not sp: 
		axes[0][2].legend(palette)
		axes[0][0].set_title('Nov-Dec')
		axes[0][1].set_title('Jan-Feb')
		axes[0][2].set_title('March-April')
	else: 
		axes[0][0].set_title('Dec-Feb')
		axes[0][0].legend(palette)

	#add region labels 
	axes[0][0].annotate('a)',xy=(0.05,0.85),xycoords='axes fraction')
	axes[1][0].annotate('b)',xy=(0.05,0.85),xycoords='axes fraction')
	
	if show: 
		plt.show()
		plt.close('all')
	else:
		if not sp:  
			plt.savefig(fig_dir+'SCA_time_series_by_drought_max_extent_w_sig_notation.png',dpi=350)
		else: 
			plt.savefig(fig_dir+'SP_time_series_by_drought_min_median_w_sig_notation.png',dpi=350)
	return west,east 

def main(sp_data,sca_data,pickles,season,palette,sar_data,hucs_data,year_of_interest,huc_level='8',agg_step=12,resolution=500,**kwargs):
	"""
	Plot the long term snow drought types and trends. 
	"""
	#plot 2001-2020 optical data 
	#################################################################
	# if not 'year_of_interest' in kwargs: 
	print('doing this one')
	print(hucs_data)
	print(pd.read_csv(hucs_data))
	# dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='dry',sp=True),sp=True)
	# warm_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm',sp=True),sp=True)
	# warm_dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm_dry',sp=True),sp=True)
	# total_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,sp=True,total=True),sp=True)


	# dry=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='dry'))
	# warm=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm'))
	# warm_dry=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm_dry')) 
	# total = generate_output(combine_rs_snotel_annually(sca_data,season,pickles,total=True)) 
	# print('example')
	# print(dry_sp)
	#sca_plot=plot_sp_sca(dry,warm,warm_dry,total,'SCA max extent',palette,2,3,fig_dir=kwargs.get('fig_dir'),stats_output=kwargs.get('stats_output')) #dry,warm,warm_dry,total,ylabel,palette,nrows,ncols,stats_output=None,sp=False,show=True,**kwargs
	
	# 	sp_plot=plot_sp_sca(dry_sp,warm_sp,warm_dry_sp,total_sp,'Snow persistence',palette,2,1,fig_dir=kwargs.get('fig_dir'),stats_output=kwargs.get('stats_output'),sp=True)

	#################################################################

	#plot specific years for WSCA 
	#################################################################
#else: 
	# print(type(kwargs.get('year_of_interest')))
	# print(season)
	# print(agg_step)
	# print(f'short_term_snow_drought_{2020}_water_year_{season}_{agg_step}_day_time_step_w_all_dates')
	# print(pickles)
	# snotel_data = pickles+f'short_term_snow_drought_{year_of_interest}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
	
	# #instantiate the acquireData class and read in snotel, sentinel and modis/viirs data 
	# input_data = obtain_data.AcquireData(sar_data,sca_data,snotel_data,hucs_data,huc_level,resolution)
	# short_term_snow_drought = input_data.get_snotel_data()
	# sar_data = input_data.get_sentinel_data('filter')
	# optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
	
	# dry = _4_rs.split_dfs_within_winter_season(merge_dfs(short_term_snow_drought,optical_data,'dry',sar_data=sar_data),None)
	# warm = _4_rs.split_dfs_within_winter_season(merge_dfs(short_term_snow_drought,optical_data,'warm',sar_data=sar_data),None)
	# warm_dry = _4_rs.split_dfs_within_winter_season(merge_dfs(short_term_snow_drought,optical_data,'warm_dry',sar_data=sar_data),None)
	# total = _4_rs.split_dfs_within_winter_season(merge_dfs(short_term_snow_drought,optical_data,'total',sar_data=sar_data),None)

	# early = combine_sar_data(dry,warm,warm_dry,total,0)
	# mid = combine_sar_data(dry,warm,warm_dry,total,1)
	# late = combine_sar_data(dry,warm,warm_dry,total,2)
	
	#print(early)

	#################################################################

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
		stats = variables['stats']
		fig_dir = variables['fig_dir']
		optical_csv_dir = variables['optical_csv_dir']
		sentinel_csv_dir = variables['sentinel_csv_dir']
		hucs_data = variables['hucs_data']
		resolution=variables['resolution']
		year_of_interest = variables['year_of_interest']
		#set a few script specific user params
		# plotting_param = 'Snow persistence' #'SCA percent of total'
		# #plot_func = 'quartile'
		# elev_stat = 'elev_mean'
	#call main func for plotting optical data
	#main(sp_data,sca_data,pickles,season,palette,stats_output=stats,fig_dir=fig_dir)

	#call main func for plotting sar and optical data 
	main(sp_data,sca_data,pickles,season,palette,sar_data=sentinel_csv_dir,year_of_interest=year_of_interest,hucs_data=hucs_data,resolution=resolution,stats_output=stats,fig_dir=fig_dir) 