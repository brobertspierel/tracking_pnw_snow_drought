import pandas as pd 
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt 
import geopandas as gpd 
import json 
import glob
import pickle 
from functools import reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib as mpl

"""Make a plot(s) of the number of snow droughts in different decades by drought type. 

Inputs: this relies on the csvs that are the output (in addition to the figure) of the 
_2_plot_sd_counts_point_wise_comparison or the _2_plot_sd_counts_comparison scripts. These are 
effectively the same except one is for calculating snow droughts at the basin scale and one at the 
point scale. 

"""

def get_decades(df,start_year,end_year): 
	"""Subset a df for 1991-2000 water years"""
	return df.loc[(df['year'] >= start_year) & (df['year'] <= end_year)]

def split_source(df,source): 
	return df[[col for col in df.columns if col.startswith(source)]]

def split_by_sd_type(df,sd_type):
	"""Take a decade df that has all the time periods and sd types and split them apart."""
	if (sd_type == 'Dry') | (sd_type == 'Warm'): 
		return df[[col for col in df.columns if (sd_type in col) & ~('/' in col)]] #not sure if that will work right 
	elif sd_type == 'Warm/dry': 
		return df[[col for col in df.columns if sd_type in col]] 

def split_seasonal_window(df,window): 
	"""Split the cols of a df by seasonal window labeled in the col header."""
	return df[[col for col in df.columns if window in col]]

def create_dfs_by_sd_and_window(df):
	s_output = []
	ua_output = []

	decades = [(1991,2000),(2001,2010),(2011,2020)]
	# s_output = {}
	# d_output = {}
	for item in decades: 
		#gets decades but still snotel/daymet combined
		subset=get_decades(df,item[0],item[1])
		#remove the year col for summing 
		subset = subset.iloc[:,1:]
		#split the df into daymet and snotel 
		s = split_source(subset, 's')
		ua = split_source(subset, 'ua')
		#do some formatting of col names to facilitate slotting into existing script 
		s.columns = [col[2:] for col in s.columns] #hardcoded to remove the s or d from the column names 
		ua.columns = [col[2:] for col in ua.columns]
		#get the decadal sum 
		s = pd.DataFrame(s.sum()).T
		ua = pd.DataFrame(ua.sum()).T
		s.columns = [col.replace('_',' ') for col in s.columns]
		ua.columns = [col.replace('_',' ') for col in ua.columns]
		#add the decade identifier to the df 
		s.columns = [col + f' {item[1]}' for col in s.columns]
		ua.columns = [col + f' {item[1]}' for col in ua.columns]
		#now split the df up by the temporal window 
		s_season = []
		ua_season = []
		for i in ['Early','Mid','Late']: 
			s_sub=split_seasonal_window(s,i)
			ua_sub=split_seasonal_window(ua,i)
			#remove the seasonal window identifiers so they will concat by row 
			s_sub.columns = [col.replace(i+' ','') for col in s_sub.columns]
			ua_sub.columns = [col.replace(i+' ','') for col in ua_sub.columns]
			s_season.append(s_sub)
			ua_season.append(ua_sub)
		#create a decade of data with sd types and windows as index 
		s_output.append(pd.concat(s_season))
		ua_output.append(pd.concat(ua_season))

	return pd.concat(s_output), pd.concat(ua_output)

def create_summary_dfs_by_sd(df): 
	s_dry = []
	s_warm = []
	s_wd = []
	ua_dry = []
	ua_warm = []
	ua_wd = []

	decades = [(1991,2000),(2001,2010),(2011,2020)]
	s_output = {}
	ua_output = {}
	for item in decades: 
		#gets decades but still snotel/daymet combined
		subset=get_decades(df,item[0],item[1])
		#remove the year col for summing 
		subset = subset.iloc[:,1:]
		#split the df into daymet and snotel 
		s = split_source(subset, 's')
		ua = split_source(subset, 'ua')
		#first get the sum by drought type for snotel
		s_dry.append(split_by_sd_type(s,'Dry').sum().sum())
		s_warm.append(split_by_sd_type(s,'Warm').sum().sum())
		s_wd.append(split_by_sd_type(s,'Warm/dry').sum().sum())
		#then sum by drought type for daymet 
		ua_dry.append(split_by_sd_type(ua,'Dry').sum().sum())
		ua_warm.append(split_by_sd_type(ua,'Warm').sum().sum())
		ua_wd.append(split_by_sd_type(ua,'Warm/dry').sum().sum())
	#put the lists into a dict to make a df 
	s_output.update({'Dry':s_dry,'Warm':s_warm,'Warm/dry':s_wd})
	ua_output.update({'Dry':ua_dry,'Warm':ua_warm,'Warm/dry':ua_wd})
	#now convert dicts to dfs
	s_df = pd.DataFrame().from_dict(s_output)
	s_df['decade']=[i[1] for i in decades] #add the last year of each decade 
	ua_df = pd.DataFrame().from_dict(ua_output)
	ua_df['decade']=[i[1] for i in decades]

	return s_df, ua_df

def plot_decadal_trends_by_sd_type(s,ua,fig_output,spatial): 
	"""Create plots of huc8, huc6 and pts mode trajectories by sd type."""
	s_colors = ['#ccc596','#e4b047','#D95F0E','#666666','#000000']
	ua_colors = ['#d4cfd9','#3333FF','#133F55','#666666','#000000']
	colors = ['#c0c0c0','#787878','#3b3b3b','#000000']
	fig,(ax1,ax2) = plt.subplots(1,2, figsize=(8,6), 
							sharex=True,
							sharey=True,
							gridspec_kw={'wspace':0,'hspace':0})
	
	s['total'] = s.sum(axis=1)
	ua['total'] = ua.sum(axis=1)
	print('s is: ')
	print(s)

	for x,sd in zip(range(4),['Dry','Warm','Warm/dry','total']): 
		if x <=2: 
			#changed 10/25/2021 to just use the same grayscale for both 
			s.plot(x='decade',y=sd,ax=ax1,c=colors[x],linewidth=3,linestyle='--')
			ua.plot(x='decade',y=sd,ax=ax2,c=colors[x],linewidth=3,linestyle='-')
			ax1.grid(axis='y',alpha=0.25)
			ax2.grid(axis='y',alpha=0.25)
			ax1.set_xticks(s.decade)
			ax2.set_xticks(ua.decade)
			ax1.set_xticklabels(ax1.get_xticks(),rotation=45)
			ax2.set_xticklabels(ax2.get_xticks(),rotation=45) 
			ax1.set_ylabel('Snow drought counts')
			#ax2.yaxis.set_visible(False)
			#ax2.set_yticks(color='w')
			ax1.set_xlabel('')
			ax2.set_xlabel('')
			ax1.annotate(f'{chr(97+0)}',xy=(0.1,0.9),xycoords='axes fraction',fontsize=12,weight='bold')#f'{chr(97)}'
			ax2.annotate(f'{chr(97+1)}',xy=(0.1,0.9),xycoords='axes fraction',fontsize=12,weight='bold')#f'{chr(97)}'
		else: 
			ax1 = ax1.twinx()
			ax2 = ax2.twinx()
			s.plot(x='decade',y=sd,ax=ax1,c=colors[x],linewidth=3,linestyle='--')
			ua.plot(x='decade',y=sd,ax=ax2,c=colors[x],linewidth=3,linestyle='-')
			ax1.grid(axis='y',alpha=0.25)
			ax2.grid(axis='y',alpha=0.25)
			ax1.set_xticks(s.decade)
			ax2.set_xticks(ua.decade)
			ax1.set_xticklabels(ax1.get_xticks(),rotation=45)
			ax2.set_xticklabels(ax2.get_xticks(),rotation=45) 
			ax1.set_ylabel('Snow drought counts')
			#ax2.yaxis.set_visible(False)
			#ax2.set_yticks(color='w')
			ax1.set_xlabel('')
			ax2.set_xlabel('')
			ax1.annotate(f'{chr(97+0)}',xy=(0.1,0.9),xycoords='axes fraction',fontsize=12,weight='bold')#f'{chr(97)}'
			ax2.annotate(f'{chr(97+1)}',xy=(0.1,0.9),xycoords='axes fraction',fontsize=12,weight='bold')#f'{chr(97)}'

	fig_fn = os.path.join(fig_output,f'snotel_ua_swe_{spatial}_level_decadal_count_totals_full_season_w_delta_swe_proj_final1_w_all_droughts.jpg')
	if not os.path.exists(fig_fn): 
		plt.savefig(fig_fn,
			dpi=500, 
			bbox_inches = 'tight',
	    	pad_inches = 0.1
	    	)
	else: 
		print(f'figure {fig_fn} created')
		# plt.show()
		# plt.close()

def export_stats(df,stats_output,spatial,dataset): 
	"""Read in a df and write out a csv to specified dir."""
	out_fn = os.path.join(stats_output,f'{dataset}_{spatial}_level_decadal_stats_by_drought_for_seasonal_window_fig_proj_draft1.csv')
	if not os.path.exists(out_fn): 
		df.to_csv(out_fn)
	else: 
		print(f'The file {out_fn} already exists, skipping')

def main(input_csvs,fig_output,stats_output): 
	#use this chunk of code when you have all the hucs/pts in the same directory 
	# csvs = glob.glob(input_dir+'*annual_counts.csv') #hardcoded for amended filenames, change to run output of _2_plot... directly 
	# try: 
	# 	huc8 = pd.read_csv([c for c in csvs if 'HUC8' in c][0])
	# except IndexError: 
	# 	print('Check the huc8 file')
	# try: 
	# 	huc6 = pd.read_csv([c for c in csvs if 'HUC6' in c][0])
	# except IndexError: 
	# 	print('Check the huc6 file')
	# try: 	
	# 	pts = pd.read_csv([c for c in csvs if 'pt_mode' in c][0])
	# except IndexError: 
	# 	print('Check the pts file')

	####################################################
	#use for one file 
	huc6 = pd.read_csv(input_csvs)
	#print(huc8)
	####################################################
	#create summary of outputs (dfs) of occurrences by sd type, seasonal window and decade 
	#this will be the input to the plotting script that was previously making the peak decade plots 
	#and is now just plotting things by decade, snow drought type and seasonal window 
	# s,ua = create_dfs_by_sd_and_window(pts)
	# #when they are concatenated there are nans introduced, remove those 
	# s = s.apply(lambda x: pd.Series(x.dropna().values).astype(int))
	# #remove the / from the col names 
	# s.columns = [col.replace('/', ' ') for col in s.columns]
	# #add a column with the periods/windows 
	# s['Snotel'] = ['Early','Mid','Late']
	# s.index = s.Snotel
	# print(s)
	# s.drop(columns=['Snotel'],inplace=True)
	# #do the same for daymet 
	# ua = ua.apply(lambda x: pd.Series(x.dropna().values).astype(int))
	# ua.columns = [col.replace('/', ' ') for col in ua.columns]
	# ua['ua_swe'] = ['Early','Mid','Late']
	# ua.index = ua.ua_swe
	# ua.drop(columns=['ua_swe'],inplace=True)

	
	# export_stats(s,stats_output,'pts','snotel')
	# export_stats(ua,stats_output,'pts','ua_swe')

	###########################################################
	#create the output dfs for summary of occurrences by sd type and decade 
	#s_huc8, ua_huc8 = create_summary_dfs_by_sd(huc8)
	s_huc6, ua_huc6 = create_summary_dfs_by_sd(huc6)
	#s_pts, ua_pts = create_summary_dfs_by_sd(pts)

	# currently set up so that you need to change the params each time. This could be added to a for loop but then 
	# would be more annoying to run just one. 
	plot_decadal_trends_by_sd_type(s_huc6, ua_huc6, fig_output, 'huc6')
	# export_stats(s_huc6,stats_output,'huc6','snotel')
	# export_stats(ua_huc6,stats_output,'huc6','ua_swe')
	
	#plot_decadal_trends_by_sd_type(s_pts, ua_pts, fig_output, 'pts')
	# export_stats(s_pts,stats_output,'pts','snotel')
	# export_stats(ua_pts,stats_output,'pts','ua_swe')

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		input_dir=variables['stats']
		fig_dir = variables['fig_dir']
		stats_dir = variables['stats_dir']
	
	main(input_dir, fig_dir, stats_dir)












