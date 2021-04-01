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
import re
import math 
from scipy import stats
from functools import reduce
import _pickle as cPickle
#class CleanOptialRSData(): 

western = ['1708','1801','1710','1711','1709']
eastern = ['1701','1702','1705','1703','1601','1707','1706','1712','1704']


def split_basins(input_df,grouping_col,**kwargs): 
	"""Read in a dir of landsat sp data from GEE and make into a new df."""
	input_df[grouping_col] = input_df[grouping_col].astype('str')
	west_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(western))]
	east_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(eastern))]
	
	#replace instances of inf with nan and drop the grouping_col so its not in the mean
	west_df.replace(np.inf,np.nan,inplace=True)
	east_df.replace(np.inf,np.nan,inplace=True)
	
	# try: #commented out 3/24/2021 might need to uncomment for plotting  
	# 	west_df.drop(columns=[grouping_col,'elev_mean'],inplace=True) #added the hardcoded drop of the elev col to clean up for plotting
	# 	east_df.drop(columns=[grouping_col,'elev_mean'],inplace=True)
	# except Exception as e: 
	# 	pass
		#print(e)
	# west_df['year'] = kwargs.get('year')
	# east_df['year'] = kwargs.get('year')
	# west_mean = west_df.median(axis=0)
	# east_mean = east_df.median(axis=0)

	return west_df,east_df

def split_dfs_within_winter_season(df,region,sp=False): 
	"""Splits a single df by date ranges in a winter season."""
	
	early_df = df.loc[(df['date'].dt.month>=11)] 
	mid_df = df.loc[(df['date'].dt.month>=1)&(df['date'].dt.month<=2)]
	late_df = df.loc[(df['date'].dt.month>=3)&(df['date'].dt.month<=4)]

	return {region:[early_df,mid_df,late_df]}


def merge_dfs(snotel_data,rs_data,drought_type,huc_level='8',col_of_interest='NDSI_Snow_Cover',elev_stat='elev_mean',**kwargs): #added drought_type arg so the drought type is supplied externally 3/15/2021
	"""Merge snotel snow drought data with RS data."""
	
	combined = _4a_rs.create_snow_drought_subset(snotel_data,drought_type,huc_level)
	#merge em 
	combined=combined.merge(rs_data, on=['date',f'huc{huc_level}'], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
	#get the rs data for the time periods of interest for a snow drought type 
	combined.rename(columns={col_of_interest:f'{drought_type}_{col_of_interest}'},inplace=True)
	#combined = combined.groupby([f'huc{huc_level}', 'date'])[f'{drought_type}_{col_of_interest}'].transform(max) #doesn't really matter which stat (max,min,first) because they are all the same 
	combined = combined.sort_values(f'{drought_type}_{col_of_interest}').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')

	#print(combined)
	#print(combined[['huc8','date',f'{drought_type}_{col_of_interest}']])
	#check if a couple of args are in kwargs, they can be anything that will evaluate to True
	if 'groupby' in kwargs: 
		rs_df = combined.groupby('date')[f'{drought_type}_{col_of_interest}'].sum().reset_index()
		#dry_rs = dry_combined.groupby('huc'+huc_level)[f'dry_{col_of_interest}',elev_stat].max().reset_index() #changed col from pct change to filter 2/1/2021

		if 'scale_it' in kwargs: 
			scaler = (combined[f'{drought_type}_{col_of_interest}'].count()/rs_data.shape[0])
			rs_df[f'{drought_type}_{col_of_interest}'] = rs_df[f'{drought_type}_{col_of_interest}']*scaler

		return rs_df

	else: 
		return combined

def combine_rs_snotel_annually(input_dir,season,pickles,agg_step=12,resolution=500,huc_level='8',col_of_interest='NDSI_Snow_Cover',elev_stat='elev_mean',sp=False,total=False,**kwargs): 
	"""Get RS data for snow drought time steps and return those data split by region."""
	
	west_dfs_list = []
	east_dfs_list = []
	years = []
	optical_files = sorted(glob.glob(input_dir+'*.csv'))

	for file in optical_files: 
		year = re.findall('(\d{4})', os.path.split(file)[1])[1] #gets a list with the start and end of the water year, take the second one. expects files to be formatted a specific way from GEE 

		#decide which season length to use depending on the RS aggregation type (SP or SCA)
		if 'SP' in file: 
			snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates_first_day_start'
		elif 'SCA' in file:
			snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
		else: 
			print('Your file contains neither sp nor SCA, try again')

		input_data = obtain_data.AcquireData(None,file,snotel_data,None,huc_level,resolution)
		
		short_term_snow_drought = input_data.get_snotel_data()
		optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
		optical_data[f'huc{huc_level}'] = pd.to_numeric(optical_data['huc'+huc_level]) 
		optical_data['date'] = _4a_rs.convert_date(optical_data,'date')

		#convert pixel counts to area
		if not sp: 
			optical_data=rs_funcs.convert_pixel_count_sq_km(optical_data,col_of_interest,resolution)

		#optical_data['year'] = optical_data['date'].dt.year

		if not total: 
			#combine the remote sensing and snotel data using the snotel dates of snow droughts to extract rs data 
			merged=merge_dfs(short_term_snow_drought,optical_data,kwargs.get('drought_type')) #snotel_data,rs_data,drought_type
		else: 
			pass
			#print('Calculating total with no snow droughts')
		#output = split_dfs_within_winter_season
		try: 
			split_dfs=split_basins(merged,f'huc{huc_level}',year=year) #returns the merged df split into two dfs, west (0) and east (1)
		
		except UnboundLocalError as e: 
			split_dfs=split_basins(optical_data,f'huc{huc_level}',year=year) #returns the merged df split into two dfs, west (0) and east (1)
		
		west_dfs_list.append(split_dfs[0])
		east_dfs_list.append(split_dfs[1])
		
	output_west_df = pd.concat(west_dfs_list,ignore_index=True)
	output_east_df = pd.concat(east_dfs_list,ignore_index=True)

	return output_west_df,output_east_df #returns two dfs, one for each region for all the years for one drought type 



def get_anom_col(input_df,base_col,skip_col='year'): #not currently in use 3/15/2021
	"""Helper function."""
	#long_term_mean=input_df[mean_col].mean()
	#input_df['mean'] = long_term_mean
	for column in input_df.columns: 
		if not (column == base_col) | (column == skip_col) | (column == 'mean'): 

			input_df[column] = (input_df[column]/input_df[base_col])#*100
	input_df.drop(columns=[base_col],inplace=True)
	return input_df

def generate_output(input_tuple,sp=False):
	output = {} 
	for i,j in zip(input_tuple,['west','east']): 
		if not sp: 
			chunk = split_dfs_within_winter_season(i,j)
			output.update(chunk)
		else: 
			output.update({j:[i]})
	return output

def aggregate_dfs(input_dict,index,region,drought_type,sp=False): 
	"""Helper function for plotting."""

	#get the sum of the dates in one period- this enables us to pick the time when snow extent is greatest across the AOI
	# print('input')
	# print(input_dict[region][index])
	if not sp: #function defaults to working on SCA, need to specify sp = True to run that 
		output = input_dict[region][index].groupby('date')[f'{drought_type}NDSI_Snow_Cover'].sum().reset_index()
		#print('output ',output)
		output['year'] = output['date'].dt.year	
		#get the date when snow extent is at its max for that period- maybe mean or median makes more sense? 
		output = output.groupby('year')[f'{drought_type}NDSI_Snow_Cover'].max().reset_index() #change if a different stat is introduced above

	else: 
		output = input_dict[region][index].groupby('date')[f'{drought_type}NDSI_Snow_Cover'].median().reset_index()
		output['year'] = output['date'].dt.year	
		output = output.groupby('year')[f'{drought_type}NDSI_Snow_Cover'].min().reset_index() #not sure about this stat- we're taking the median of dates and then the median of those? maybe max is better? 		

	
	#rename the col so the legend looks better
	output.rename(columns={f'{drought_type}NDSI_Snow_Cover':drought_type.title()})

	return output

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

def main(sp_data,sca_data,pickles,season,palette,huc_level='8',**kwargs):
	"""
	Plot the long term snow drought types and trends. 
	"""
	dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='dry',sp=True),sp=True)
	warm_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm',sp=True),sp=True)
	warm_dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm_dry',sp=True),sp=True)
	total_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,sp=True,total=True),sp=True)


	dry=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='dry'))
	warm=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm'))
	warm_dry=generate_output(combine_rs_snotel_annually(sca_data,season,pickles,drought_type='warm_dry')) 
	total = generate_output(combine_rs_snotel_annually(sca_data,season,pickles,total=True)) 
	
	#sca_plot=plot_sp_sca(dry,warm,warm_dry,total,'SCA max extent',palette,2,3,fig_dir=kwargs.get('fig_dir'),stats_output=kwargs.get('stats_output')) #dry,warm,warm_dry,total,ylabel,palette,nrows,ncols,stats_output=None,sp=False,show=True,**kwargs
	
	sp_plot=plot_sp_sca(dry_sp,warm_sp,warm_dry_sp,total_sp,'Snow persistence',palette,2,1,fig_dir=kwargs.get('fig_dir'),stats_output=kwargs.get('stats_output'),sp=True)

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
		#set a few script specific user params
		plotting_param = 'Snow persistence' #'SCA percent of total'
		#plot_func = 'quartile'
		elev_stat = 'elev_mean'

	main(sp_data,sca_data,pickles,season,palette,stats_output=stats,fig_dir=fig_dir)
