import sys 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

"""Make a plot of the temporal distribution of max drought types for basins and decades. 
Inputs: 
- this script depends on the csv files which are output by _3_plot_sd_by_decades_revised.py
	UPDATED 9/23/2021- this is not plotting the 'peak' snow drought but merely counts. This depends on outputs
	from _2_plot_sd_counts_comparison_new_SWE or equivalent. 
  Note that these outputs need to be summed and included in one csv. See example files below for formatting. 
-output dir, this can be passed as a commandline arg
"""

def fix_headers(df): 
	df.columns = df.columns.str.replace(' ','')
	df.columns = df.columns.str.replace('\n','_')
	df.columns = df.columns.str.replace('__','_')
	df.columns = df.columns.str.replace('/','_')
	try: 
		df.drop(columns=['Snotel'], inplace = True)
	except KeyError: 
		df.drop(columns=['UASWE'], inplace = True)

	return df 

def modify_headers(df, modifier): 
	df.columns = [f'{modifier}_{col}' for col in df.columns]
	return df 

def reformat_df(df,row): 
	df = df.iloc[row].to_frame().T
	return df.reindex(sorted(df.columns), axis=1)

def plot_format(df,modifier,row): 
	df = df.T.reset_index()
	try: 
		df.rename(columns={row:modifier},inplace=True)
	except KeyError: 
		print('Double check the values column for the header.')
	df['index'] = df['index'].str.replace('_', ' ')
	return df 

def make_decade_count_fig(ua_df,snotel_df,output_dir,sort_col='huc8'): 
	fontsize = 10
	fig, axs = plt.subplots(3,3,
				sharey=True,
				sharex='col',
				gridspec_kw={'wspace':0,'hspace':0},
				figsize=(8,6))

	early_df = plot_format(reformat_df(ua_df,0),'UASWE',0).merge(plot_format(reformat_df(snotel_df,0), 'SNOTEL', 0), on='index', how='inner')
	mid_df = plot_format(reformat_df(ua_df,1),'UASWE', 1).merge(plot_format(reformat_df(snotel_df,1), 'SNOTEL', 1), on='index', how='inner')
	late_df = plot_format(reformat_df(ua_df,2),'UASWE', 2).merge(plot_format(reformat_df(snotel_df,2), 'SNOTEL', 2), on='index', how='inner')

	print('early')
	print(early_df)
	dfs = [early_df,mid_df,late_df]
	pers = ['Early','Mid','Late']
	droughts = ['Dry', 'Warm2', 'Warmdry2'] #this is a dumb hack because Warm was getting lumped with Warmdry
	drought_label = ['Dry', 'Warm', 'Warm/dry']
	count = 0
	for row in range(3): 
		#get a df of the time of year, these are the rows 
		df = dfs[row]
		for col in range(3): 
			drought_df = df.loc[df['index'].str.startswith(droughts[col])]
			#add a col that will be used for labeling
			drought_df['label'] = drought_df['index'].str[-4:]
			#drought_df['label'] = drought_df['label'].astype(int)
			print(drought_df)
			print(drought_df.dtypes)
			if (row == 0) & (col == 2):
				legend = True
			else: 
				legend = False
			drought_df.plot(x='label',y='UASWE',ax=axs[row][col],legend=legend,c='black')
			drought_df.plot(x='label',y='SNOTEL',ax=axs[row][col],legend=legend,c='black',linestyle='--')
			#add grid lines
			axs[row][col].grid(axis='y',alpha=0.25)
			#add a letter to id the plot 
			axs[row][col].annotate(f'{chr(97+count)}',xy=(0.85,0.1),xycoords='axes fraction',fontsize=10,weight='bold')
			count +=1
			if row == 0: 
				axs[row][col].set_title(drought_label[col],fontsize=10)
			if col == 0: 
				axs[row][col].set_ylabel(pers[row],fontsize=10)
			# else: 
			# 	print('removing ticks')
			# 	axs[row][col].yaxis.set_ticks([])
			if row == 2: 
				axs[row][col].set_xlabel('')
				axs[row][col].tick_params(axis='x',rotation=45)

	#fig.text(0.5, 0.04, 'common X', ha='center')
	fig.text(0.025, 0.5, 'Snow drought count', va='center', rotation='vertical')

	# early_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax1, legend=False)
	# mid_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax2, legend=False)
	# late_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax3)
	# #add horizontal grid lines

	# for ax, per in zip([ax1,ax2,ax3],['Early','Mid','Late']): 
	# 	ax.tick_params(axis='x', labelsize=fontsize)
	# 	ax.tick_params(axis='y', labelsize=fontsize)
	# 	ax.set_xlabel(' ')
	# 	ax.grid(axis='y',alpha=0.25)
	# 	ax.annotate(per,xy=(0.75,0.9),xycoords='axes fraction',fontsize=fontsize)#f'{chr(97)}'
	
	# #do a couple of ax specific things 
	# ax1.set_ylabel('Number of basins',fontsize=fontsize)
	# ax3.legend(fontsize=12,loc='upper left')
	#plt.rcParams['legend.title_fontsize'] = fontsize


	# plt.tight_layout()
	# plt.show()
	# plt.close('all')
	output_fn = os.path.join(output_dir,f'{sort_col}_snotel_ua_swe_decades_sd_type_seasonal_window_counts_w_delta_swe_proj_lineplot_draft1.jpg')
	if not os.path.exists(output_fn): 
		plt.savefig(output_fn, 
			dpi=500, 
			bbox_inches = 'tight',
    	pad_inches = 0.1
			)

# def make_decade_count_fig(ua_df,snotel_df,output_dir,sort_col='huc8'): 
# 	fontsize = 10
# 	fig, (ax1,ax2,ax3) = plt.subplots(3,3, sharex=True,sharey=True,
# 				gridspec_kw={'wspace':0,'hspace':0},
# 				figsize=(8,6))

# 	early_df = plot_format(reformat_df(ua_df,0),'UASWE',0).merge(plot_format(reformat_df(snotel_df,0), 'SNOTEL', 0), on='index', how='inner')
# 	mid_df = plot_format(reformat_df(ua_df,1),'UASWE', 1).merge(plot_format(reformat_df(snotel_df,1), 'SNOTEL', 1), on='index', how='inner')
# 	late_df = plot_format(reformat_df(ua_df,2),'UASWE', 2).merge(plot_format(reformat_df(snotel_df,2), 'SNOTEL', 2), on='index', how='inner')

# 	print('early')
# 	print(early_df)

# 	early_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax1, legend=False)
# 	mid_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax2, legend=False)
# 	late_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax3)
# 	#add horizontal grid lines

# 	for ax, per in zip([ax1,ax2,ax3],['Early','Mid','Late']): 
# 		ax.tick_params(axis='x', labelsize=fontsize)
# 		ax.tick_params(axis='y', labelsize=fontsize)
# 		ax.set_xlabel(' ')
# 		ax.grid(axis='y',alpha=0.25)
# 		ax.annotate(per,xy=(0.75,0.9),xycoords='axes fraction',fontsize=fontsize)#f'{chr(97)}'
	
# 	#do a couple of ax specific things 
# 	ax1.set_ylabel('Number of basins',fontsize=fontsize)
# 	ax3.legend(fontsize=12,loc='upper left')
# 	#plt.rcParams['legend.title_fontsize'] = fontsize


# 	plt.tight_layout()
# 	plt.show()
# 	plt.close('all')
# 	# output_fn = os.path.join(output_dir,f'{sort_col}_snotel_ua_swe_decades_sd_type_seasonal_window_counts_w_delta_swe_proj_draft1.jpg')
# 	# if not os.path.exists(output_fn): 
# 	# 	plt.savefig(output_fn, 
# 	# 		dpi=500, 
# 	# 		bbox_inches = 'tight',
#  #    	pad_inches = 0.1
# 	# 		)

def main(output_dir): 

	#these are obviously hardcoded and could be changed in a future version 
	huc8_ua = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/final_stats/ua_swe/total_counts/ua_swe_huc8_level_decadal_stats_by_drought_type_w_delta_swe_proj_draft1.csv"))
	huc8_s = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/final_stats/ua_swe/total_counts/snotel_huc8_level_decadal_stats_by_drought_type_w_delta_swe_proj_draft1.csv"))
	# huc6_ua = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/final_stats/ua_swe/decades/ua_swe_huc6_level_decadal_stats_by_drought_for_next_fig_proj_draft1.csv"))
	# huc6_s = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/final_stats/ua_swe/decades/snotel_huc6_level_decadal_stats_by_drought_for_next_fig_proj_draft1.csv"))
	# pts_ua = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/final_stats/ua_swe/decades/ua_swe_pts_level_decadal_stats_by_drought_for_seasonal_window_fig_proj_draft1.csv"))
	# pts_s = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/final_stats/ua_swe/decades/snotel_pts_level_decadal_stats_by_drought_for_seasonal_window_fig_proj_draft1.csv"))

	# print(huc8_ua)
	# print(huc8_s)
	make_decade_count_fig(huc8_ua,huc8_s,output_dir,sort_col='pts')


if __name__ == '__main__':
	
	output_dir = str(sys.argv[1])
	main(output_dir)
