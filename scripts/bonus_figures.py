import sys 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

def fix_headers(df): 
	df.columns = df.columns.str.replace(' ','_')
	df.columns = df.columns.str.replace('\n','_')
	df.columns = df.columns.str.replace('__','_')
	df.columns = df.columns.str.replace('/','_')
	try: 
		df.drop(columns=['Snotel'], inplace = True)
	except KeyError: 
		df.drop(columns=['Daymet'], inplace = True)

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

def make_decade_count_fig(daymet_df,snotel_df,output_dir,sort_col='huc8'): 
	fontsize = 10
	fig, (ax1,ax2,ax3) = plt.subplots(1,3, sharex=True,sharey=True,
				gridspec_kw={'wspace':0,'hspace':0},
				figsize=(8,6))

	early_df = plot_format(reformat_df(daymet_df,0),'Daymet',0).merge(plot_format(reformat_df(snotel_df,0), 'SNOTEL', 0), on='index', how='inner')
	mid_df = plot_format(reformat_df(daymet_df,1),'Daymet', 1).merge(plot_format(reformat_df(snotel_df,1), 'SNOTEL', 1), on='index', how='inner')
	late_df = plot_format(reformat_df(daymet_df,2),'Daymet', 2).merge(plot_format(reformat_df(snotel_df,2), 'SNOTEL', 2), on='index', how='inner')

	early_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax1, legend=False)
	mid_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax2, legend=False)
	late_df.plot.bar(x='index',color=['#267eab','#D95F0E'],ax=ax3)
	#add horizontal grid lines

	for ax, per in zip([ax1,ax2,ax3],['Early','Mid','Late']): 
		ax.tick_params(axis='x', labelsize=fontsize)
		ax.tick_params(axis='y', labelsize=fontsize)
		ax.set_xlabel(' ')
		ax.grid(axis='y',alpha=0.25)
		ax.annotate(per,xy=(0.75,0.9),xycoords='axes fraction',fontsize=fontsize)#f'{chr(97)}'
	
	#do a couple of ax specific things 
	ax1.set_ylabel('Number of basins',fontsize=fontsize)
	ax3.legend(fontsize=12,loc='upper left')
	#plt.rcParams['legend.title_fontsize'] = fontsize


	plt.tight_layout()
	# plt.show()
	# plt.close('all')
	output_fn = os.path.join(output_dir,f'{sort_col}_snotel_daymet_decades_basin_counts_draft2.jpg')
	if not os.path.exists(output_fn): 
		plt.savefig(output_fn, dpi=400)

def main(output_dir): 

	huc8_d = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/writing/updated_stats/csvs_for_last_figs/HUC8_w_delta_SWE_daymet.csv"))
	huc8_s = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/writing/updated_stats/csvs_for_last_figs/HUC8_w_delta_SWE_snotel.csv"))
	huc6_d = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/writing/updated_stats/csvs_for_last_figs/HUC6_w_delta_SWE_daymet.csv"))
	huc6_s = fix_headers(pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/writing/updated_stats/csvs_for_last_figs/HUC6_w_delta_SWE_snotel.csv"))
	
	# huc8_d = huc8_d.reindex(sorted(huc8_d.columns), axis=1)


	# print(huc8_d.T)
	# #just get early 
	# print(huc8_d.iloc[0].to_frame().T)

	make_decade_count_fig(huc6_d,huc6_s,output_dir,sort_col='huc6')


if __name__ == '__main__':
	output_dir = "/vol/v1/general_files/user_files/ben/paper_figures/figures/final_figs_updated/"
	main(output_dir)
