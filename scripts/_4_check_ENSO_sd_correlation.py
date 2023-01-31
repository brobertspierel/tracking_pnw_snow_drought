import os 
import sys
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import scipy.stats as stats
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

pd.options.mode.chained_assignment = None  # default='warn'


def add_mean_col(df): 
	#take the mean of cols that are not year 
	df['mean'] = df[[c for c in df if not c == 'year']].mean(axis=1)
	return df 

def main(enso_data,sd_data,output_dir): 
	try: 
		en_df = pd.read_csv(enso_data)
	except Exception as e: 
		en_df = pd.read_excel(enso_data,sheet_name=0)
	#subset enso data 
	early = add_mean_col(en_df[['november','december','year']])
	mid = add_mean_col(en_df[['january','february','year']])
	late = add_mean_col(en_df[['march','april','year']])
	#create some iterable lists of labels and dfs
	cols = ['Dry','Warm','Warm/dry']
	per_labels = ['Early','Mid','Late']
	pers = [early,mid,late]

	fig,axs = plt.subplots(3,3,
							figsize=(8,6),
							sharex=True,
							sharey=True,
							gridspec_kw={'wspace':0.0,'hspace':0.0})
	
	for row in range(3): 
		#get the sd data, this will be the same for the row
		df = pd.read_csv(sd_data[row])
		en = pers[row]
		for col in range(3): 
			sd_df = df[[cols[col],'Year']] #snow drought type 
			sd_df.rename(columns={'Year':'year'},inplace=True) #rename year col to match enso data 
			
			
			if row == 0:
				# print('correcting year') 
				# print(en)	
				#these are calendar years but sd data are for water years. 
				#shift the enso data so they line up with the water years. 
				en['year'] = en['year']+1
				# print('after')
				# print(en)
			else: 
				pass
			merged = sd_df.merge(en,how='inner',on='year')	
			
			# axs[row][col].scatter(merged['mean'],
			# 					merged[cols[col]],
			# 					facecolors='none', 
			# 					edgecolors='black',
			# 					s=50,
			# 					alpha=0.25)

			#ENSO
			axs[row][col].plot(merged['year'],
								merged['mean'], 
								c='black',
								linestyle='-', 
								lw=2
								)
			#sd
			ax2 = axs[row][col].twinx()
			ax2.plot(merged['year'],
					merged[cols[col]], 
					c='black',
					linestyle='--', 
					lw=2
					)

			#deal with labeling
			if row == 0: 
				axs[row][col].set_title(cols[col],fontsize=10)
			if col == 0: 
				axs[row][col].set_ylabel(per_labels[row],fontsize=10)
			if not col == 2: 
				ax2.yaxis.set_visible(False)
			if row == 2: 
				axs[row][col].xaxis.set_tick_params(rotation=90)


			#add correlation coefficients 
			corr, _ = pearsonr(merged['mean'], merged[cols[col]])
			rho, pval = stats.spearmanr(merged['mean'], merged[cols[col]])
			axs[row][col].annotate(f'r = {round(corr,2)}',xy=(0.03,0.88),xycoords='axes fraction',fontsize=10)

	#add a legend 
	custom_lines = [Line2D([0], [0], color='black', lw=2),
					Line2D([0], [0], color='black', lw=2, linestyle='--')]            
	axs[0][2].legend(custom_lines, ['Ni\u00F1o 3.4', 'Snow droughts'])
	
	#fig.text(0.5, 0.035, f'Ni\u00F1o 3.4 winter period mean', ha='center',fontsize=10)
	fig.text(0.95, 0.5, 'Snow drought counts', va='center', rotation='vertical',fontsize=10) #secondary axis
	fig.text(0.03, 0.5, 'Ni\u00F1o 3.4 winter period mean (deg C)', va='center', rotation='vertical',fontsize=10) #primary axis

	# plt.show()
	# plt.close('all')
	fig_fn = os.path.join(output_dir,'ENSO_time_series_fig_draft11.jpg')
	if not os.path.exists(fig_fn): 
		plt.savefig(fig_fn, 
					dpi=500, 
					bbox_inches = 'tight',
					pad_inches = 0.1, 
					)

if __name__ == '__main__':
	enso_indices = "/vol/v1/general_files/user_files/ben/excel_files/ENSO/noaa_nino_3.4.xls"#"/vol/v1/general_files/user_files/ben/excel_files/ENSO/ENSO_data.csv"
	early_sd = "/vol/v1/general_files/user_files/ben/excel_files/ENSO/early_sd.csv"
	mid_sd = "/vol/v1/general_files/user_files/ben/excel_files/ENSO/mid_sd.csv"
	late_sd = "/vol/v1/general_files/user_files/ben/excel_files/ENSO/late_sd.csv"
	output = "/vol/v1/general_files/user_files/ben/paper_figures/figures/ua_swe_figs/final_round/other/"

	main(enso_indices,
		[early_sd,mid_sd,late_sd],
		output)


# try: 
# 		en_df = pd.read_csv(enso_data)
# 	except Exception as e: 
# 		en_df = pd.read_excel(enso_data,sheet_name=0)
# 	print(en_df)
# 	early = ['OND']#['OND','NDJ','DJF'] #nov,dec
# 	mid = ['JFM']#['DJF','JFM','FMA'] #jan,feb
# 	late = ['MAM']#['FMA','MAM','AMJ'] #apr,mar
# 	cols = ['Dry','Warm','Warm/dry']
	
# 	enso_pers = {'Early':'NDJ','Mid':'DJF','Late':'MAM'}

# 	per_labels = ['Early','Mid','Late']

# 	pers = [early,mid,late]
# 	fig,axs = plt.subplots(3,3,
# 							figsize=(8,6),
# 							sharex=True,
# 							sharey=True,
# 							gridspec_kw={'wspace':0,'hspace':0})
# 	count = 0 
# 	for row in range(3): 
# 		#get the sd data, this will be the same for the row
# 		df = pd.read_csv(sd_data[row])

# 		for col in range(3): 
# 			sd_df = df[[cols[col],'Year']] #snow drought type 
			
# 			for per in pers[row]:
# 				print(en_df)
# 				print('per is: ')
# 				print(per)
# 				en = en_df[[per,'Year']]
# 				print(en)
# 				if 'N' in per:
# 					print('correcting year') 
# 					#these are calendar years but sd data are for water years. 
# 					#shift the enso data so they line up with the water years. 
# 					en['Year'] = en['Year']+1
# 					print(en)
# 					count +=1
# 				else: 
# 					pass
# 				merged = sd_df.merge(en,how='inner',on='Year')	
# 				print(merged)
				
# 				#ENSO
# 				axs[row][col].plot(merged['Year'],
# 									merged[per], 
# 									c='black',
# 									linestyle='-'
# 									)
# 				#sd
# 				ax2 = axs[row][col].twinx()
# 				ax2.plot(merged['Year'],
# 						merged[cols[col]], 
# 						c='black',
# 						linestyle='--'
# 						)

# 			#deal with labeling
# 			if row == 0: 
# 				axs[row][col].set_title(cols[col])
# 			if col == 0: 
# 				axs[row][col].set_ylabel(per_labels[row])

# 			#add correlation coefficients 
# 			rho, pval = stats.spearmanr(merged[per], merged[cols[col]])
# 			axs[row][col].annotate(f'r = {round(rho,2)}',xy=(0.05,0.85),xycoords='axes fraction',fontsize=10)
