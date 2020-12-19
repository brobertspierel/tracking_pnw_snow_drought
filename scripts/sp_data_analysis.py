import os 
import sys
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
def get_sp_data(csv_dir): 
	csv_list = []
	#for csv in sorted(glob.glob(csv_dir+'*huc08_mean_elev.csv')): #this is hardcoded and should be changed  
	df = pd.read_csv(csv_dir,parse_dates=True) 
	df.rename(columns={'system:time_start':'date'},inplace=True)
	df['date'] = pd.to_datetime(df['date'])
	#df['huc'] = os.path.split(csv)[1].split('_')[2] #hardcoded, this should be changed
	#csv_list.append(df)
	#output_df = pd.concat(csv_list,axis=0)
	
	#plot_df = output_df[output_df['site_num']==1704]
	#get the huc ids
	return df
def plot_long_term_sp_data(input_df): 
	fig,ax = plt.subplots(4,6,figsize=(10,10),sharex=True,sharey=True)
	ax = ax.flatten()
	count = 0
	for i in output_df['site_num'].unique(): 
		plot_df = output_df[output_df['site_num']==i]
		#plot the anom from the mean
		#sp_mean = plot_df['nd'].mean()
		plot_df['mean'] = plot_df['nd'].mean()
		plot_df['anom'] = (plot_df['nd']-plot_df['mean'])+plot_df['mean']
		print(plot_df)
		#plot_df = plot_df[(plot_df['year']>2000)&(plot_df['year']<2021)]
		ax[count].plot(plot_df['year'],plot_df['mean'],linestyle='--',color='darkblue')
		ax[count].plot(plot_df['year'],plot_df['anom'],color='darkorange')
		
		#sns.lineplot(x='year',y='mean',data=plot_df,ax=ax[count],linestyle=':',color='darkblue')
		#sns.lineplot(x='year',y='anom',data=plot_df,ax=ax[count])
		ax[count].set_title(f'HUC 6 code {plot_df["site_num"].iloc[0]}')
		ax[count].set_ylabel('Snow persistence')
		count += 1 
	#ax[-1].set_visible(False)
	#ax[-2].set_visible(False)
	fig.delaxes(ax[-1])
	fig.delaxes(ax[-2])
	#print(output_df)
	#plt.plot(output_df['year'],output_df['nd'])
	plt.tight_layout()
	plt.show()
	plt.close()

# def main(): 
# 	#get_sp_data("/vol/v1/general_files/user_files/ben/excel_files/landsat_sp_data/")
# if __name__ == '__main__':
#     main()
