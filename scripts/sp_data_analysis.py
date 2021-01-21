import os 
import sys
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

western = ['1708','1801','1710','1711','1709']
eastern = ['1701','1702','1705','1703','1601','1707','1706','1712','1704']


def get_sp_data(input_csv): 
	csv_list = []
	#for csv in sorted(glob.glob(csv_dir+'*huc08_mean_elev.csv')): #this is hardcoded and should be changed  
	df = pd.read_csv(input_csv,parse_dates=True) 
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

def plot_elevation_distribution(input_df,elev_field,data_field,huc_field): 
	
	input_df[huc_field] = input_df[huc_field].astype('str')
	input_df.drop(columns=['.geo'],inplace=True)
	#plot the elevation distribution
	labels = ['Eastern basins','Western basins']
	colors= ['lightgreen','lightblue']
	count=0
	fig,ax=plt.subplots(2)
	for j in list([eastern,western]): 
		elev_df = input_df.loc[input_df[huc_field].str.contains('|'.join(j))]
		#elev_df = elev_df.filter(like=('elev|huc'),axis=1)
		elev_df=elev_df[elev_df.filter(regex='elev|huc').columns]
		print(elev_df)
		elev_df.boxplot(column=['elev_min','elev_mean','elev_max'],ax=ax[count])
		ax[count].set_title(labels[count])
		ax[count].set_ylabel('Elevation (m asl')
		count+=1
	plt.tight_layout()
	plt.show()
	plt.close('all')

def plot_quartiles(input_df,elev_field,data_field,huc_field,ylabel):
	#now plot the snow obs by elevation
	input_df[huc_field] = input_df[huc_field].astype('str')
	input_df.drop(columns=['.geo'],inplace=True)
	labels = ['Eastern basins','Western basins']
	colors= ['lightgreen','lightblue']
	low, high = input_df[elev_field].quantile([0.25,0.75])
	df_low = input_df.loc[input_df[elev_field]<=low] #get the 25% quartile
	df_high = input_df.loc[input_df[elev_field]>=high] #get the 75% quartile 
	fig,ax=plt.subplots(2,2,sharey=True)
	#split the plotting df.loc[df['type'].isin(substr)] df.loc[df['type].str.contains('|'.join(substr))]
	count = 0 
	for i in list([eastern,western]): 
		print('count is: ',count)
		print('i is: ',i)
		low_plot_df = df_low.loc[df_low[huc_field].str.contains('|'.join(i))]
		high_plot_df = df_high.loc[df_high[huc_field].str.contains('|'.join(i))]
		low_plot_df=low_plot_df.groupby('date').mean().dropna() #collapse the basins into one mean
		high_plot_df=high_plot_df.groupby('date').mean().dropna()

		print('low',low_plot_df)
		print('high',high_plot_df)
		sns.boxplot(low_plot_df[data_field],ax=ax[count][0],orient='v',color=colors[count])

		ax[count][0].set_title(f'{labels[count]} 25th elevation quartile')
		if ylabel == None: 
			ax[count][0].set_ylabel('MODIS snow persistence')
		else: 
			ax[count][0].set_ylabel(ylabel)
		ax[count][0].set_axisbelow(True)

		ax[count][0].grid(True,axis='both')
		#$ax[count][0].grid(True)

		sns.boxplot(high_plot_df[data_field],ax=ax[count][1],orient='v',color=colors[count])
		ax[count][1].set_title(f'{labels[count]} 75th elevation quartile')
		if ylabel == None: 
			ax[count][1].set_ylabel('MODIS snow persistence')
		else: 
			ax[count][1].set_ylabel(ylabel)
		ax[count][1].set_axisbelow(True)		
		ax[count][1].grid(True,axis='both')
		#ax[count][1].xaxis.grid(True)
		count+=1
	plt.tight_layout()
	plt.show()
	plt.close('all')
		#print(west_df)
		#east_df = input_df.loc[input_df['huc_id'].str.contains('|'.join(eastern))]
def plot_time_series(df): 
	'''Plot a simple time series.'''
	fig,ax = plt.subplots()
	
	ax.plot(df['date'],df['NDSI_Snow_Cover'])
	plt.show()
	plt.close('all')
	#print(df_low)
	#print('low is: ', low)
	#print('high is: ', high)
def main(): 
	df=get_sp_data("/vol/v1/general_files/user_files/ben/excel_files/modis_data/MODIS_500m_masked_revised_2015-10-01_2016-07-01_huc08_all_elev_stats.csv")
	print(df.head())
	plot_time_series(df)
	#plot_quartiles(df,'elev_mean','NDSI_Snow_Cover','huc8',None)

if __name__ == '__main__':
    main()
