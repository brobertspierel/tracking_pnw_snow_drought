#script for analyzing sp decline from rs data

import os 
import sys
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

def get_csv(csv): 
	df = pd.read_csv(csv)
	print(df)
	#print(df.head())
	#print(df.columns)
	df = df[['year','site_num','first']]
	print(df)
	return df

def plot_data(input_df): 
	fig,ax=plt.subplots()
		#df_slice.loc[df_slice['pct_err']<=100].plot(column='pct_err',ax=ax,legend=True,cax=cax,cmap=cmap,norm=norm)
		            #ax1 = wy_df.T.plot.line(ax=ax1,color=['violet','purple','darkred',])

	input_df.groupby('site_num').plot.line(ax=ax, x='year',y='first',legend=False,color='darkgreen',alpha=0.25)
	ax.set_title('MODIS binary snow core winter months Oregon Snotel Sites')
	ax.set_ylabel('snow pixels/pixel count')
	#ax.set_xticklabels(ax.get_xticklabels(input_df['year'].unique()), rotation=45, ha='right')
	ax.set_xticks(input_df['year'].unique())
	plt.xticks(rotation=45)
	plt.show()
	plt.close('all')
	#ax = input_df.plot(columns)

def read_bin_data(file): 
	loaded_array = np.fromfile(file, dtype=np.uint8)
	loaded_array = loaded_array.reshape((720,720))
	loaded_array = np.where(loaded_array<=1,loaded_array,np.nan)
	print(loaded_array.shape)
	plt.plot(loaded_array)
	plt.show()
	plt.close('all')
def main():
	input_csv = "/vol/v1/general_files/user_files/ben/sp_decline/gee_output/sp_or_snotel_test_corrected.csv"
	input_bin = "/vol/v1/general_files/user_files/ben/sp_decline/pm_data/EASE2_N25km.snowice.20181119-20181125.v04.bin"
	#read_bin_data(input_bin)
	plot_data(get_csv(input_csv))

if __name__ == '__main__':
	main()