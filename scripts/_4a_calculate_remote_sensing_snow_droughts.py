import os 
import sys
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import geopandas as gpd
import _3_obtain_all_data as obtain_data
import remote_sensing_functions as rs_funcs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager

#assign some lists of HUC4 top level basins by longitude (should consider amending)
eastern = ['1701','1702','1705','1703','1601','1707','1706','1712','1704']
western = ['1708','1801','1710','1711','1709']


def convert_date(input_df,col_of_interest): 
	"""Helper function."""
	input_df[col_of_interest] = pd.to_datetime(input_df[col_of_interest],errors='coerce')
	return input_df[col_of_interest]


def create_snow_drought_subset(input_df,col_of_interest,huc_level): 
	"""Helper function."""

	drought_list = ['dry','warm','warm_dry','date']
	try: 
		drought_list.remove(col_of_interest)
	except Exception as e: 
		print(f'Error was: {e}')
	df = input_df.drop(columns=drought_list)
	
	df['huc_id'] = df['huc_id'].astype('int')
	
	df[col_of_interest] = convert_date(df,col_of_interest)
	
	#rename cols to match rs data for ease 
	df.rename(columns={col_of_interest:'date','huc_id':'huc'+huc_level},inplace=True)
	#get rid of na fields
	
	df = df.dropna()

	return df

#functions for plotting 

def plot_quantiles(input_df,elev_field,data_field,huc_field,ylabel,water_year,palette):
	"""Plot basins by elevation quantile."""
	palette = list(palette.values())

	font = {'family' : 'Times New Roman',
			        'weight' : 'normal',
			        'size'   : 14}

	plt.rc('font', **font)
	#now plot the snow obs by elevation
	input_df[huc_field] = input_df[huc_field].astype('str')
	
	#drop unneeded cols if they're still there 
	try: 
		input_df.drop(columns=['.geo'],inplace=True)
	except KeyError: 
		pass

	#assign geographic labels 
	labels = ['Eastern basins','Western basins']
	
	#assign quantiles- currently hardcoded for low and high quartiles 
	#print('elev col',input_df[[elev_field,huc_field]])
	#print('elev counts',input_df[elev_field].value_counts())
	# low, high = input_df[elev_field].quantile([0.25,0.75])
	# #print(low,high)
	# #print('input',input_df)
	# #print(input_df.dtypes)
	# df_low = input_df.loc[input_df[elev_field]<=low] #get the 25% quartile df
	# #print('low df')
	# #print(df_low)
	# df_high = input_df.loc[input_df[elev_field]>=high] #get the 75% quartile df
	
	fig,ax=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,8))

	count = 0 

	for i in list([eastern,western]): 

		#get a low and high df for east and west (depending on the iteration)
		region_df = input_df.loc[input_df[huc_field].str.contains('|'.join(i))]
		
		low,high = region_df[elev_field].quantile([0.25,0.75])

		df_low = region_df.loc[input_df[elev_field]<=low] #get the 25% quartile df
		#print('low df')
		#print(df_low)
		df_high = region_df.loc[input_df[elev_field]>=high] #get the 75% quartile df

		# print('low_plot_df')
		# print(low_plot_df)
		# high = df_high.loc[df_high[huc_field].str.contains('|'.join(i))]
		# print('high_plot_df')
		# print(high_plot_df)
		#assign the color palette. This should probably be made into a global variable or put it into the params 
		#pal=['#a6cee3','#1f78b4','#b2df8a','#33a02c']
		try: 
			sns.boxplot(x="variable", y="value", data=pd.melt(df_low[data_field]),ax=ax[count][0],palette=palette)
			ax[count][0].set_xlabel(' ')
			ax[count][0].set_xticklabels(['Dry','Warm','Warm/dry','No drought'])
			ax[count][0].set_axisbelow(True)
			ax[count][0].grid(True,axis='both')

			sns.boxplot(x="variable", y="value", data=pd.melt(df_high[data_field]),ax=ax[count][1],palette=palette)
			ax[count][1].set_xlabel(' ')
			ax[count][1].set_xticklabels(['Dry','Warm','Warm/dry','No drought'])
			ax[count][1].set_axisbelow(True)		
			ax[count][1].grid(True,axis='both')
			
			#title the subplots 
			ax[count][0].set_title(f'{water_year} water year {labels[count]} \n 25th elevation quartile')
			ax[count][1].set_title(f'{water_year} water year {labels[count]} \n 75th elevation quartile')
			
			#add an axis title only in the case that its the left plot 
		
			if ylabel == None: 
				ax[count][0].set_ylabel('MODIS snow persistence')
				#ax[count][0].set_ylabel('MODIS snow persistence')
			else: 
				ax[count][0].set_ylabel(ylabel)
				#ax[count][0].set_ylabel(ylabel)
			
			ax[count][1].set_ylabel(' ')
		
				
			
			count+=1
		except Exception as e: 
			print(f'The error is: {e}')
			raise
			print('You likely forgot to change the year of interest arg or one of the other inputs. Double check year consistency and try again.')
	
	plt.tight_layout()
	plt.show()
	plt.close('all')

def main():
	"""
	Plot the long term snow drought types and trends. 
	"""
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		season = variables["season"]
		pnw_shapefile = variables["pnw_shapefile"]
		huc_shapefile = variables['huc_shapefile']
		us_boundary = variables['us_boundary']
		stations = variables["stations"]		
		pickles = variables["pickles"]
		agg_step = variables["agg_step"]
		year_of_interest = int(variables["year_of_interest"])
		hucs_data = variables["hucs_data"]
		sentinel_csv_dir = variables["sentinel_csv_dir"]
		optical_csv_dir = variables["optical_csv_dir"]
		huc_level = variables["huc_level"]
		resolution = variables["resolution"]
		palette = variables["palette"]

		#user defined functions 
		plot_func = 'quartile'
		elev_stat = 'elev_mean'
		#self,sentinel_data,optical_data,snotel_data,hucs_data,huc_level,resolution): 
		#get all the data 
		snotel_data = pickles+f'short_term_snow_drought_{year_of_interest}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
		
		#instantiate the acquireData class and read in snotel, sentinel and modis/viirs data 
		input_data = obtain_data.AcquireData(sentinel_csv_dir,optical_csv_dir,snotel_data,hucs_data,huc_level,resolution)
		short_term_snow_drought = input_data.get_snotel_data()
		sentinel_data = input_data.get_sentinel_data('filter')
		optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
		
		# pd.set_option("display.max_rows", None, "display.max_columns", None) #change to print an entire df
		# #combine the sentinel and optical data 
		
		#drop redundant columns 
		sentinel_data.drop(columns=['elev_min','elev_mean','elev_max'],inplace=True)
		rs_df=rs_funcs.merge_remote_sensing_data(optical_data,sentinel_data)
		#remove snow persistence values lower than 20% as per (Saavedra et al)
		if 'SP' in optical_csv_dir: 
			rs_df = rs_df.loc[rs_df['NDSI_Snow_Cover']>= 0.2]
		else: 
			pass
		
		rs_df['wet_snow_by_area'] = rs_df['filter']/rs_df['NDSI_Snow_Cover'] #calculate wet snow as fraction of snow covered area
		
		#make sure that the cols used for merging are homogeneous in type 
		rs_df['huc8'] = pd.to_numeric(rs_df['huc'+huc_level])
		rs_df['date'] = convert_date(rs_df,'date')

		
		
		#create the different snow drought type dfs 

		#do dry first 
		dry_combined = create_snow_drought_subset(short_term_snow_drought,'dry',huc_level)
		#merge em 
		dry_combined=dry_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
		#get the rs data for the time periods of interest for a snow drought type 
		#dry_optical=dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].mean()
		dry_combined.rename(columns={'wet_snow_by_area':'dry_WSCA'},inplace=True)
		dry_combined = dry_combined.sort_values('dry_WSCA').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')

		#dry_sar = dry_combined.groupby('huc'+huc_level)['dry_WSCA',elev_stat].median() #changed col from pct change to filter 2/1/2021
		#then do warm 
		warm_combined = create_snow_drought_subset(short_term_snow_drought,'warm',huc_level)
		#merge em 
		warm_combined=warm_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner')
		#get the rs data for the time periods of interest for a snow drought type 
		#warm_optical=warm_combined.groupby('huc'+huc_level)['ndsi_pct_change'].min() 
		warm_combined.rename(columns={'wet_snow_by_area':'warm_WSCA'},inplace=True)
		warm_combined = warm_combined.sort_values('warm_WSCA').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')

		#warm_sar = warm_combined.groupby('huc'+huc_level)['warm_WSCA',elev_stat].median()
		
		#then do warm/dry
		warm_dry_combined = create_snow_drought_subset(short_term_snow_drought,'warm_dry',huc_level)
		#merge em 
		warm_dry_combined=warm_dry_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner')
		#get the rs data for the time periods of interest for a snow drought type 
		#warm_dry_optical=warm_dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].sum()
		warm_dry_combined.rename(columns={'wet_snow_by_area':'warm_dry_WSCA'},inplace=True)
		warm_dry_combined = warm_dry_combined.sort_values('warm_dry_WSCA').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')

		print(warm_dry_combined)
		#warm_dry_sar = warm_dry_combined.groupby('huc'+huc_level)['warm_dry_WSCA',elev_stat].median()

		#try making a df of time steps that DO NOT have snow droughts for comparing
		no_snow_drought = create_snow_drought_subset(short_term_snow_drought,'date',huc_level)
		no_drought_combined=no_snow_drought.merge(rs_df, on=['date','huc'+huc_level],how='inner')

		no_drought_combined.rename(columns={'wet_snow_by_area':'no_drought_WSCA'},inplace=True)
		no_drought_combined = warm_dry_combined.sort_values('warm_dry_WSCA').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')

		#no_drought_sar = no_drought_combined.groupby('huc'+huc_level)['no_drought_WSCA'].median()
		#print(no_drought_sar)
		#print('no drought sar', no_drought_sar.shape)
		

		# #plot it 
		# if plot_func.lower() == 'quartile': 
		# 	#dfs = [dry_sar.reset_index(),warm_sar.reset_index(),warm_dry_sar.reset_index()]
		# 	dfs = dry_sar.reset_index().merge(warm_sar.reset_index(),on=['huc'+huc_level],how='outer')
		# 	dfs = dfs.merge(warm_dry_sar.reset_index(),on=['huc'+huc_level],how='outer')
		# 	dfs.drop(columns={f'{elev_stat}_x',f'{elev_stat}_y'},inplace=True)
		# 	print('dfs shape',dfs.shape)
			
		# 	dfs = dfs.merge(no_drought_sar.reset_index(),on=['huc'+huc_level],how='outer')
		
		# 	#do a little bit of cleaning
		# 	dfs.replace(np.inf,np.nan,inplace=True)
		# 	#dfs[['dry_WSCA','warm_WSCA','warm_dry_WSCA','no_drought_WSCA']]>=1 = 1 #=dfs[dfs['dry_WSCA','warm_WSCA','warm_dry_WSCA','no_drought_WSCA']>=1,]
			
		# 	print('dfs look like: ',dfs)
		# 	#anywhere wet snow exceeeds snow covered area ie value is greater than 1, set it to 1
		# 	# dfs['dry_WSCA'] = dfs['dry_WSCA']
		# 	#dfs.loc[dfs['dry_WSCA','warm_WSCA','warm_dry_WSCA'] > 1] = 1  
		# 	dfs.loc[dfs['dry_WSCA'] > 1,'dry_WSCA'] = np.nan
		# 	dfs.loc[dfs['warm_WSCA'] > 1,'warm_WSCA'] = np.nan
		# 	dfs.loc[dfs['warm_dry_WSCA'] > 1,'warm_dry_WSCA'] = np.nan  
		# 	dfs.loc[dfs['no_drought_WSCA'] > 1,'no_drought_WSCA'] = np.nan
		# 	#df.loc[df.ID == 103, ['FirstName', 'LastName']] = 'Matt', 'Jones'

		# 	print('dfs now look like: ', dfs)
		# 	print(dfs.count()) 
		# 	#dfs[['dry_WSCA','warm_WSCA','warm_dry_WSCA','no_drought_WSCA']] = np.where(dfs[['dry_WSCA','warm_WSCA','warm_dry_WSCA','no_drought_WSCA']] >= 1, 1,dfs)

		# 	# print(dfs.columns)
		# 	# pd.set_option("display.max_rows", None, "display.max_columns", None)
		# 	# print(dfs)
		# 	# print(dfs.mean())
		# 	#dfs.drop(columns=['elev_max_x'],inplace=True)
		# 	#dfs['elev_max'] = dfs['elev_max_y'].astype('int')
		# 	plot_quantiles(dfs,elev_stat,['dry_WSCA','warm_WSCA','warm_dry_WSCA','no_drought_WSCA'],'huc'+huc_level,'Wet snow covered area (sq km)',year_of_interest,palette) #amended 2/1/2021
		
		# elif plot_func.lower() == 'combined': 

		# 	#read in shapefiles for mapping 
		# 	hucs_shp = gpd.read_file(huc_shapefile)
		# 	us_bounds = gpd.read_file(us_boundary)

		# 	#make sure shapefile merging col matches rs data 
		# 	hucs_shp['huc8'] = hucs_shp['huc8'].astype('int32')
			
		# 	#get the droughts data 
		# 	hucs_shp_drought = hucs_shp.merge(warm_dry_optical,how='inner',on='huc'+huc_level)
		# 	hucs_shp_drought = hucs_shp_drought.merge(warm_dry_sar,how='inner',on='huc'+huc_level)
		# 	hucs_shp_drought = hucs_shp_drought.merge(warm_dry_ratio,how='inner',on='huc'+huc_level)
		# 	#hucs_shp_drought['sar_pct_change'] = hucs_shp_drought['sar_pct_change'].replace(np.inf, np.nan)
		# 	print(hucs_shp_drought)		

		# 	#get the no drought rs data 
		# 	hucs_shp_no_drought = hucs_shp.merge(no_drought_optical,how='inner',on='huc8')
		# 	hucs_shp_no_drought = hucs_shp_no_drought.merge(no_drought_sar,how='inner',on='huc8')
		# 	hucs_shp_no_drought = hucs_shp_no_drought.merge(no_drought_ratio,how='inner',on='huc8')
			

		# 	#print(hucs_shp_drought.shape)
		# 	#print(hucs_shp_no_drought.shape)
		# 	#get the basin bounds for mapping 
		# 	minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds

		# 	fig,(ax1,ax2) = plt.subplots(2)
		# 	us_bounds.plot(ax=ax1,color='white', edgecolor='black')

		# 	# #make colorbar axes 
		# 	divider = make_axes_locatable(ax1)
		# 	cax = divider.append_axes('right', size='5%', pad=0.05)
		# 	hucs_shp_drought.plot(ax=ax1,column='filter',vmin=hucs_shp_drought['filter'].min(),vmax=hucs_shp_drought['filter'].max(),legend=True,cax=cax)
			
		# 	# fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(15,15))
		# 	# #subset = hucs_shp_no_drought[hucs_shp_no_drought['sar_pct_change']]
		# 	# #plt.scatter(hucs_shp_drought['sar_pct_change'],hucs_shp_drought['ndsi_pct_change'])
		# 	# #plot US bounds- need to add Canada and states/provinces 
		# 	# us_bounds.plot(ax=ax1,color='white', edgecolor='black')
			
		# 	# #need to change this to make background color for the hucs 
		# 	# #hucs_shp_drought.plot(ax=ax1,color='gray',edgecolor='darkgray')
			
		# 	# #make colorbar axes 
		# 	# divider = make_axes_locatable(ax1)
		# 	# cax = divider.append_axes('right', size='5%', pad=0.05)
			
		# 	# #plot it 
		# 	# hucs_shp_drought.plot(ax=ax1,column='sar_pct_change',vmin=hucs_shp_drought['sar_pct_change'].min(),vmax=hucs_shp_drought['sar_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
			
		# 	# #repeat 
		# 	# ax1.set_title(f'{year_of_interest} mean warm/dry sar change in wet snow area')
		# 	ax1.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		# 	ax1.set_ylim(miny - 1, maxy + 1)

		# 	# us_bounds.plot(ax=ax2,color='white', edgecolor='black')
		# 	# #hucs_shp.plot(ax=ax2,color='gray',edgecolor='darkgray')
		# 	divider = make_axes_locatable(ax2)
		# 	cax = divider.append_axes('right', size='5%', pad=0.05)
		# 	hucs_shp_no_drought.plot(ax=ax2,column='filter',vmin=hucs_shp_drought['filter'].min(),vmax=hucs_shp_drought['filter'].max(),legend=True,cax=cax)

		# 	# hucs_shp_drought.plot(ax=ax2,column='ndsi_pct_change',vmin=hucs_shp_drought['ndsi_pct_change'].min(),vmax=hucs_shp_drought['ndsi_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
		# 	# ax2.set_title(f'{year_of_interest} mean warm/dry optical change in SP')
		# 	ax2.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		# 	ax2.set_ylim(miny - 1, maxy + 1)
			
		# 	# us_bounds.plot(ax=ax3,color='white', edgecolor='black')
		# 	# #hucs_shp.plot(ax=ax3,color='gray',edgecolor='darkgray')
		# 	# divider = make_axes_locatable(ax3)
		# 	# cax = divider.append_axes('right', size='5%', pad=0.05)
		# 	# hucs_shp_no_drought.plot(ax=ax3,column='sar_pct_change',vmin=hucs_shp_drought['sar_pct_change'].min(),vmax=hucs_shp_drought['sar_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
		# 	# ax3.set_title(f'{year_of_interest} mean no drought sar change in SP')
		# 	# ax3.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		# 	# ax3.set_ylim(miny - 1, maxy + 1)

		# 	# us_bounds.plot(ax=ax4,color='white', edgecolor='black')
		# 	# #hucs_shp.plot(ax=ax4,color='gray',edgecolor='darkgray')
		# 	# divider = make_axes_locatable(ax4)
		# 	# cax = divider.append_axes('right', size='5%', pad=0.05)
		# 	# hucs_shp_no_drought.plot(ax=ax4,column='ndsi_pct_change',vmin=hucs_shp_drought['ndsi_pct_change'].min(),vmax=hucs_shp_drought['ndsi_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
		# 	# ax4.set_title(f'{year_of_interest} mean no drought optical change in SP')
		# 	# ax4.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
		# 	# ax4.set_ylim(miny - 1, maxy + 1)


		# 	plt.tight_layout()
		# 	plt.show()
		# 	plt.close('all')

if __name__ == '__main__':
    main()

#working 
# sentinel_data.drop(columns=['elev_min','elev_mean','elev_max'],inplace=True)
# 		rs_df=rs_funcs.merge_remote_sensing_data(optical_data,sentinel_data)
# 		#remove snow persistence values lower than 20% as per (Saavedra et al)
# 		if 'SP' in optical_csv_dir: 
# 			rs_df = rs_df.loc[rs_df['NDSI_Snow_Cover']>= 0.2]
# 		else: 
# 			pass
		
# 		rs_df['wet_snow_by_area'] = rs_df['filter']/rs_df['NDSI_Snow_Cover'] #calculate wet snow as fraction of snow covered area
		
# 		#make sure that the cols used for merging are homogeneous in type 
# 		rs_df['huc8'] = pd.to_numeric(rs_df['huc'+huc_level])
# 		rs_df['date'] = convert_date(rs_df,'date')

		
		
# 		#create the different snow drought type dfs 

# 		#do dry first 
# 		dry_combined = create_snow_drought_subset(short_term_snow_drought,'dry',huc_level)
# 		#merge em 
# 		dry_combined=dry_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
# 		#get the rs data for the time periods of interest for a snow drought type 
# 		#dry_optical=dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].mean()
# 		dry_combined.rename(columns={'wet_snow_by_area':'dry_WSCA'},inplace=True)

# 		dry_sar = dry_combined.groupby('huc'+huc_level)['dry_WSCA',elev_stat].median() #changed col from pct change to filter 2/1/2021
# 		#then do warm 
# 		warm_combined = create_snow_drought_subset(short_term_snow_drought,'warm',huc_level)
# 		#merge em 
# 		warm_combined=warm_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner')
# 		#get the rs data for the time periods of interest for a snow drought type 
# 		#warm_optical=warm_combined.groupby('huc'+huc_level)['ndsi_pct_change'].min() 
# 		warm_combined.rename(columns={'wet_snow_by_area':'warm_WSCA'},inplace=True)
# 		warm_sar = warm_combined.groupby('huc'+huc_level)['warm_WSCA',elev_stat].median()
		
# 		#then do warm/dry
# 		warm_dry_combined = create_snow_drought_subset(short_term_snow_drought,'warm_dry',huc_level)
# 		#merge em 
# 		warm_dry_combined=warm_dry_combined.merge(rs_df, on=['date','huc'+huc_level], how='inner')
# 		#get the rs data for the time periods of interest for a snow drought type 
# 		#warm_dry_optical=warm_dry_combined.groupby('huc'+huc_level)['ndsi_pct_change'].sum()
# 		warm_dry_combined.rename(columns={'wet_snow_by_area':'warm_dry_WSCA'},inplace=True)
# 		warm_dry_sar = warm_dry_combined.groupby('huc'+huc_level)['warm_dry_WSCA',elev_stat].median()

# 		#try making a df of time steps that DO NOT have snow droughts for comparing
# 		no_snow_drought = create_snow_drought_subset(short_term_snow_drought,'date',huc_level)
# 		no_drought_combined=no_snow_drought.merge(rs_df, on=['date','huc'+huc_level],how='inner')

# 		no_drought_combined.rename(columns={'wet_snow_by_area':'no_drought_WSCA'},inplace=True)
# 		no_drought_sar = no_drought_combined.groupby('huc'+huc_level)['no_drought_WSCA'].median()
# 		#print(no_drought_sar)
# 		#print('no drought sar', no_drought_sar.shape)
		

# fig,(ax1,ax2,ax3,ax4) = plt.subplots(ncols=2,figsize=(15,15))
# 		us_bounds.plot(ax=ax1,color='white', edgecolor='black')
# 		hucs_shp.plot(ax=ax1,color='gray',edgecolor='darkgray')
# 		divider = make_axes_locatable(ax1)
# 		cax = divider.append_axes('right', size='5%', pad=0.05)
# 		hucs_shp.plot(ax=ax1,column='sar_pct_change',vmin=hucs_shp['sar_pct_change'].min(),vmax=hucs_shp['sar_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
		
# 		ax1.set_title(f'{year_of_interest} mean sar change in wet snow area')
# 		ax1.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
# 		ax1.set_ylim(miny - 1, maxy + 1)

# 		us_bounds.plot(ax=ax2,color='white', edgecolor='black')
# 		hucs_shp.plot(ax=ax2,color='gray',edgecolor='darkgray')
# 		divider = make_axes_locatable(ax2)
# 		cax = divider.append_axes('right', size='5%', pad=0.05)
# 		hucs_shp.plot(ax=ax2,column='ndsi_pct_change',vmin=hucs_shp['ndsi_pct_change'].min(),vmax=hucs_shp['ndsi_pct_change'].max(),legend=True,cax=cax)#,cmap=cmap,vmin=1980,vmax=2019)#, column='Value1')
# 		ax2.set_title(f'{year_of_interest} mean optical change in SP')
# 		ax2.set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
# 		ax2.set_ylim(miny - 1, maxy + 1)
		
# 		plt.tight_layout()
# 		plt.show()
# 		plt.close('all')







# def plot_quartiles(input_df,elev_field,data_field,huc_field,ylabel):
# 	#now plot the snow obs by elevation
# 	input_df[huc_field] = input_df[huc_field].astype('str')
# 	try: 
# 		input_df.drop(columns=['.geo'],inplace=True)
# 	except KeyError: 
# 		pass
# 	labels = ['Eastern basins','Western basins']
# 	colors= ['lightgreen','lightblue']
# 	#print(input_df)
# 	low, high = input_df[elev_field].quantile([0.25,0.75])
# 	df_low = input_df.loc[input_df[elev_field]<=low] #get the 25% quartile
# 	df_high = input_df.loc[input_df[elev_field]>=high] #get the 75% quartile 
# 	fig,ax=plt.subplots(2,2,sharey=True)
# 	#split the plotting df.loc[df['type'].isin(substr)] df.loc[df['type].str.contains('|'.join(substr))]
# 	count = 0 
# 	for i in list([eastern,western]): 
		
# 		low_plot_df = df_low.loc[df_low[huc_field].str.contains('|'.join(i))]
# 		high_plot_df = df_high.loc[df_high[huc_field].str.contains('|'.join(i))]
# 		#low_plot_df=low_plot_df.groupby('date').mean()#.dropna() #collapse the basins into one mean
# 		#high_plot_df=high_plot_df.groupby('date').mean()#.dropna()

# 		sns.boxplot(low_plot_df[data_field],ax=ax[count][0],orient='v',color=colors[count])
# 		#sns.boxplot(high_plot_df[data_field],ax=ax[count][0],orient='v',color=colors[count])


# 		ax[count][0].set_title(f'{labels[count]} 25th elevation quartile')
# 		if ylabel == None: 
# 			ax[count][0].set_ylabel('MODIS snow persistence')
# 		else: 
# 			ax[count][0].set_ylabel(ylabel)
# 		ax[count][0].set_axisbelow(True)

# 		ax[count][0].grid(True,axis='both')
# 		#$ax[count][0].grid(True)

# 		sns.boxplot(high_plot_df[data_field],ax=ax[count][1],orient='v',color=colors[count])
# 		ax[count][1].set_title(f'{labels[count]} 75th elevation quartile')
# 		if ylabel == None: 
# 			ax[count][1].set_ylabel('MODIS snow persistence')
# 		else: 
# 			ax[count][1].set_ylabel(ylabel)
# 		ax[count][1].set_axisbelow(True)		
# 		ax[count][1].grid(True,axis='both')
# 		#ax[count][1].xaxis.grid(True)
# 		count+=1
# 	plt.tight_layout()
# 	plt.show()
# 	plt.close('all')
