import pandas as pd 
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt 
import geopandas as gpd 
import json 
import glob
import datetime
from functools import reduce
from _1_calculate_revised_snow_drought import FormatData,CalcSnowDroughts
import snow_drought_definition_revision as sd
from snow_drought_definition_revision import DefineClusterCenters
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def plot_dist_to_cluster_center(df_list,source,huc_col='huc8',**kwargs): #list of dfs with year, huc and dist to center (sqdist). List has three dfs, one for each period 
	"""Make a plot of dist to cluster centroids."""

	if source.lower() == 'snotel': 
		source = 's'
	elif source.lower() == 'daymet': 
		source = 'd'
	else: 
		print('Check the source variable. It can be daymet or snotel as of 6/12/2021.')
	
	fig,axs = plt.subplots(3,3, gridspec_kw = {'wspace':0, 'hspace':0}, sharex=True, sharey=True)
	
	#read in some of the base shapefiles: 
	hucs_shp = gpd.read_file(kwargs.get('huc_shapefile'))
	us_bounds = gpd.read_file(kwargs.get('us_boundary'))
	pnw_states = gpd.read_file(kwargs.get('pnw_shapefile'))
	canada = gpd.read_file(kwargs.get("canada"))
	hucs_shp[huc_col] = hucs_shp[huc_col].astype('int32')

	cols = ['dry','warm','warm_dry']
	xlabels=['Dry', 'Warm', 'Warm/dry']
	ylabels=['Early','Mid','Late']
	cmap='cubehelix'
	vmin=0
	vmax=4
	for row in range(3): 
		for col in range(3): 
			df = df_list[row] #this is the temporal chunk 
			df = df[[f'{source}_sqdist', f'{source}_{cols[col]}',huc_col]].dropna() #get the drought type from the df 
			df[huc_col] = df[huc_col].astype(int)
			df = df.groupby(huc_col)[f'{source}_sqdist'].mean().reset_index()
			plot_dict = dict(zip(list(df[huc_col]),list(df[f'{source}_sqdist']))) #make a dict from two cols to add to the spatial data 
			hucs_shp['dist'] = hucs_shp[huc_col].map(plot_dict)
			print(hucs_shp.dist.min())
			print(hucs_shp.dist.max())
			#plot it! 
			canada.plot(ax=axs[row][col],color='#f2f2f2', edgecolor='darkgray',lw=0.5)
			us_bounds.plot(ax=axs[row][col],color='#f2f2f2', edgecolor='black',lw=0.5)
			pnw_states.plot(ax=axs[row][col],color='#f2f2f2',edgecolor='darkgray',lw=0.5)
			im = hucs_shp.plot(ax=axs[row][col],column='dist',vmin=vmin,vmax=vmax,cmap=cmap)
			
			#set the extent to the PNW 
			minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds
			axs[row][col].set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
			axs[row][col].set_ylim(miny - 1, maxy + 1)

			#set titles  
			axs[0][col].set_title(xlabels[col],fontdict={'fontsize': 14})
			
			axs[row][0].set_ylabel(ylabels[row],fontsize=14)	

	
	cax = fig.add_axes([0.9, 0.125, 0.03, 0.75]) #formatted like [left, bottom, width, height]
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	fig.colorbar(sm, cax=cax)
	# plt.show() 
	# plt.close('all')
	plt.savefig(os.path.join(kwargs.get('output_dir'),f'{source}_long_term_mean_dist_by_{huc_col}_delta_swe_draft1.png'),dpi=350)


def main(daymet_dir, pickles, start_date='1980-10-01', end_date='2020-09-30', huc_col='huc8', **kwargs):
	"""Testing improved definitions of snow drought. Original definitions based on Dierauer et al 2019 but trying to introduce some more nuiance into the approach."""

	################################################################
	#first do the daymet data 
	#read in all the files in this dir and combine them into one df
	early=FormatData(glob.glob(daymet_dir+f'*_12_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	mid=FormatData(glob.glob(daymet_dir+f'*_2_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	late=FormatData(glob.glob(daymet_dir+f'*_4_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	################################################################
	#next do the snotel data 
	output=[]

	#read in some pickled objects, these look like a list of dfs with each being a station for the full time period 
	for item in ['PREC','TAVG','WTEQ']:
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		df=FormatData(files,drop_cols=['year','month','day']).read_in_pickles()
		output.append(df) #the df here is 365 days x ~30 yrs x 237 stations so these are pretty big dfs
	
	#join the three enviro params 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on=['date','id']), output)
	
	
	#convert the temp column from F to C 
	output_df['TAVG'] = (output_df['TAVG']-32)*(5/9) 

	#convert prec and swe cols from inches to cm 
	output_df['PREC'] = output_df['PREC']*2.54
	output_df['WTEQ'] = output_df['WTEQ']*2.54
	
	#remove rows that have one of the data types missing- this might need to be amended because 
	#it means that there are different numbers of records in some of the periods. 
	output_df=output_df.dropna()
	
	#cast the snotel id col to int to add the hucs 
	output_df['id'] = output_df['id'].astype('int')

	#add the as yet nonexistant hucs data to the outputs 
	hucs = kwargs.get('hucs')
	output_df[huc_col] = output_df['id'].map(hucs)

	#there are multiple snotel stations in some of the basins, 
	#combine those so there is just one number per basin like the 
	#daymet and RS data. 

	output_df=output_df.groupby([huc_col,'date'])[['PREC','WTEQ','TAVG']].mean().reset_index()

	period_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
			#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)

		##########working below here
		############################
		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991,sort_col=huc_col).prepare_df_cols()
			#print('snotel')
			#print(snotel_drought)
		else: 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',sort_col=huc_col).prepare_df_cols()

		#get cols of interest 
		#snotel_drought=snotel_drought[['huc8','year','dry','warm','warm_dry']]
		#rename cols so they don't get confused when data are merged 
		#snotel_drought.columns=['huc8','year']+['s_'+column for column in snotel_drought.columns if not (column =='huc8') | (column=='year')]
		
		#then do the same for daymet  
		if (p1 == 'mid') | (p1 == 'late'): 
			daymet_drought=CalcSnowDroughts(p2,start_year=1991,sort_col=huc_col).prepare_df_cols()
		else: 
			daymet_drought=CalcSnowDroughts(p2,sort_col=huc_col).prepare_df_cols()
		
	##########################################
	
		#run the kmeans with drought types as intiilization conditions (centroids) for the clusters
		
		#these are all of the huc 4 basins in the study area 
		huc4s = ['1708','1801','1710','1711','1709','1701','1702','1705','1703','1601','1707','1706','1712','1704']
		s_output = []
		d_output = []
		for huc4 in huc4s: 
			huc4_s = sd.prep_clusters(snotel_drought,huc4,huc_col=huc_col) #get the subset of the snow drought data for a given huc4
			huc4_d = sd.prep_clusters(daymet_drought,huc4,huc_col=huc_col)
			#make the centroids that serve as the intialization for the kmeans clusters- these are like endmembers (ish)
			s_centroids = DefineClusterCenters(huc4_s,'WTEQ','PREC','TAVG').combine_centroids() #makes a numpy array with four centroids
			d_centroids = DefineClusterCenters(huc4_d,'swe','prcp','tavg').combine_centroids() #makes a numpy array with four centroids

			#clusters should be like: {0:dry, 1:warm, 2:warm_dry, 3:no_drought} 6/8/2021 DOUBLE CHECK
			#run kmeans for the snotel data 
			s_clusters = sd.run_kmeans(huc4_s[['WTEQ','PREC','TAVG']].to_numpy(),huc4_s['label'],s_centroids)
			s_clusters = sd.add_drought_cols_to_kmeans_output(s_clusters,huc_col=huc_col) #add a few cols needed for plotting 
			#run kmeans for the daymet data 
			d_clusters = sd.run_kmeans(huc4_d[['swe','prcp','tavg']].to_numpy(),huc4_d['label'],d_centroids)
			d_clusters = sd.add_drought_cols_to_kmeans_output(d_clusters,huc_col=huc_col) #add a few cols needed for plotting 

			s_output.append(s_clusters)
			d_output.append(d_clusters)
		s_plot = pd.concat(s_output)

		# print('plot example: ')
		# print(s_plot)
		#select the cols of interest and rename so there's no confusion when dfs are merged 
		s_plot=s_plot[[huc_col,'year','sqdist','dry', 'warm','warm_dry']]
		s_plot.columns=[huc_col,'year']+['s_'+column for column in s_plot.columns if not (column ==huc_col) | (column=='year')]

		d_plot = pd.concat(d_output)
		d_plot=d_plot[[huc_col,'year','sqdist','dry', 'warm','warm_dry']]
		d_plot.columns=[huc_col,'year']+['d_'+column for column in d_plot.columns if not (column ==huc_col) | (column=='year')]
		# print(s_plot)
		# print(d_plot)
		

		#merge the two datasets into one df 
		dfs = s_plot.merge(d_plot,on=[huc_col,'year'],how='inner')
		period_list.append(dfs)
		# print('final output is: ')
		# print(dfs)
		# print(dfs.columns)

	plot_dist_to_cluster_center(period_list,'snotel',**kwargs)
	#compare each drought type and record the results in a new col 

	# dfs['dry']=dfs['s_dry']==dfs['d_dry']
	# dfs['warm']=dfs['s_warm']==dfs['d_dry']
	# dfs['warm_dry']=dfs['s_warm_dry']==dfs['d_warm_dry']

	#print(dfs.groupby(['huc8'])['dry','warm','warm_dry'].sum())
	# pd.set_option('display.max_columns', None)
	# pd.set_option('display.max_rows', None)
	# #print(dfs)
	

	#####################################################################

	#test robert's unmixing idea 
	#I think this generally works but the issue is that displaying it somehow is hard. Additionally, it still requires some kind of thresholds to define what's what 
	# test_df = snotel_drought.loc[snotel_drought.huc8==16010201]

	# swe_min=test_df['WTEQ'].min()
	# precip_min = test_df['PREC'].min()
	# max_temp = test_df['TAVG'].max()

	# test_df['swe_pct'] = 1-(test_df['WTEQ'].min()/test_df['WTEQ'])

	# print(test_df)

	####################################################################
	#I think this is deprecated 6/12/2021
	
	#print('daymet',daymet_drought)
	#daymet_drought=daymet_drought[['huc8','year','dry','warm','warm_dry']]
	
	# daymet_drought.columns=['huc8','year']+['d_'+column for column in daymet_drought.columns if not (column =='huc8') | (column=='year')]
	# print(daymet_drought)
	#merge the two datasets into one df 
	# dfs = snotel_drought.merge(daymet_drought,on=['huc8','year'],how='inner')
	#print('THe combined output looks like: ', dfs)
	#compare each drought type and record the results in a new col 

	# dfs['dry']=dfs['s_dry']==dfs['d_dry']
	# dfs['warm']=dfs['s_warm']==dfs['d_dry']
	# dfs['warm_dry']=dfs['s_warm_dry']==dfs['d_warm_dry']

	#print(dfs.groupby(['huc8'])['dry','warm','warm_dry'].sum())
	# pd.set_option('display.max_columns', None)
	# pd.set_option('display.max_rows', None)
	# #print(dfs)
	# period_list.append(dfs)
	

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		pickles=variables['pickles']
		stations=variables['stations']
		daymet_dir=variables['daymet_dir']
		palette=variables['palette']	
		pnw_shapefile = variables["pnw_shapefile"]
		huc_shapefile = variables['huc_shapefile']
		us_boundary = variables['us_boundary']
		output_dir=variables['output_dir']
		canada=variables['canada']
		palette = list(palette.values())
	
	hucs=pd.read_csv(stations)

	#change to run for huc8
	#get just the id cols 
	hucs = hucs[['huc_06','id']]
	
	#rename the huc col
	hucs.rename({'huc_06':'huc6'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc6))
	
	main(daymet_dir,pickles,huc_col='huc6',hucs=hucs_dict,palette=palette,pnw_shapefile=pnw_shapefile,huc_shapefile=huc_shapefile,us_boundary=us_boundary,output_dir=output_dir,canada=canada)