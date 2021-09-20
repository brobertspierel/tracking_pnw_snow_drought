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
from _1_calculate_revised_snow_drought import FormatData,CalcSnowDroughts
import snow_drought_definition_revision as sd
from snow_drought_definition_revision import DefineClusterCenters

#supress the SettingWithCopy warning in pandas
pd.options.mode.chained_assignment = None  # default='warn'

def define_snow_drought_recentness(input_df,grouping_col,output_var):
	"""Classify how recently the snow droughts are happening in a given basin."""

	#we want to get periods of 1990-2000, 2001-2010 and 2011-2020

	output_dict = {}
	for item in input_df[grouping_col].unique(): 
		df_subset = input_df[input_df[grouping_col]==item]
		filter_values = pd.IntervalIndex.from_tuples([(1980, 1989), (1990, 1999), (2000, 2009),(2010,2019)],closed='both')
		out = df_subset[['dry','warm','warm_dry']].apply(pd.cut,bins=filter_values)
		counts = out.apply(pd.Series.value_counts)
		if not output_var: #default here is to get the max value which is for recentness plot. If output_var evaluates to True (ie not None) it will not send the max but just the counts 
			counts=counts.idxmax()
		else: 
			pass
		output_dict.update({int(item):counts})

	return output_dict

def main(daymet_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',huc_col = 'huc8', **kwargs):
	"""Testing improved definitions of snow drought. Original definitions based on Dierauer et al 2019 but trying to introduce some more nuiance into the approach."""
	print(f'The huc col being processed is: {huc_col}')
	#pickle these outputs to make the plot troubleshooting easier 
	s_fn = os.path.join(pickles,f'snotel_periods_list_{huc_col}_w_delta_swe_year_limit_updated4.p')
	d_fn = os.path.join(pickles,f'daymet_periods_list_{huc_col}_w_delta_swe_year_limit_updated4.p')
	if not os.path.exists(s_fn): 
		################################################################
		#first do the daymet data 
		#read in all the files in this dir and combine them into one df
		early=FormatData(glob.glob(daymet_dir+f'*_12_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
		mid=FormatData(glob.glob(daymet_dir+f'*_2_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
		late=FormatData(glob.glob(daymet_dir+f'*_4_{huc_col}.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
		################################################################
		#next do the snotel data 
		output=[]
		print('early is: ',early)
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
		#there are a couple of erroneous temp values, remove those 
		output_df = output_df.loc[output_df['TAVG'] <= 50]
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

		snotel_periods = []
		daymet_periods = []
		for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
				#get snotel first
			#make a temporal chunk of data 
			snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)
			print('processing ', p1)
			
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
			#print('daymet',daymet_drought)
			#daymet_drought=daymet_drought[['huc8','year','dry','warm','warm_dry']]
			
			#daymet_drought.columns=['huc8','year']+['d_'+column for column in daymet_drought.columns if not (column =='huc8') | (column=='year')]

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
				s_clusters = sd.add_drought_cols_to_kmeans_output(s_clusters, huc_col=huc_col) #add a few cols needed for plotting 
				#run kmeans for the daymet data 
				d_clusters = sd.run_kmeans(huc4_d[['swe','prcp','tavg']].to_numpy(),huc4_d['label'],d_centroids)
				d_clusters = sd.add_drought_cols_to_kmeans_output(d_clusters, huc_col=huc_col) #add a few cols needed for plotting 

				s_output.append(s_clusters)
				d_output.append(d_clusters)
			s_plot = pd.concat(s_output)

			#select the cols of interest and rename so there's no confusion when dfs are merged 
			s_plot=s_plot[[huc_col,'year','dry','warm','warm_dry']]
			#s_plot.columns=[huc_col,'year']+['s_'+column for column in s_plot.columns if not (column == huc_col) | (column=='year')]

			d_plot = pd.concat(d_output)
			d_plot=d_plot[[huc_col,'year','dry','warm','warm_dry']]
			#d_plot.columns=[huc_col,'year']+['d_'+column for column in d_plot.columns if not (column == huc_col) | (column=='year')]
			#remove stations that have less than the full 30 years of data 
			print('daymet first')
			print(d_plot)
			s_plot = sd.remove_short_dataset_stations(s_plot,huc_col)
			d_plot = sd.remove_short_dataset_stations(d_plot,huc_col)
			print('daymet')
			print(d_plot)
			snotel_periods.append(s_plot)
			daymet_periods.append(d_plot)
		#if they don't already exist, pickle the outputs so this doesn't have to be done every time 
		print('Writing to pickle...')
		pickle.dump( snotel_periods, open( s_fn, "wb" ) )
		pickle.dump( daymet_periods, open( d_fn, "wb" ) )
	else: 
		print('Reading from pickle...')
		snotel_periods = pickle.load( open( s_fn, "rb" ) )
		daymet_periods = pickle.load( open( d_fn, "rb" ) )
		
	

	###############################################################

	#plot_func the long term 'recentness' of snotel snow drought
	print('making plot...')
	#read in shapefiles
	hucs_shp = gpd.read_file(huc_shapefile)
	us_bounds = gpd.read_file(us_boundary)
	pnw_states = gpd.read_file(pnw_shapefile)
	canada = gpd.read_file(kwargs.get("canada"))
	hucs_shp[huc_col] = hucs_shp[huc_col].astype('int32')
	nrow=3
	ncol=3
	#define the overall figure- numbers are hardcoded
	fig,axs = plt.subplots(nrow,ncol,
							sharex=True,
							sharey=True,
							figsize=(8,6),
							gridspec_kw={'wspace':0,'hspace':0}
							)
	#adjust font size
	#plt.rcParams.update({'font.size': 16})

	# s_colors = ['#ccc596','#e4b047','#D95F0E']
	# d_colors = ['#d4cfd9','#95aac5','#267eab']
	colors = ['#ccc596','#e4b047','#D95F0E']
	#iterate down rows (each row is a time period)
	for x in range(3): 

		#get counts for each of the drought types for the time period 
		#print(snotel_periods[x])
		if kwargs.get('dataset') == 'daymet': 
			recentness=define_snow_drought_recentness(daymet_periods[x],huc_col,None)
		elif kwargs.get('dataset') == 'snotel': 
			recentness=define_snow_drought_recentness(snotel_periods[x],huc_col,None)	
		dry = {}
		warm = {}
		warm_dry = {}


		for k,v in recentness.items(): 
			try: 
				#get the start year of the decade for each drought type 
				dry.update({k:int(v.at['dry'].left)})
				warm.update({k:int(v.at['warm'].left)})
				warm_dry.update({k:int(v.at['warm_dry'].left)})
			except Exception as e: 
				print(f'The error here was {e} and is likely the result of not running in max mode above.')
				raise


		#pd.set_option("display.max_rows", None, "display.max_columns", None) #change to print an entire df

		hucs_shp['dry_colors'] = hucs_shp[huc_col].map(dry)
		hucs_shp['warm_colors'] = hucs_shp[huc_col].map(warm)
		hucs_shp['warm_dry_colors'] = hucs_shp[huc_col].map(warm_dry)
		
		#print(hucs_shp)

		#iterate through cols (across a row) with each col being a drought type. 
		plot_cols=['dry_colors','warm_colors','warm_dry_colors']
		xlabels=['Dry', 'Warm', 'Warm/dry']
		ylabels=['Early','Mid','Late']

		#just export some data
		export_df = hucs_shp[[huc_col,'dry_colors','warm_colors','warm_dry_colors']]
		for i in [1990,2000,2010]: 
			export_df[f'dry_{i}'] = np.where(export_df['dry_colors'] == i,1,0)
			export_df[f'warm_{i}'] = np.where(export_df['warm_colors'] == i,1,0)
			export_df[f'warm_dry_{i}'] = np.where(export_df['warm_dry_colors'] == i,1,0)
		csv_fn = os.path.join(kwargs.get('stats_dir'),f'{ylabels[x]}_season_{kwargs.get("dataset")}_most_prevalent_decades_by_sd_type_{huc_col}_revised_w_delta_swe_draft2.csv') 
		if not os.path.exists(csv_fn): 
			export_df.to_csv(csv_fn)

		for y in range(3): 
			bounds = [1990,2000,2010,2020]
			cmap = ListedColormap(colors)
			minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds

			#plot dry snow drought
			#ax=plt.subplot(gs[x,y])
			canada.plot(ax=axs[x][y],color='#f2f2f2', edgecolor='darkgray',lw=0.5)
			us_bounds.plot(ax=axs[x][y],color='#f2f2f2', edgecolor='black',lw=0.5)
			pnw_states.plot(ax=axs[x][y],color='#f2f2f2',edgecolor='darkgray',lw=0.5)
			#hucs_shp.plot(ax=ax1,color='gray',edgecolor='darkgray')
			hucs_shp.plot(ax=axs[x][y],column=plot_cols[y],cmap=cmap,vmin=1990,vmax=2020)#, column='Value1')
			#axs[x][y].set_title('Dry snow drought')
			axs[x][y].set_xlim(minx - 1, maxx + 1) # added/substracted value is to give some margin around total bounds
			axs[x][y].set_ylim(miny - 1, maxy + 1)
			axs[0][y].set_title(xlabels[y],fontdict={'fontsize': 12})
			axs[x][0].set_ylabel(ylabels[x],fontsize=12)
			axs[x][y].set_facecolor('aliceblue') 
	#from another script 
	# 	cax = fig.add_axes([0.9, 0.125, 0.03, 0.75]) #formatted like [left, bottom, width, height]
	# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	# fig.colorbar(sm, cax=cax)

	bounds = [1990, 2000, 2010, 2020]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	#fig.subplots_adjust(right=0.2)
	#working
	# cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.72])
	# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
	#updated working
	if kwargs.get('dataset') == 'snotel': 
		cbar_ax = fig.add_axes([0.125, 0.02, 0.775, 0.02]) 
		fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')

	#plt.subplots_adjust(hspace=0.0001,wspace=0.0001)
	plt.savefig(os.path.join(kwargs.get('fig_dir'),f'{kwargs.get("dataset")}_droughts_by_decade_{huc_col}_w_delta_swe_revised_draft_3.jpg'), 
		dpi=500,
		bbox_inches = 'tight',
		pad_inches = 0.1
		)
	# plt.show()
	# plt.close('all')

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
		fig_dir = variables['fig_dir']
		stats_dir = variables['stats_dir']
		canada=variables['canada']
		palette = list(palette.values())

hucs=pd.read_csv(stations)

#get just the id cols 
hucs = hucs[['huc_06','id']]

#rename the huc col
hucs.rename({'huc_06':'huc6'},axis=1,inplace=True)

hucs_dict=dict(zip(hucs.id,hucs['huc6']))

main(daymet_dir,pickles,huc_col='huc6',hucs=hucs_dict,palette=palette,stations=stations,
	pnw_shapefile=pnw_shapefile,huc_shapefile=huc_shapefile,us_boundary=us_boundary,output_dir=output_dir,
	canada=canada,fig_dir=fig_dir,dataset='daymet',stats_dir=stats_dir)