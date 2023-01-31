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
from _1_calculate_revised_snow_drought_new_SWE import FormatData,CalcSnowDroughts
import snow_drought_definition_revision_new_SWE as sd
from snow_drought_definition_revision_new_SWE import DefineClusterCenters
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from scipy.stats import pearsonr
import matplotlib.ticker as ticker
import pymannkendall as mk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def plot_counts(df_list,output_dir,huc_col='huc8',**kwargs): 
	print('Entered the plotting function: ')
	labels=['Snotel','UA SWE']
	
	nrow=3
	ncol=3
	fig,axs = plt.subplots(nrow,ncol,
							sharex=True,sharey=True,
							gridspec_kw={'wspace':0,'hspace':0},
                			figsize=(8,6))
	
	cols=['dry','warm','warm_dry']
	#cols = ['s_'+col for col in cols] + ['ua_'+col for col in cols]
	xlabels=['Dry', 'Warm', 'Warm/dry']
	ylabels=['Early','Mid','Late']
	output_dict = {}
	mk_dict = {}
	count = 0 
	for x in range(3): 
		for y in range(3): 
			print(f'the row is {x} and the col is {y}')
			counts = df_list[x].groupby([huc_col],dropna=False)[['s_'+col for col in cols] + ['ua_'+col for col in cols]].count().reset_index()

			counts[huc_col] = counts[huc_col].astype(int)
			print(counts[['s_'+col for col in cols] + ['ua_'+col for col in cols]].max())
			print(df_list[x].groupby([huc_col])[['s_'+col for col in cols] + ['ua_'+col for col in cols]].size())

			#read in some shapefiles to make the maps 
			hucs_shp = gpd.read_file(huc_shapefile)
			us_bounds = gpd.read_file(us_boundary)
			#pnw_states = gpd.read_file(pnw_shapefile)
			canada = gpd.read_file(kwargs.get("canada"))
			hucs_shp[huc_col] = hucs_shp[huc_col].astype('int32')
			#make a dict like {huc_ID:overall counts for a basin for a sd type}
			s_dict = dict(zip(list(counts[huc_col]),list(counts['s_'+cols[y]])))
			ua_dict = dict(zip(list(counts[huc_col]),list(counts['ua_'+cols[y]])))

			hucs_shp['s_counts'] = hucs_shp[huc_col].map(s_dict)
			hucs_shp['ua_counts'] = hucs_shp[huc_col].map(ua_dict)
			
			colors = ['#ccc596','#e4b047','#D95F0E','#823908']
			bounds = [0,5,10,15,20]
			cmap = ListedColormap(colors)
			minx, miny, maxx, maxy = hucs_shp.geometry.total_bounds

			#plot dry snow drought
			#ax=plt.subplot(gs[x,y])
			canada.plot(ax=axs[x][y],color='#f2f2f2', edgecolor='darkgray',lw=0.5)
			us_bounds.plot(ax=axs[x][y],color='#f2f2f2', edgecolor='darkgray',lw=0.5)
			#pnw_states.plot(ax=axs[x][y],color='#f2f2f2',edgecolor='darkgray',lw=0.5)
			#hucs_shp.plot(ax=ax1,color='gray',edgecolor='darkgray')
			if kwargs.get('dataset').lower() == 'snotel': 
				hucs_shp.plot(ax=axs[x][y],column='s_counts',cmap=cmap,vmin=0,vmax=20)
				print('snotel looks like: ')
				print(hucs_shp)
			elif kwargs.get('dataset').lower() == 'ua_swe': 
				hucs_shp.plot(ax=axs[x][y],column='ua_counts',cmap=cmap,vmin=0,vmax=20)
				print('ua swe looks like')
				print(hucs_shp)
			else: 
				print('That is not a legitimate dataset choice. You can choose ua_swe or snotel only as of 9/23/2021')
			#axs[x][y].set_title('Dry snow drought')
			axs[x][y].set_xlim(minx - 0.5, maxx + 0.5) # added/substracted value is to give some margin around total bounds
			axs[x][y].set_ylim(miny - 0.5, maxy)
			axs[0][y].set_title(xlabels[y],fontdict={'fontsize': 12})
			axs[x][0].set_ylabel(ylabels[x],fontsize=12)
			axs[x][y].set_facecolor('aliceblue') 
	
	#bounds = [1990, 2000, 2010, 2020]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	#fig.subplots_adjust(right=0.2)
	#working
	# cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.72])
	# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
	#updated working
	if kwargs.get('dataset') == 'snotel': 
		cbar_ax = fig.add_axes([0.125, 0.02, 0.775, 0.02]) #like: [left, bottom, width, height]
		fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')

	# #plt.subplots_adjust(hspace=0.0001,wspace=0.0001)
	# plt.savefig(os.path.join(kwargs.get('fig_dir'),f'{kwargs.get("dataset")}_droughts_by_decade_{huc_col}_MAPS_w_delta_swe_proj_draft_1.jpg'), 
	# 	dpi=500,
	# 	bbox_inches = 'tight',
	# 	pad_inches = 0.1
	# 	)
	plt.show()
	plt.close('all')

def main(model_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',huc_col = 'huc8', **kwargs):
	"""Testing improved definitions of snow drought. Original definitions based on Dierauer et al 2019 but trying to introduce some more nuiance into the approach."""
	print(f'The huc col being processed is: {huc_col}')
	################################################################
	#first do the UA swe data - this is now (9/20/2021) in two different files, one from UA SWE and one from PRISM. 
	#These are combined in _0_combine_ua_swe_w_prism.py
	early=FormatData(glob.glob(model_dir+f'*_12_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	mid=FormatData(glob.glob(model_dir+f'*_2_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	late=FormatData(glob.glob(model_dir+f'*_4_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
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
	#there are a couple of erroneous temp values, remove those 
	output_df = output_df.loc[output_df['TAVG'] < 50]
	output_df = output_df.loc[output_df['TAVG'] > -40]

	#convert prec and swe cols from inches to mm 
	output_df['PREC'] = output_df['PREC']*25.4
	output_df['WTEQ'] = output_df['WTEQ']*25.4
	
	#remove rows that have one of the data types missing- this might need to be amended because 
	#it means that there are different numbers of records in some of the periods. 
	output_df=output_df.dropna() #commented out 9/21/2021
	
	#cast the snotel id col to int to add the hucs 
	output_df['id'] = output_df['id'].astype('int')

	#add the as yet nonexistant hucs data to the outputs 
	hucs = kwargs.get('hucs')
	output_df[huc_col] = output_df['id'].map(hucs)

	#there are multiple snotel stations in some of the basins, 
	#combine those so there is just one number per basin like the 
	#ua swe and RS data. 

	output_df=output_df.groupby([huc_col,'date'])[['PREC','WTEQ','TAVG']].mean().reset_index()

	period_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
		#get snotel first
		#make a temporal chunk of data- this is all seasonal windows irrespective of the year 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)
		
		#####
		#####
		#new for UA SWE- there are some basins which extend into Canada but PRISM and UA SWE do not. 
		#as of 9/22/2021 these basins are removed because there are no stats for the area north of the border. 
		snotel_chunk = snotel_chunk.loc[~snotel_chunk[huc_col].isin(kwargs.get('remove_ids'))]
		p2 = p2.loc[~p2[huc_col].isin(kwargs.get('remove_ids'))]

		# print('Snotel chunk looks like: ')
		# pd.set_option("display.max_rows", None, "display.max_columns", None)
		# print(snotel_chunk.head(100))
		##########working below here
		############################
		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			print('processing snotel')
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991,sort_col=huc_col).prepare_df_cols()
		else: 
			print('processing snotel')
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',sort_col=huc_col).prepare_df_cols()
		
		#then do the same for ua SWE  
		if (p1 == 'mid') | (p1 == 'late'):
			print('processing uaswe') 
			ua_swe_drought=CalcSnowDroughts(p2,start_year=1991,sort_col=huc_col).prepare_df_cols()
		else: 
			print('processing uaswe')
			ua_swe_drought=CalcSnowDroughts(p2,sort_col=huc_col).prepare_df_cols()

	##########################################
	
		#run the kmeans with drought types as intiilization conditions (centroids) for the clusters
		
		#these are all of the huc 4 basins in the study area 
		huc4s = ['1708','1801','1710','1711','1709','1701','1702','1705','1703','1601','1707','1706','1712','1704']
		#use a subset to run a test for maritime snowpack
		#huc4s = ['1711','1709','1702','1703','1708','1707','1703','1702']
		#use a subset to run an alpine(ish) snowpack
		#huc4s = ['1704','1706','1601','1705']
		s_output = []
		ua_output = []
		for huc4 in huc4s: 
			huc4_s = sd.prep_clusters(snotel_drought,huc4,p1,huc_col=huc_col) #get the subset of the snow drought data for a given huc4
			huc4_ua = sd.prep_clusters(ua_swe_drought,huc4,p1,huc_col=huc_col) #period added 9/21/2021 to make sure these get attributed to the correct water year
			#make the centroids that serve as the intialization for the kmeans clusters- these are like endmembers (ish)
			s_centroids = DefineClusterCenters(huc4_s,'WTEQ','PREC','TAVG').combine_centroids() #makes a numpy array with four centroids
			ua_centroids = DefineClusterCenters(huc4_ua,'swe','ppt','tmean').combine_centroids() #makes a numpy array with four centroids

			#clusters should be like: {0:dry, 1:warm, 2:warm_dry, 3:no_drought} 6/8/2021 DOUBLE CHECK
			#run kmeans for the snotel data
			s_clusters = sd.run_kmeans(huc4_s[['WTEQ','PREC','TAVG']].to_numpy(),huc4_s['label'],s_centroids)
			s_clusters = sd.add_drought_cols_to_kmeans_output(s_clusters, huc_col=huc_col) #add a few cols needed for plotting 
			#run kmeans for the daymet data 
			ua_clusters = sd.run_kmeans(huc4_ua[['swe','ppt','tmean']].to_numpy(),huc4_ua['label'],ua_centroids)
			ua_clusters = sd.add_drought_cols_to_kmeans_output(ua_clusters, huc_col=huc_col) #add a few cols needed for plotting 

			s_output.append(s_clusters)
			ua_output.append(ua_clusters)
		s_plot = pd.concat(s_output)

		#select the cols of interest and rename so there's no confusion when dfs are merged 
		s_plot=s_plot[[huc_col,'year','dry','warm','warm_dry']]
		s_plot.columns=[huc_col,'year']+['s_'+column for column in s_plot.columns if not (column == huc_col) | (column=='year')]

		ua_plot = pd.concat(ua_output)
		ua_plot=ua_plot[[huc_col,'year','dry','warm','warm_dry']]
		ua_plot.columns=[huc_col,'year']+['ua_'+column for column in ua_plot.columns if not (column == huc_col) | (column=='year')]
	
		#merge the two datasets into one df 
		dfs = s_plot.merge(ua_plot,on=[huc_col,'year'],how='inner')
		# print('The final ds looks like: ')
		# print(dfs.head(50))
		#deal with the scenario that there are basins with less than 30 years of data, remove those here
		dfs = sd.remove_short_dataset_stations(dfs,huc_col)
		print('the output dfs look like: ')
		print(dfs)
		print(f'There are now {len(dfs[huc_col].unique())} unique stations in the seasonal window')
		period_list.append(dfs)

	plot_counts(period_list,kwargs.get('stats_dir'),huc_col=huc_col,**kwargs)

if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		pickles=variables['pickles']
		stations=variables['stations']
		ua_swe_dir=variables['ua_swe_dir']
		stats_dir = variables['stats_dir']
		fig_dir = variables['fig_dir']
		pnw_shapefile = variables["pnw_shapefile"]
		huc_shapefile = variables['huc_shapefile']
		us_boundary = variables['us_boundary']
		canada=variables['canada']
	
	hucs=pd.read_csv(stations)
	
	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))

	#check if the fig dir exists, if not create it 
	if not os.path.exists(fig_dir): 
		os.mkdir(fig_dir)

	remove_ids = [17110004,
				17110001,
				17110005,
				17020006,
				17020002,
				17020001,
				17010216,
				17010215,
				17010104]
	main(ua_swe_dir,pickles,
		huc_col='huc8',
		hucs=hucs_dict,
		stats_dir=stats_dir,
		fig_dir=fig_dir,
		remove_ids=remove_ids, 
		pnw_shapefile=pnw_shapefile,
		huc_shapefile=huc_shapefile,
		us_boundary=us_boundary,
		canada=canada,
		dataset='snotel'
		#test_ids = maritime_test
		)
