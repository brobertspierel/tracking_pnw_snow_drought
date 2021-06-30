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
from _1_calculate_snow_droughts_mult_sources import FormatData,CalcSnowDroughts
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from matplotlib import gridspec

def prepare_snotel_data(pickles,start_date='1980-10-01',end_date='2020-09-30',**kwargs): 
	output=[]
	for item in ['PREC','TAVG','WTEQ']:
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		df=FormatData(files,drop_cols=['year','month','day']).read_in_pickles()
		output.append(df)
	
	#join the three enviro params 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on=['date','id']), output)
	#print(output_df)
	
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
	output_df['huc8'] = output_df['id'].map(hucs)

	#there are multiple snotel stations in some of the basins, 
	#combine those so there is just one number per basin like the 
	#daymet and RS data. 

	output_df=output_df.groupby(['huc8','date'])['PREC','WTEQ','TAVG'].mean().reset_index()
	#print(output_df)

	return output_df

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



def main(daymet_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',**kwargs):
	
	################################################################
	#first do the daymet data 
	#read in all the files in this dir and combine them into one df
	early=FormatData(glob.glob(daymet_dir+'*_12_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	mid=FormatData(glob.glob(daymet_dir+'*_2_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	late=FormatData(glob.glob(daymet_dir+'*_4_huc8.csv'),drop_cols=['system:index','.geo','dayl','vp']).read_in_csvs()
	#print('MID WINTER',mid[['date','huc8','swe']].head(50))
	#for period in [early,mid,late]: 
		
		#calculate snow droughts for each period 

	################################################################
	#next do the snotel data 
	
	snotel_df=prepare_snotel_data(pickles,start_date,end_date,**kwargs)

	#read in some pickled objects, these look like a list of dfs with each being a station for the full time period 
	snotel_periods=[]
	daymet_periods=[]
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
			#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(snotel_df)
		
		#calculate the snow droughts for that chunk 
		if (p1 == 'mid') | (p1 == 'late'): 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG',start_year=1991).calculate_snow_droughts()
			snotel_periods.append(snotel_drought)
			
		else: 
			snotel_drought=CalcSnowDroughts(snotel_chunk,swe_c='WTEQ',precip='PREC',temp='TAVG').calculate_snow_droughts()
			snotel_periods.append(snotel_drought)

		#then do the same for daymet  
		if (p1 == 'mid') | (p1 == 'late'): 
			daymet_drought=CalcSnowDroughts(p2,start_year=1991).calculate_snow_droughts()
			daymet_periods.append(daymet_drought)
		else: 
			daymet_drought=CalcSnowDroughts(p2).calculate_snow_droughts()
			daymet_periods.append(daymet_drought)

	# # ###############################################################

	#plot_func the long term 'recentness' of snotel snow drought

	#read in shapefiles
	hucs_shp = gpd.read_file(huc_shapefile)
	us_bounds = gpd.read_file(us_boundary)
	pnw_states = gpd.read_file(pnw_shapefile)
	canada = gpd.read_file(kwargs.get("canada"))
	hucs_shp['huc8'] = hucs_shp['huc8'].astype('int32')
	nrow=3
	ncol=3
	#define the overall figure- numbers are hardcoded
	fig,axs = plt.subplots(nrow,ncol,sharex=True,sharey=True,gridspec_kw={'wspace':0,'hspace':0,
                                    'top':0.95, 'bottom':0.05, 'left':0.05, 'right':0.95},
                       figsize=(nrow*3,ncol*2))  # <- adjust figsize but keep ratio n/m#constrained_layout=True,gridspec_kw=dict(wspace=0.0, hspace=0.0))#figsize=(ncol + 1, nrow + 1),gridspec_kw=dict(wspace=0.0, hspace=0.0))
	
	#adjust font size
	#plt.rcParams.update({'font.size': 16})

	s_colors = ['#ccc596','#e4b047','#D95F0E']
	d_colors = ['#d4cfd9','#95aac5','#267eab']
	#iterate down rows (each row is a time period)
	for x in range(3): 
	
		#get counts for each of the drought types for the time period 
		recentness=define_snow_drought_recentness(snotel_periods[x],'huc8',None)
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

		hucs_shp['dry_colors'] = hucs_shp.huc8.map(dry)
		hucs_shp['warm_colors'] = hucs_shp.huc8.map(warm)
		hucs_shp['warm_dry_colors'] = hucs_shp.huc8.map(warm_dry)
		
		#print(hucs_shp)

		#iterate through cols (across a row) with each col being a drought type. 
		plot_cols=['dry_colors','warm_colors','warm_dry_colors']
		xlabels=['Dry', 'Warm', 'Warm/dry']
		ylabels=['Early','Mid','Late']

		#just export some data
		# export_df = hucs_shp[['huc8','dry_colors','warm_colors','warm_dry_colors']]
		# for i in [1990,2000,2010]: 
		# 	export_df[f'dry_{i}'] = np.where(export_df['dry_colors'] == i,1,0)
		# 	export_df[f'warm_{i}'] = np.where(export_df['warm_colors'] == i,1,0)
		# 	export_df[f'warm_dry_{i}'] = np.where(export_df['warm_dry_colors'] == i,1,0)
		# csv_fp = os.path.join(kwargs.get('output_dir'),f'{ylabels[x]}_season_daymet_most_prevalent_decades_by_sd_type_binary_max.csv') 
		# export_df.to_csv(csv_fp)


		for y in range(3): 
			bounds = [1990,2000,2010,2020]
			cmap = ListedColormap(s_colors)
			#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True,figsize=(18,14))
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
			axs[0][y].set_title(xlabels[y],fontdict={'fontsize': 14})
			axs[x][0].set_ylabel(ylabels[x],fontsize=14)
			

	bounds = [1990, 2000, 2010, 2020]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	# cb2 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
	#                                 norm=norm,
	#                                 boundaries=bounds,#[0] + bounds + [13]
	#                                 ticks=bounds,
	#                                 spacing='proportional',
	#                                 orientation='vertical')
	
	# #turn off the box in the fourth plot. This is where we'll put the caption. Might want to annotate it in here as well. 
	# ax4.axis('off')
	#plt.gca().set_axis_off()
	fig.subplots_adjust(right=0.6)
	cbar_ax = fig.add_axes([0.94, 0.2, 0.02, 0.7])
	fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
	

	# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
	# plt.colorbar(cax=cax)
	#plt.savefig(os.path.join(output_dir,'daymet_recentness_draft2.png'),bbox_inches='tight',pad_inches=, dpi=300)
	plt.show()
	plt.close('all')
	
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

	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	main(daymet_dir,pickles,hucs=hucs_dict,palette=palette,stations=stations,
		pnw_shapefile=pnw_shapefile,huc_shapefile=huc_shapefile,us_boundary=us_boundary,output_dir=output_dir,canada=canada)