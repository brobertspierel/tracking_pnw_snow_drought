import os 
import sys 
import pandas as pd 
import matplotlib.pyplot as plt 
import json 
import glob
from _1_calculate_revised_snow_drought_new_SWE import FormatData,CalcSnowDroughts
import snow_drought_definition_revision_new_SWE as sd
from snow_drought_definition_revision_new_SWE import DefineClusterCenters
from scipy.stats import pearsonr
from functools import reduce
from scipy import stats
import numpy as np 
import matplotlib as mpl
from scipy.stats import kde

"""
Make a comparison of the predictor variables (temp, precip and swe). This is based on a station (SNOTEL) to grid cell comparison. 
There are obviously issues with that because of the disparity in spatial area but we include it as a logical check on the data 
going into the snow drought classifications
Inputs- make sure that the data you are inputting for UA swe is for the gridcell, not an aggregation over the hucX level. 
That is the ua_swe_dir argument in the params file. 
"""

def plot_daily_pt_based_comparison(snotel, ua_swe, **kwargs): #these will come in as lists like: [early, mid, late]
	"""Make regression or scatter plots of the variables used in defining snow droughts for each section of the winter."""
	cols = ['swe','ppt','tmean'] #these are hardcoded and assume you've changed the swe cols to match the UA SWE cols 

	fig,axs = plt.subplots(3,3,figsize=(8,6),
							sharex='row',
							sharey='row',
							gridspec_kw={'wspace':0.17,'hspace':0.25})
	count = 0 
	for x in range(3): #iterate through the cols
		#get a merged df of the two input datasets for that subset of the season 
		df = snotel[x].merge(ua_swe[x],on=['site_num','date'],how='inner')
		#apply some logical limits 
		cols_plus = cols + ['s_'+col for col in cols]
		df = df[cols_plus]
		print(type(df))
		#df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
		
		df = df.loc[(df['tmean']<=40) & (df['s_tmean']<=40)] 

		for y in range(3): #iterate through the rows
			
			# df=df.loc[np.abs(df[cols[y]]-df[cols[y]].mean()) <= (3*df[cols[y]].std())]
			# df=df.loc[np.abs(df[f's_{cols[y]}']-df[f's_{cols[y]}'].mean()) <= (3*df[f's_{cols[y]}'].std())]
			# print(df)
			# print(df.shape)
			print(f'the plot is: [{y}][{x}]')

			print('col is: ',cols[y])
			print('snotel')
			print(df[f's_{cols[y]}'].mean())
			print('ua_swe')
			print(df[cols[y]].mean()
				)
			
			#set the vmin, vmax depending on the row (swe, precip, temp)
			if y == 0: 
				vmin = 0
				vmax = 3000
			elif y == 1: 
				vmin = 0
				vmax = 350
			elif y == 2: 
				vmin = -40
				vmax = 40
			# print('the vmin and vmax are: ',vmin,' ',vmax)
			# print('the data look like: ')
			# print(df[cols[y]])
			
			# xnp = df[cols[y]]
			# ynp = df[f's_{cols[y]}']
			# nbins = 50

			# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
			# k = kde.gaussian_kde(df[[cols[y],f's_{cols[y]}']].to_numpy().T)
			# xi, yi = np.mgrid[xnp.min():xnp.max():nbins*1j, ynp.min():ynp.max():nbins*1j]
			# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
			 
			# # plot a density
			# #axs[3].set_title('Calculate Gaussian KDE')
			# axs[y][x].pcolormesh(xi, 
			# 					yi, 
			# 					zi.reshape(xi.shape), 
			# 					shading='auto', 
			# 					norm=mpl.colors.LogNorm(),
			# 					cmap=mpl.cm.gray)



			axs[y][x].hist2d(df[cols[y]],
							df[f's_{cols[y]}'],
							bins=100,
							range=[[vmin,vmax],[vmin,vmax]],
							norm=mpl.colors.LogNorm(), 
							cmap=mpl.cm.gray 
							)
			#10/22/2021 trying a 2d hist plot instead of a scatter so there's more of a heatmap vibe
			# axs[y][x].scatter(df[cols[y]],
			# 					df[f's_{cols[y]}'],
			# 					s=50,
			# 					facecolors='None',
			# 					edgecolors='black',
			# 					alpha=0.25, 
			# 					vmin=vmin, 
			# 					vmax=vmax
			# 					) 
			axs[y][x].set_xlim(vmin,vmax) # added/substracted value is to give some margin around total bounds
			axs[y][x].set_ylim(vmin,vmax)
			#add pearson correlation 
			corr, _ = pearsonr(df[cols[y]],df[f's_{cols[y]}'])
			rho, pval = stats.spearmanr(df[cols[y]], df[f's_{cols[y]}'])
			#add the spearman or pearson values
			axs[y][x].annotate(f'r = {round(corr,2)}',xy=(0.05,0.85),xycoords='axes fraction',fontsize=10)
			#add a letter identifier 
			axs[y][x].annotate(f'{chr(97+count)}',xy=(0.85,0.1),xycoords='axes fraction',fontsize=10,weight='bold')

			count += 1
	axs[0][0].set_title('Early',fontdict={'fontsize':'medium'})
	axs[0][1].set_title('Mid',fontdict={'fontsize':'medium'})
	axs[0][2].set_title('Late',fontdict={'fontsize':'medium'})
	axs[0][0].set_ylabel('SWE (mm)',fontsize=10)
	axs[1][0].set_ylabel('Daily precip (mm)',fontsize=10)
	axs[2][0].set_ylabel('Tavg (deg C)',fontsize=10)

	fig.text(0.5, 0.04, 'UA SWE', ha='center',fontsize=10)
	fig.text(0.025, 0.5, 'SNOTEL', va='center', rotation='vertical',fontsize=10)

	# plt.show()
	# plt.close('all')

	fig_fn = os.path.join(kwargs.get('fig_dir'),'snotel_ua_swe_pred_var_comparison_no_SD_designation_pearson_adjusted_heatmap.jpg')
	if not os.path.exists(fig_fn): 
		plt.savefig(fig_fn,
			dpi=500, 
			bbox_inches = 'tight',
	    	pad_inches = 0
			)
	else: 
		print(f'The file {fig_fn} already exists')

def main(model_dir,pickles,start_date='1980-10-01',end_date='2020-09-30',huc_col = 'site_num', **kwargs):
	"""Compare the agreement of predictor variables from Daymet and SNOTEL used for classifying snow droughts"""
	
	#check if the fig output dir exists, if not make it
	if not os.path.exists(kwargs.get('fig_dir')): 
		os.mkdir(kwargs.get('fig_dir'))

	print(f'The huc col being processed is: {huc_col}')
	################################################################
	#first do the daymet data 
	#read in all the files in this dir and combine them into one df
	early=FormatData(glob.glob(model_dir+f'*_12_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	mid=FormatData(glob.glob(model_dir+f'*_2_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	late=FormatData(glob.glob(model_dir+f'*_4_months_data_output_formatted_combined.csv'),drop_cols=[]).read_in_csvs()
	################################################################

	#next do the snotel data 
	output=[]

	#read in some pickled objects, these look like a list of dfs with each being a station for the full time period 
	for item in ['PRCP','TAVG','WTEQ']: #note that PREC is accumulated precip and PRCP is the daily precip- 
		#we use this one to match with the Daymet format and avoid issues with accumulated error in the cumulative data. 
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		df=FormatData(files,drop_cols=['year','month','day']).read_in_pickles()
		output.append(df) #the df here is 365 days x ~30 yrs x 237 stations so these are pretty big dfs
	
	#join the three enviro params 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on=['date','id']), output)
	#convert the temp column from F to C 
	output_df['TAVG'] = (output_df['TAVG']-32)*(5/9) 
	#convert prec and swe cols from inches to mm - note that elsewhere this is cm but daymet is mm. Make sure these are homogenized
	
	output_df['PRCP'] = output_df['PRCP']*25.4
	output_df['WTEQ'] = output_df['WTEQ']*25.4
	#remove rows that have one of the data types missing- this might need to be amended because 
	#it means that there are different numbers of records in some of the periods. 
	output_df=output_df.dropna()
	
	#cast the snotel id col to int to add the hucs 
	output_df['id'] = output_df['id'].astype('int')
	output_df['date'] = pd.to_datetime(output_df['date'])

	#add the as yet nonexistant hucs data to the outputs 
	hucs = kwargs.get('hucs')
	output_df['huc8'] = output_df['id'].map(hucs)
	#get only the snotel stations that are being used in the other scripts
	#this should allow for multiple stations in the same basin  
	output_df = output_df.loc[output_df['huc8'].isin(kwargs.get('basins'))] #note that this is hardcoded because we need the snotel col otherwise

	snotel_list = []
	ua_list = []
	for p1,p2 in zip(['early','mid','late'],[early,mid,late]): 
			#get snotel first
		#make a temporal chunk of data 
		snotel_chunk=FormatData(None,time_period=p1).split_yearly_data(output_df)

		#rename the snotel cols to match ua_swe and add s to snotel so they're not confused after merging
		snotel_chunk.rename(columns={'WTEQ':'s_swe','PRCP':'s_ppt','TAVG':'s_tmean','id':'site_num'},inplace=True) 

		#new for UA SWE- there are some basins which extend into Canada but PRISM and UA SWE do not. 
		#as of 9/22/2021 these basins are removed because there are no stats for the area north of the border. 
		snotel_chunk = snotel_chunk.loc[~snotel_chunk[huc_col].isin(kwargs.get('remove_ids'))]
		p2 = p2.loc[~p2[huc_col].isin(kwargs.get('remove_ids'))]

		
		snotel_list.append(snotel_chunk)
		ua_list.append(p2)

	plot_daily_pt_based_comparison(snotel_list,ua_list,**kwargs)
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

		hucs=pd.read_csv(stations)
	
		#get just the id cols 
		hucs = hucs[['huc_08','id']]
		print('hucs shape is: ')
		print(hucs.shape)
		#rename the huc col
		hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
		hucs_dict=dict(zip(hucs.id,hucs.huc8))

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
		#using this list of ids to make sure we're just getting the basins used in the other scripts that have 30 years of data 
		basins_oi = [17080001, 17080002, 17080004, 17080005, 18010201, 18010202, 18010203, 18010204, 
		17100301, 17100302, 17100307, 17100309, 17110009, 17110010, 17110013, 17110014, 
		17110015, 17110018, 17090001, 17090002, 17090004, 17090005, 17090006, 17090010, 
		17090011, 17010205, 17010213, 17010301, 17010302, 17010304, 17010308, 17020009, 
		17020010, 17020011, 17050104, 17050108, 17050111, 17050113, 17050116, 17050120, 
		17050124, 17050201, 17050202, 17050203, 17030001, 17030002, 17030003, 16010201, 
		16010202, 16010204, 17070102, 17070103, 17070105, 17070202, 17070204, 17070301, 
		17070302, 17070304, 17070305, 17070306, 17060104, 17060105, 17060201, 17060203, 
		17060204, 17060208, 17060210, 17060302, 17060303, 17060306, 17060307, 17060308, 
		17120002, 17120003, 17120005, 17040202, 17040204, 17040205, 17040207, 17040208, 
		17040209, 17040212, 17040213, 17040214, 17040217, 17040218, 17040219, 17040220, 
		17040221]

	main(ua_swe_dir,pickles,
		stats_dir=stats_dir,
		fig_dir=fig_dir, 
		remove_ids=remove_ids, 
		# huc_col='huc8',
		hucs=hucs_dict, 
		basins=basins_oi
		)

# data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
# x, y = data.T
# print(data.T)
# print(x)
# print(y)