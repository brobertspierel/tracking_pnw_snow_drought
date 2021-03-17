Plot remote sensing data for different snow drought types
=========================================================

This documents methods for calculating snow drought from remote sensing data and making various plots to support that work. 
This documentation covers functions and methods in the following scripts: 

1. ``_4a_calculate_remote_sensing_snow_droughts.py``
2. ``_4b_calculate_long_term_sp.py``
3. ``_4bv1_calculate_long_term_sp.py``
4. ``_4c_calculate_optical_bias.py``

Example product generation: 

Plot snow covered area (SCA) or snow persistence (SP) for dry, warm, and warm/dry snow droughts. Data for these plots was generated using: ` GEE code <https://code.earthengine.google.com/c4796925848d9d10fca00052651a2b2e>`_

**Code**::
		
		western = ['1708','1801','1710','1711','1709']
		eastern = ['1701','1702','1705','1703','1601','1707','1706','1712','1704']


		def split_basins(input_df,grouping_col,**kwargs): 
			"""Read in a dir of landsat sp data from GEE and make into a new df."""
			input_df[grouping_col] = input_df[grouping_col].astype('str')
			west_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(western))]
			east_df = input_df.loc[input_df[grouping_col].str.contains('|'.join(eastern))]
			
			#replace instances of inf with nan and drop the grouping_col so its not in the mean
			west_df.replace(np.inf,np.nan,inplace=True)
			east_df.replace(np.inf,np.nan,inplace=True)
			
			try: 
				west_df.drop(columns=[grouping_col,'elev_mean'],inplace=True) #added the hardcoded drop of the elev col to clean up for plotting
				east_df.drop(columns=[grouping_col,'elev_mean'],inplace=True)
			except Exception as e: 
				pass
				#print(e)
			# west_df['year'] = kwargs.get('year')
			# east_df['year'] = kwargs.get('year')
			# west_mean = west_df.median(axis=0)
			# east_mean = east_df.median(axis=0)

			return west_df,east_df

		def split_dfs_within_winter_season(df,region,sp=False): 
			"""Splits a single df by date ranges in a winter season."""
			
			early_df = df.loc[(df['date'].dt.month>=11)] 
			mid_df = df.loc[(df['date'].dt.month>=1)&(df['date'].dt.month<=2)]
			late_df = df.loc[(df['date'].dt.month>=3)&(df['date'].dt.month<=4)]

			return {region:[early_df,mid_df,late_df]}


		def merge_dfs(snotel_data,rs_data,drought_type,huc_level='8',col_of_interest='NDSI_Snow_Cover',elev_stat='elev_mean',**kwargs): #added drought_type arg so the drought type is supplied externally 3/15/2021
			"""Merge snotel snow drought data with RS data."""
			
			combined = _4a_rs.create_snow_drought_subset(snotel_data,drought_type,huc_level)
			#merge em 
			combined=combined.merge(rs_data, on=['date',f'huc{huc_level}'], how='inner') #changed rs_df to sentinel data 2/1/2021 to accommodate missing modis data temporarily 
			#get the rs data for the time periods of interest for a snow drought type 
			combined.rename(columns={col_of_interest:f'{drought_type}_{col_of_interest}'},inplace=True)
			#combined = combined.groupby([f'huc{huc_level}', 'date'])[f'{drought_type}_{col_of_interest}'].transform(max) #doesn't really matter which stat (max,min,first) because they are all the same 
			combined = combined.sort_values(f'{drought_type}_{col_of_interest}').drop_duplicates(subset=[f'huc{huc_level}', 'date'], keep='first')

			#print(combined)
			#print(combined[['huc8','date',f'{drought_type}_{col_of_interest}']])
			#check if a couple of args are in kwargs, they can be anything that will evaluate to True
			if 'groupby' in kwargs: 
				rs_df = combined.groupby('date')[f'{drought_type}_{col_of_interest}'].sum().reset_index()
				#dry_rs = dry_combined.groupby('huc'+huc_level)[f'dry_{col_of_interest}',elev_stat].max().reset_index() #changed col from pct change to filter 2/1/2021

				if 'scale_it' in kwargs: 
					scaler = (combined[f'{drought_type}_{col_of_interest}'].count()/rs_data.shape[0])
					rs_df[f'{drought_type}_{col_of_interest}'] = rs_df[f'{drought_type}_{col_of_interest}']*scaler

				return rs_df

			else: 
				return combined

			def combine_rs_snotel_annually(input_dir,season,pickles,agg_step=12,resolution=500,huc_level='8',col_of_interest='NDSI_Snow_Cover',elev_stat='elev_mean',sp=False,total=False,**kwargs): 
				"""Get RS data for snow drought time steps and return those data split by region."""
				
				west_dfs_list = []
				east_dfs_list = []
				years = []
				optical_files = sorted(glob.glob(input_dir+'*.csv'))

				for file in optical_files: 
					year = int(re.findall('(\d{4})-\d{2}-\d{2}', os.path.split(file)[1])[1]) #gets a list with the start and end of the water year, take the second one. expects files to be formatted a specific way from GEE 
					#decide which season length to use depending on the RS aggregation type (SP or SCA)
					if 'SP' in file: 
						snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates_first_day_start'
					elif 'SCA' in file:
						snotel_data = pickles+f'short_term_snow_drought_{year}_water_year_{season}_{agg_step}_day_time_step_w_all_dates'
					else: 
						print('Your file contains neither sp nor SCA, try again')

					input_data = obtain_data.AcquireData(None,file,snotel_data,None,huc_level,resolution)
					
					short_term_snow_drought = input_data.get_snotel_data()
					optical_data = input_data.get_optical_data('NDSI_Snow_Cover')
					optical_data[f'huc{huc_level}'] = pd.to_numeric(optical_data['huc'+huc_level]) 
					optical_data['date'] = _4a_rs.convert_date(optical_data,'date')

					#convert pixel counts to area
					if not sp: 
						optical_data=rs_funcs.convert_pixel_count_sq_km(optical_data,col_of_interest,resolution)

					#optical_data['year'] = optical_data['date'].dt.year

					if not total: 
						#combine the remote sensing and snotel data using the snotel dates of snow droughts to extract rs data 
						merged=merge_dfs(short_term_snow_drought,optical_data,kwargs.get('drought_type')) #snotel_data,rs_data,drought_type
					else: 
						print('Calculating total with no snow droughts')
					#output = split_dfs_within_winter_season
					try: 
						split_dfs=split_basins(merged,f'huc{huc_level}',year=year) #returns the merged df split into two dfs, west (0) and east (1)
					
					except UnboundLocalError as e: 
						split_dfs=split_basins(optical_data,f'huc{huc_level}',year=year) #returns the merged df split into two dfs, west (0) and east (1)
					
					west_dfs_list.append(split_dfs[0])
					east_dfs_list.append(split_dfs[1])
					
				output_west_df = pd.concat(west_dfs_list,ignore_index=True)
				output_east_df = pd.concat(east_dfs_list,ignore_index=True)

				return output_west_df,output_east_df #returns two dfs, one for each region for all the years for one drought type 

Example args to call these functions: ::
		dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='dry',sp=True),sp=True)
		warm_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm',sp=True),sp=True)
		warm_dry_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,drought_type='warm_dry',sp=True),sp=True)
		total_sp = generate_output(combine_rs_snotel_annually(sp_data,'core_winter',pickles,sp=True,total=True),sp=True)

In the example the args are: 
	* sp_data- directory of outputs from GEE code linked above 
	* "core_winter"- the season used to generate short term snotel-based droughts outlined previously. This is a pickle
	* pickles- directory of short term snow drought pickles
	* sp- boolean for snow persistence data. Arg is optional. 
	* total- boolean- is this a snow drought type or just the full dataset. Opptional argument 



.. toctree::
   :maxdepth: 1
   :caption: Included scripts:
