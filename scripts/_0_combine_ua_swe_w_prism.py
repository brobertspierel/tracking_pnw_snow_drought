import glob
import os
import sys 
import pandas as pd 
import json


if __name__ == '__main__':
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)        
		#construct variables from param file
		swe = variables['ua_swe']
		prism = variables['prism']
		agg_type = variables['agg_type']
		output_dir = variables['stats_dir']

		#check if the output dir exists, if not create it
		if not os.path.exists(output_dir): 
			os.mkdir(output_dir)
			
		swe_files = glob.glob(swe+'*.csv')
		#print(swe_files)
		prism_files = glob.glob(prism+'*.csv')

		for year in range(1990,2021): #exclusive
			for month_range in [(11,12),(1,2),(3,4)]: 
				#print(f'year is: {year}, month end is: {month_range[1]}')
				#very hardcoded based on output file structures set up in previous scripts 
				ua_file = [f for f in swe_files if (str(year) in f) & (f'{year}_{month_range[0]}_to_{month_range[1]}_months' in f)]
				p_file = [f for f in prism_files if (str(year) in f) & (f'start_month_{month_range[0]}_end_month_{month_range[1]}_WY{year}' in f)] #adjust for new data 
				try: 
					ua_df = pd.read_csv(ua_file[0])
					ua_df.drop(columns=['Unnamed: 0'],inplace=True)
					#make sure prior to merging that date is datetime and huc is int 
					ua_df['date'] = pd.to_datetime(ua_df['date'])
					ua_df[agg_type] = ua_df[agg_type].astype(int)
					print('ua data ')
					print(ua_df.head(10))
				except IndexError: 
					print('There was an issue with the length of the swe data')
				
				try:
					p_df = pd.read_csv(p_file[0])
					p_df.drop(columns=['system:index','.geo'],inplace=True)
					#make sure prior to merging that date is datetime and huc is int 
					p_df['date'] = pd.to_datetime(pd.to_datetime(p_df['date']).dt.date)
					p_df[agg_type] = p_df[agg_type].astype(int)
					print('prism data')
					print(p_df.head(10))
				except IndexError: 
					print('There was an issue with the prism data')
				try: 
					if (len(ua_file) == 1) & (len(p_file) == 1): 
						#merge the two dataframes
						out_df = ua_df.merge(p_df, on=[agg_type,'date'],how='inner')
						print(out_df)
				except IndexError: 
					print('something is wrong with one of your lists')
				try: 
					#create an output fn 
					out_fn = os.path.split(ua_file[0])[1][:-4]
					out_fp = os.path.join(output_dir,out_fn+'_combined.csv')
					if not os.path.exists(out_fp): 
						out_df.to_csv(out_fp)
				except IndexError: 
					print('There was an issue getting the filename for the output')