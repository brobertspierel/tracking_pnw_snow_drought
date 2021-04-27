import pickle 
import glob 
import pandas as pd
import numpy as np 
import os 
import sys 
import json
from functools import reduce
from daymet_snow_drought_calcs_GEE import split_yearly_data, calculate_snow_drought


def read_in_pickles(files,drop_cols): 
	lists=[]
	for file in files: 
		lists.append(pd.read_pickle(file))
	flat_list = [item for sublist in lists for item in sublist]
	output=pd.concat(flat_list,ignore_index=True)

	#make sure the date col is cast as a date type 
	#output['date']= pd.to_datetime(df['date'])

	try: 
		output=output.drop(columns=drop_cols,axis=1) 
		return output
	except AttributeError as e: 
		return output


test= "/vol/v1/general_files/user_files/ben/pickles/ID_PREC_1980-10-01_2020-09-30_snotel_data_list"

def main(pickles,drop_cols=None,start_date='1980-10-01',end_date='2020-09-30',**kwargs): 
	output=[]

	#read in some pickled objects, these look like a list of dfs with each being a station for the full time period 
	for item in ['PREC','WTEQ','TAVG']:
		#get the pickled objects for each parameter  
		files = glob.glob(pickles+f'*{item}_{start_date}_{end_date}_snotel_data_list') #hardcoded currently
		
		df=read_in_pickles(files,drop_cols)
		output.append(df)
	#join the three enviro params 
	output_df = reduce(lambda left,right: pd.merge(left,right,how='left',on=['date','id']), output)
	output_df['id'] = output_df['id'].astype('int')
	#add the as yet nonexistant hucs data to the outputs 
	hucs = kwargs.get('hucs')
	output_df['huc8'] = output_df['id'].map(hucs)

	for period in ['early','mid','late']: 
		
		#make a temporal chunk of data 
		chunk=split_yearly_data(output_df,period)
		#calculate the snow droughts for that chunk 
		droughts=calculate_snow_drought(chunk,swe='WTEQ',precip='PREC',temp='TAVG')

if __name__ == '__main__':
	
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		#construct variables from param file
		pickles=variables['pickles']
		stations=variables['stations']
	hucs=pd.read_csv(stations)

	#get just the id cols 
	hucs = hucs[['huc_08','id']]
	
	#rename the huc col
	hucs.rename({'huc_08':'huc8'},axis=1,inplace=True)
	
	hucs_dict=dict(zip(hucs.id,hucs.huc8))
	
	main(pickles,drop_cols=['year','month','day'],hucs=hucs_dict)
	#test= r"/vol/v1/general_files/user_files/ben/pickles/ID_PREC_1980-10-01_2020-09-30_snotel_data_list"
