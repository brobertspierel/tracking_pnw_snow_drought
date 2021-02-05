import os 
import sys
import pandas as pd 
import numpy as np 
import json

def main(): 

	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		stations = variables["stations"]
		stations_no_hucs = variables["stations_no_hucs"]
		output_dir = variables["output_dir"]

		hucs = pd.read_csv(stations)
		only_stations = pd.read_csv(stations_no_hucs)

		print(hucs.shape)
		print(only_stations.shape)	
		only_stations.rename(columns={'site_num':'id'},inplace=True)
		hucs.drop(columns=['.geo'],inplace=True)
		df = hucs.merge(only_stations,on=['id'],how='inner')
		print(df.shape)
		df.to_csv(output_dir+'NWCC_high_resolution_stations_w_GEE_generated_huc_ids.csv')


if __name__ == '__main__':
    main()