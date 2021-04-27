import pandas as pd 
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt 
import geopandas as gpd 
import json 
import glob
import datetime

# class MakeDataReadable(): 
# 	def __init__(self,data_dir): 
		# self.


def read_data(input_files): 
	files = []
	for file in input_files: 
		df = pd.read_csv(file)
		files.append(df)

	return pd.concat(files)

def split_yearly_data(df,time_period,date_col='date'): 
	"""Make chunks of data from a year."""
	#cast the date col to date 
	while True: 
		try: 
			df[date_col]= pd.to_datetime(df[date_col])
			break
		except KeyError as e:
			print('Your date column has a different name than date. The columns in this df are: ')
			print(df.columns)
			date_col = input('Which one would you like to use?')

	while True: 
		if time_period.lower() == 'early': 
			output = df.loc[(df[date_col].dt.month>=11)&(df[date_col].dt.month<=12)]
			break 
		elif time_period.lower() == 'mid': 
			output = df.loc[(df[date_col].dt.month>=1)&(df[date_col].dt.month<=2)]
			break
		elif time_period.lower() == 'late': 
			output = df.loc[(df[date_col].dt.month>=3)&(df[date_col].dt.month<=4)]
			break 
		else: 
			time_period = input('Your choice of time_period is invalid.\nYou can choose one of early, mid, or late. \nType your choice and hit enter.')
	return output

def calculate_snow_drought(input_df,swe='swe',precip='prcp',temp='tavg'):

	#print(input_df.head())
	print(input_df)
	#add a year col for the annual ones 
	# input_df['year'] = input_df['date'].dt.year
	#print(input_df.head())

	#get agg stats for swe and precip
	swe_prcp = input_df.groupby(['huc8','date']).agg({swe:'max',precip:'sum'})
	
	#get agg stats for temp, this is a little harder because its a degree day model 
	temp = input_df.loc[input_df[temp]>0].groupby(['huc8','year'])[temp].count().reset_index()

	#combine the three vars 
	sd_df = swe_prcp.merge(temp,how='inner',on=['huc8','date'])
	#print(sd_df)
	#get the long-term means 
	# means = sd_df.groupby('huc8').agg({swe:'mean',precip:'mean',temp:'mean'})
	
	# #rename the means cols so when they merge they have distinct names 
	# means.columns = ['mean_'+column for column in means.columns]
	# #print(means)
	
	# #merge the means with the summary stats for each year/basin- this can be split for the three processing periods 
	# sd_df = sd_df.merge(means,how='inner',on='huc8')

	# print('finally')
	# print(sd_df)

	# #define some stats for if/else to get snow droughts.
	# #do dry first
	# sd_df['dry'] = np.where((sd_df[swe]<sd_df[f'mean_{swe}']) & 
	# 	(sd_df[precip]<sd_df[f'mean_{precip}'])&(sd_df[temp]<=sd_df[f'mean_{temp}']),sd_df['year'],np.nan)
	
	# #next do warm
	# sd_df['warm'] = np.where((sd_df[swe]<sd_df[f'mean_{swe}']) & 
	# 	(sd_df[precip]>=sd_df[f'mean_{precip}']),sd_df['year'],np.nan)
	
	# #then do warm/dry 
	# sd_df['warm_dry'] = np.where((sd_df[swe]<sd_df[f'mean_{swe}']) & 
	# 	(sd_df[precip]<sd_df[f'mean_{precip}'])&(sd_df[temp]>sd_df[f'mean_{temp}']),sd_df['year'],np.nan)



	# print(sd_df)

	# # print(sd_df['dry'].value_counts())
	# # print(sd_df['warm'].value_counts())
	# # print(sd_df['warm_dry'].value_counts())

	# #input_df['dry'] = np.where((input_df['swe']<10), 1, 0)

	# return sd_df
	#input_df['warm']=
	#input_df['warm_dry']=


# 	if (v[f'WTEQ_{year}'].max() < v[f'stat_WTEQ'][0]) and (precip_df[f'PREC_{year}'].max()<precip_df['stat_PREC'][0]) and ((temp_df[f'TAVG_{year}'][temp_df[f'TAVG_{year}']>0].count())<temp_df['stat_TAVG'][0]): #(temp_df[f'TAVG_{year}'].mean()<temp_df[f'stat_TAVG'][0]):
# 				#dry.append(year)#dry.update({station_id:year})
# 				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':year,'warm':np.nan,'warm_dry':np.nan},ignore_index=True)

# 			elif (v[f'WTEQ_{year}'].max() < v[f'stat_WTEQ'][0]) and (precip_df[f'PREC_{year}'].max()>=precip_df['stat_PREC'][0]): 
# 				#warm.append(year)#warm.update({station_id:year})
# 				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':year,'warm_dry':np.nan},ignore_index=True)

# 			elif (v[f'WTEQ_{year}'].max() < v[f'stat_WTEQ'][0]) and (precip_df[f'PREC_{year}'].max()<precip_df['stat_PREC'][0]) and ((temp_df[f'TAVG_{year}'][temp_df[f'TAVG_{year}']>0].count())>temp_df['stat_TAVG'][0]): 
# 				#warm_dry.append(year)#warm_dry.update({station_id:year})
# 				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':np.nan,'warm_dry':year},ignore_index=True)

# 			else: 
# 				pass
# 				print(f'station {k} for {year} was normal or above average. The swe value was: {v[f"WTEQ_{year}"].max()} and the long term mean max value was: {v[f"stat_WTEQ"][0]}')
# 				output_df=output_df.append({'station_id':k,'huc_id':hucs[k],'dry':np.nan,'warm':np.nan,'warm_dry':np.nan},ignore_index=True)
# 	# create a list of our conditions
# conditions = [
#     (df['likes_count'] <= 2),
#     (df['likes_count'] > 2) & (df['likes_count'] <= 9),
#     (df['likes_count'] > 9) & (df['likes_count'] <= 15),
#     (df['likes_count'] > 15)
#     ]

# # create a list of the values we want to assign for each condition
# values = ['tier_4', 'tier_3', 'tier_2', 'tier_1']

# # create a new column and use np.select to assign values to it using our lists as arguments
# df['tier'] = np.select(conditions, values)

# # display updated DataFrame
# df.head() 
def main(data_dir):
	#files = glob.glob(data_dir+'*.csv')
	
	#read in all the files in this dir and combine them into one df
	early=read_data(files)

	for period in ['11','mid','late']: 
		
		#make a temporal chunk of data 
		#chunk=split_yearly_data(combined,period)
		#calculate the snow droughts for that chunk 
		droughts=calculate_snow_drought(chunk)

if __name__ == '__main__':
	data_dir = sys.argv[1]
	# with open(str(params)) as f:
	# 	variables = json.load(f)		
	# 	#construct variables from param file
	# 	sp_data = variables['sp_data']

	main(data_dir)

	#droughts=calculate_snow_drought(combined)
	
	# print(combined)
	# print(combined.columns)
