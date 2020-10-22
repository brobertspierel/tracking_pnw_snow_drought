import os 
import sys
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Python3 program to find the closest  
# perfect square taking minimum steps 
# to reach from a number  
  
# Function to check if a number is  
# perfect square or not 
from math import sqrt
import math
def is_perfect(n): #https://djangocentral.com/python-program-to-check-if-a-number-is-perfect-square/
	root = math.sqrt(int(n))
	if int(root + 0.5) ** 2 == n:
	    return n
	else:
	    print(n, "is not a perfect square")

# def isPerfect(N): 
#     if (sqrt(N) - floor(sqrt(N)) != 0): 
#         return False
#     return True
  
# # Function to find the closest perfect square  
# # taking minimum steps to reach from a number  
# def getClosestPerfectSquare(N): 
#     if (isPerfect(N)):  
#         print(N, "0")  
#         return
  
#     # Variables to store first perfect  
#     # square number above and below N  
#     aboveN = -1
#     belowN = -1
#     n1 = 0
  
#     # Finding first perfect square  
#     # number greater than N  
#     n1 = N + 1
#     while (True): 
#         if (isPerfect(n1)): 
#             aboveN = n1  
#             break
#         else: 
#             n1 += 1
  
#     # Finding first perfect square  
#     # number less than N  
#     n1 = N - 1
#     while (True):  
#         if (isPerfect(n1)):  
#             belowN = n1  
#             break
#         else: 
#             n1 -= 1
              
#     # Variables to store the differences  
#     diff1 = aboveN - N  
#     diff2 = N - belowN  
  
#     if (diff1 > diff2): 
#         print(belowN, diff2)  
#     else: 
#         print(aboveN, diff1) 
  
# # Driver code  
# N = 1500
# getClosestPerfectSquare(N) 
  
# This code is contributed  
# by sahishelangia 

def get_sentinel_data(csv_dir,huc_level,orbit,water_year_start,water_year_end):
	for file in glob.glob(csv_dir+'*.csv'): 
		if (huc_level in file) and (orbit.upper() in file) and (water_year_start in file) and (water_year_end in file): 
			#get one year, orbit- this has a bunch of different stations in it
			df = pd.read_csv(file,parse_dates=True)
			df = df.drop(columns=['system:index','.geo'],axis=1)
			print(df)
			df['date'] = df['date'].str.split('T',expand=False).str.get(0)
			print(df.head())
			#print(type(df.date.iloc[1]))
			return df
		else: 
			print('That csv does not match your input params')

def plot_sentinel_data(input_df,huc_level,orbit,water_year_end):
	fig,ax=plt.subplots(10,10,figsize=(12,12),sharex=True,sharey=True) 
	huc_ids = input_df[f'huc{huc_level[1]}'].unique()
	ax = ax.flatten()
	for x in range(10*10): 
		try: 
			df = input_df[input_df[f'huc{huc_level[1]}']==huc_ids[x]] #get just the huc number from huc_level which is otherwise 04 for example
		except Exception as e: 
			print(f'exception was {e}')
			continue 
		ax[x].plot(df['date'],df['filter'],color='darkorange',lw=0.5)
		ax[x].set_title(f'HUC {huc_level} {water_year_end} water year')
		ax[x].set_ylabel('Wet snow pixel count')
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()
	plt.close('all')







def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		csv_dir=variables['csv_dir']
		huc_level=variables['huc_level']
		orbit=variables['orbit']
		water_year_start=variables['water_year_start']
		water_year_end=variables['water_year_end']
	df=get_sentinel_data(csv_dir,huc_level,orbit,water_year_start,water_year_end)
	plot_sentinel_data(df,huc_level,orbit,water_year_end)
	#is_perfect(16)
if __name__ == '__main__':
    main()
