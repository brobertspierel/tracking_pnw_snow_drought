import matplotlib.pyplot as plt 
import os
import sys
import numpy as np 
from climata.snotel import StationDailyDataIO 
from dateutil.parser import parse
from pathlib import Path
import seaborn as sns
from scipy.stats import linregress 
import pickle 
import geopandas as gpd
from calendar import monthrange
import contextily as ctx 
import collections
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
from dateutil.parser import parse
import random 
from sklearn import preprocessing
import matplotlib
from colour import Color
from pylab import cm
import multiprocessing as mp 
import pyParz 
import time
from statistics import mean
from osgeo import ogr
from subprocess import call
import subprocess
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import scipy as sp
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.lines as mlines

#register_matplotlib_converters()


################################################################################################
################################################################################################
################################################################################################
def make_site_list(input_df,huc_col,huc_level): 
        """Get a list of all the snotel sites from input csv."""
        sites=input_df#pd.read_csv(input_csv) #read in the list of snotel sites by state
        try: 
            #sites[huc_col]=#sites['huc_'].str.extract(r"\((.*?)\)")
            huc_dict=dict(zip(sites.id, sites[huc_col]))#.str.slice(0,huc_level,1))) #currently set to get huc04, change the end point to get a different huc level 
        except KeyError: 
            print('There was an issue with the format of the huc col, please double check inputs')
     
        site_ls = sites['id'].astype('str').tolist()
        return (sites,site_ls,huc_dict)

class CollectData(): 
    #these functions are used to get snotel data, clean and organize. 

    def __init__(self,station_csv,parameter,start_date,end_date,state,site_list,write_out,output_filepath): 
        self.station_csv = station_csv
        #self.station = station
        self.parameter = parameter
        self.start_date = start_date
        self.end_date = end_date
        self.state = state
        self.site_list = site_list
        self.output_filepath = output_filepath
        self.write_out = write_out #a boolean dictating whether to pickle the collected data 

    def get_snotel_data(self,station):#station, parameter,start_date,end_date): #create a function that pulls down snotel data
        """Collect snotel data from NRCS API. The guts of the following code block comes from: 
        https://pypi.org/project/climata/. It is a Python library called climata that was developed to pull down time series 
        data from various government-maintained networks."""

        data = StationDailyDataIO(
            station=station, #station id
            start_date=self.start_date, 
            end_date=self.end_date,
            parameter=self.parameter #assign parameter
        )
        #Display site information and time series data

        for series in data: 
            snow_var=pd.DataFrame([row.value for row in series.data]) #create a dataframe of the variable of interest
            date=pd.DataFrame([row.date for row in series.data]) #get a dataframe of dates
        df=pd.concat([date,snow_var],axis=1) #concat the dataframes
        df.columns=['date',f'{self.parameter}'] #rename columns
        df['year'] = pd.DatetimeIndex(df['date']).year
        df['month'] = pd.DatetimeIndex(df['date']).month
        df['day'] = pd.DatetimeIndex(df['date']).day
        #df['month'] = df['month'].astype(np.int64)
        df['id']=station.partition(':')[0] #create an id column from input station 
        return df  

    def snotel_compiler(self):#,parameter,start_date,end_date,write_out):
        """Create a list of dataframes. Each df contains the info for one station in a given state. It is intended to put all 
        of the data from one state into one object."""
        df_ls = []
        missing = []
        count = 0
        filename = self.output_filepath+f'{self.state}_{self.parameter}_{self.start_date}_{self.end_date}_snotel_data_list'
        if not os.path.exists(filename): 
            for i in self.site_list: 
                try: 
                    df=self.get_snotel_data(f'{i}:{self.state}:SNTL')#this might need to be adjusted depending on how these are called ,f'{self.parameter}',f'{self.start_date}',f'{self.end_date}')
                    df_ls.append(df)
                except UnboundLocalError as error: 
                    print(f'{i} station data is missing')
                    missing.append(i) 
                    continue
                count +=1
                print(count)
            if self.write_out.lower() == 'true': 
                print('the len of df_ls is: ' + str(len(df_ls)))
                if os.path.exists(filename): 
                    print('That file already exists, continuing...')
                else: 
                    pickle_data=pickle.dump(df_ls, open(filename,'ab'))
                #print('left if statement')
            else: 
                print('did not write data to pickle')
                return df_ls
        return filename

def pickle_opener(filename): 
    """If the 'True' argument is specified for snotel_compiler you need this function to read that pickled
    object back in."""
    df_ls = pickle.load(open(filename,'rb'))#pickle.load( open( filepath/f'{state}_snotel_data_list_{version}', 'rb' ) )
    return df_ls


def water_years(input_df,start_date,end_date): 
    """Cut dataframes into water years. The output of this function is a list of dataframes with each dataframe
    representing a year of data for a single station. """
    #print('the input df is: ',input_df)
    df_ls=[]
    df_dict={}

    for year in range(int(start_date[0:4])+1,int(end_date[0:4])+1): #loop through years, add one because its exclusive 
        #df_dict={}
        #convert starting and ending dates to datetime objects for slicing the data up by water year
        startdate = pd.to_datetime(f'{year-1}-10-01').date()
        enddate = pd.to_datetime(f'{year}-09-30').date()
        inter = input_df.set_index(['date']) #this is kind of a dumb addition, I am sure there is a better way to do this
        wy=inter.loc[startdate:enddate] #slice the water year
        wy.reset_index(inplace=True)#make the index the index again
        df_dict.update({str(year):wy})
        df_ls.append(df_dict) #append the dicts to a list
    return df_dict


################################################################################################
################################################################################################
################################################################################################
#data prep functions

class StationDataCleaning(): 
    """Get snotel data and clean, organize and prep for plotting.
    This takes as input the list/dict of dataframes that is produced and pickled in the collect data class. 
    Run from snotel_intermittence_master_V5 using snotel_intermittence_master.txt as the param file. 
    """
    def __init__(self,input_ls,parameter,season): 
            self.input_ls=input_ls
            self.parameter=parameter
            #self.new_parameter=new_parameter
            #self.start_date=start_date
            #self.end_date=end_date
            self.season = season 

    def scaling(self,df):
        """Select a subset of the data and/or resample."""

        if self.season.lower() == 'core_winter': 
            df = df[df['month'].isin(['12','01','02'])]

        elif self.season.lower() == 'extended_winter': 
            df = df[df['month'].isin(['11','12','01','02','03','04'])]

        #select spring months
        elif self.season.lower() == 'spring': 
            df = df[df['month'].isin(['03','04','05'])]

        elif self.season.lower() == 'full_season': 
            df = df[df['month'].isin(['10','11','12','01','02','03','04','05','06'])]            

        elif self.season.lower() == 'resample': 
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            #df_slice = df.loc['']
            if 'TAVG' in list(df.columns): #we don't want to sum the avg temps for a week because we will end up with a crazy value
                df[self.parameter] = df[self.parameter].resample('W').mean()
                df[self.parameter]=df[self.parameter].round(2)
            else: #any other of snow depth, swe or precip we can sum so do that
                df[self.parameter] = df[self.parameter].resample('W').sum()
                df[self.parameter]=df[self.parameter].round(2)
            #test calculating a rolling std
            #df = df.rolling(2).std()
            df = df.dropna()
            #print(df)
            df = df.reset_index()
    
        else: 
            print('That is not a valid parameter for season. Choose one of: resample, core_winter, full winter or spring')
            pass
        #df[self.parameter] = df[self.parameter].rolling(4).std()

        return df

    def convert_f_to_c(self,temp): 
        """Helper function for conversion."""
        return (temp-32)*5/9

    def convert_in_to_cm(self,inches): 
        """Helper function for conversion."""
        return inches*2.54
    def prepare_data(self,anomaly,start_date,end_date,state):#input_ls,parameter,new_parameter,start_date,end_date,season): 
        """This should change the water year lists into dataframes so that they can be fed in as dataframes with every year, one for every station."""
        station_dict = {}
        anom_dict = {} 
        num_years_dict = {}
        for df in self.input_ls: #the input_ls should be the list of dataframes from results. 
            station_id=df['id'][0]
            #prep input data 
            df1=self.scaling(df)#self.parameter,self.new_parameter,self.season)
            wy_ls=water_years(df1,start_date,end_date) #list of dicts
            concat_ls = []
            for key,value in wy_ls.items():
                 
                if not value.empty: 
                    try: 
                        df2=value.drop(['date','year','month','id','day'],axis=1) #removed month 10/22/2020
                    except KeyError: 
                        raise KeyError('Double check the cols in the input df, currently trying to drop date, year, month, id and day')
               
                    df2=df2.rename(columns={self.parameter:self.parameter+'_'+key}) #changed from new_param
                    concat_ls.append(df2)
                    
                else: 
                    continue 
            wy_df = pd.concat(concat_ls,axis=1)

            if anomaly.lower() == 'true': #calculate an anomaly from a long term mean (normal) NOTE this might need to be changed as we're not really using that stat column currently 
                anom_df = wy_df
                if self.parameter == 'PRCP': #PREC is a cumulative variable, change to avg will require redownloading. PRCP is a step variable
                    anom_df['stat_PRCP'] = (anom_df.max()).mean() #get the mean of the max for each year #.max(axis=1) #get the peak value
                elif self.parameter == 'PRCPSA': 
                    anom_df['stat_PRCPSA'] = (anom_df.max()).mean()
                elif self.parameter == 'PREC': #cumulative precip
                    anom_df = anom_df*2.54
                    anom_df['stat_PREC'] = (anom_df.max()).mean()
                elif self.parameter == 'TAVG': #temp needs to be converted to deg C and then the thawing degrees are calculated
                    #anom_df = anom_df.apply(np.vectorize(self.convert_f_to_c))
                    anom_df = (anom_df-32)*(5/9) #convert f to c 
                    anom_df['stat_TAVG'] = (anom_df[anom_df > 0 ].count()).mean() #new thawing degrees calculation. This is still somewhat suspect 
                    #TD definition: Thawing degrees (TDs) were calculated as the sum of mean daily temperatures for all winter
                    #days with a mean daily temperature above 0 °C.
            
                elif self.parameter == 'SNWD': #snow depth and swe these should probably be converted to cm but currently in inches
                    #anom_df = anom_df.apply(np.vectorize())
                    anom_df=anom_df*2.54 #convert in to cm
                    anom_df['stat_SNWD'] = (anom_df.max()).mean()
                  
                elif self.parameter == 'WTEQ': #this one will get the peak swe
                    anom_df=anom_df*2.54 #convert in to cm
                    anom_df['stat_WTEQ'] = (anom_df.max()).mean()
                  
                station_dict.update({station_id:anom_df}) #removed the transpose
            
            else: 
                station_dict.update({station_id:wy_df})
     
        return station_dict #pct_change_wy #removed season mean

class PrepPlottingData(): #DEPRECEATED? 1/15/2021
    """Create visulization of simple or multiple linear regression with sentinel 1 wet snow outputs and snotel data."""
    def __init__(self,input_df,input_csv,station_id,gee_data): 
        self.input_df=input_df #snotel data 
        self.input_csv=input_csv #sentinel 1 data from GEE
        self.station_id = station_id
        self.gee_data = gee_data 
        #self.input_dict = input_dict #this is the result of each snotel param
    def csv_to_df(self): 
        #automatically try to parse dates- default will be from the system:time_index col which is the GEE default 
        #df = pd.read_csv(self.input_csv,parse_dates={'date_time':[1]})
        try: 
            df = pd.read_csv(self.input_csv,parse_dates={'date_time':[1]})
        except KeyError: 
            try: 
                df = pd.read_csv(self.input_csv,parse_dates=True)
                print('The cols in your df are: ', df.columns, 'changing the date column...')
                #system:time_start is a GEE default, rename it
                df.rename(columns={'date':'date_time'},inplace=True)
                df = df.sort_values('date_time')

            except: 
                print('Something is wrong with the format of your df it looks like: ')
                df = pd.read_csv(self.input_csv)
                print(df.head()) 

        return df#,(df['date_time'].iloc[-2]) the second item was commented out on 11/5/2020.year #get the second to last element in case there is a nan. This will still be the water year 

    def clean_gee_data(self,id_column): 
        
        df = self.gee_data #df of gee data that was read in externally with csv_to_df
        
        if 'huc' in id_column: #cut dataframes so they're just by huc level 
            df_list = []
            #print(type(df[id_column].iloc[0]))
            for huc_id in set(list(df[str(id_column)])): 
                #print('the huc id is: ', huc_id)
                try: 
                    df1 = df.loc[df['site_num']==self.station_id]
                    #print('df here is: ',df)
                except Exception as e: 
                    #print('Double check your column headers. This should be the site id and the default is site_num')
                    print('Processing as huc levels...')
                    df1 = df.loc[df[str(id_column)]==int(huc_id)]
                    #print(df1)
                    #raise KeyError('Double check your column headers. This should be the site id and the default is site_num.')
                
                #the sentinel images in GEE are split into two tiles on a day there is an overpass, combine them. 
                try: 
                    df1 = df1.groupby([df1['date_time'].dt.date,id_column])['filter'].sum().reset_index() #filter is the default column name 
                except Exception as e:
                    print('Something went wrong with the groupby') 
                    print('Please double check the column header you are using. The defualt from GEE is filter')
                    #raise KeyError('Please double check the column header you are using. The defualt from GEE is filter.')
                #df = df.groupby([df['Date_Time'].dt.date])['B'].mean()

                #df1 = (df.set_index('date').resample('D')['filter'].sum()).reset_index()
                #print('second df is: ', df1)
                try: 
                    #get the week of year
                    df1['date_time'] = pd.to_datetime(df1['date_time'])
                    df1['week_of_year'] = df1['date_time'].dt.week 
                    
                    #add a month col
                    df1['month'] = df1['date_time'].dt.month

                    #get the week of year where October 1 falls for the year of interest 
                    base_week = datetime.datetime(df1.date_time[0].to_pydatetime().year,10,1).isocalendar()[1]
                    df1.loc[df1.month >= 10,'week_of_year'] = df1.week_of_year-base_week
                    #adjust values that are after the first of the calendar year 
                    df1.loc[df1.month < 10, 'week_of_year'] = df1.week_of_year + 12
                except IndexError: 
                    #raise
                    try: 
                        print(f'That df seems to be empty. The start date is {df["date"][0]} and the id is {self.station_id}')
                    except Exception as e: 
                        pass
                #print(df1)
                df_list.append(df1)
            df_out = pd.concat(df_list)
            #print(df_out)
            #df_out.to_csv("/vol/v1/general_files/user_files/ben/sentinel_1_snow/tests/example_output.csv")
        else: 
            print('Processing huc layout right now. If you want something else change the format of your csv')
        #test the std for the sentinel data
        #df1['filter'] = df1['filter'].rolling(2).std()
        return df_out

    def make_plot_dfs(self,year): 
        #get sentinel data 
        sentinel_df = self.gee_data#formerly self.clean_gee_data()
        #get list of weeks that had a sentinel obs 
        sentinel_weeks = sentinel_df.week_of_year.tolist()
        #get snotel station that aligns with the sentinel data 
        snotel_df = self.input_df#self.df_dict[f'{self.station_id}']
        #create a week of year col
        snotel_df['week_of_year'] = range(1,(len(snotel_df.index))+1)
         #select only the weeks of the snotel data that coincide with sentinel visits 
        snotel_df=snotel_df[snotel_df['week_of_year'].isin(sentinel_weeks)].reset_index()
        #make a dataset that combines the snotel and sentinel data for ease of plotting
        df = pd.concat(list([snotel_df.loc[:, snotel_df.columns.str.contains(str(year))],
            snotel_df.loc[:,snotel_df.columns.str.contains('stat')],
            sentinel_df['filter']]),axis=1) 
        df = df.fillna(value={year:0})
        
        return df
