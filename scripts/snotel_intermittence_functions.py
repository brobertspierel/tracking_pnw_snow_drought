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
def make_site_list(input_df): 
        """Get a list of all the snotel sites from input csv."""
        sites=input_df#pd.read_csv(input_csv) #read in the list of snotel sites by state
        if not 'site_num' in sites: 
            try: 
                sites['site_num']=sites['site name'].str.extract(r"\((.*?)\)") #strip the site numbers from the column with the name and number
                site_ls= sites['site_num'].tolist()
                #print('try')
                #print(site_ls)
            except KeyError:  #this was coming from the space instead of _. Catch it and move on. 
                sites['site_num']=sites['site_name'].str.extract(r"\((.*?)\)") #strip the site numbers from the column with the name and number
                site_ls= sites['site_num'].tolist()
                print('except')

            try: 
                sites['huc_id']=sites['huc'].str.extract(r"\((.*?)\)")
                huc_dict=dict(zip(sites.site_num, sites.huc_id.str.slice(0,4,1))) #currently set to get huc04, change the end point to get a different huc level 
            except KeyError: 
                print('There was an issue with the format of the huc col, please double check inputs')
        else: 
            site_ls = sites['site_num'].astype('str').tolist()
            huc_dict=dict(zip(sites.site_num.astype('str'), sites.huc_id.astype('str').str.slice(0,4,1))) #currently set to get huc04, change the end point to get a different huc level 
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
            filename = self.output_filepath+f'{self.state}_{self.parameter}_{self.start_date}_{self.end_date}_snotel_data_list'
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
        #df[self.parameter] = df[self.parameter].rolling(7).std()
        #select core winter months
        if self.season.lower() == 'core_winter': 
            df = df[df['month'].isin(['12','01','02'])]
        #select spring months
        elif self.season.lower() == 'spring': 
            df = df[df['month'].isin(['03','04','05'])]

        elif self.season.lower() == 'full_winter': 
            df = df[df['month'].isin(['12','01','02','03','04','05'])]            

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
            print('That is not a valid parameter for season. Choose one of: resample, core_winter or spring')
        #df[self.parameter] = df[self.parameter].rolling(4).std()

        return df

    def convert_f_to_c(self,temp): 
        return (temp-32)*5/9

    def prepare_data(self,anomaly,start_date,end_date,state):#input_ls,parameter,new_parameter,start_date,end_date,season): 
        """This should change the water year lists into dataframes so that they can be fed in as dataframes with every year, one for every station."""
        station_dict = {}
        #print('df is ',input_ls[2])
        anom_dict = {} 
        num_years_dict = {}
        for df in self.input_ls: #the input_ls should be the list of dataframes from results. 
            #print(df['date'][0].year)
            station_id=df['id'][0]
            #figure out how many years we have for all stations
            #num_years = int(end_date[:4])-int(df['date'][0].year)
            #num_years_dict.update({station_id:num_years})
            #prep input data 
            df1=self.scaling(df)#self.parameter,self.new_parameter,self.season)
            wy_ls=water_years(df1,start_date,end_date) #list of dicts
            #pct_change_wy = {k:(v[self.parameter]).pct_change()*100 for k,v in wy_ls.items()}
            concat_ls = []
            for key,value in wy_ls.items():
                 
                if not value.empty: 
                    try: 
                        df2=value.drop(['date','year','month','id','day'],axis=1)
                    except KeyError: 
                        raise KeyError('Double check the cols in the input df, currently trying to drop date, year, month, id and day')
                    #df1 = value[new_parameter]
                    #df2 = df2.replace(np.nan,0) #this might need to be changed

                    df2=df2.rename(columns={self.parameter:self.parameter+'_'+key}) #changed from new_param
                    concat_ls.append(df2)
                    
                else: 
                    continue 
            wy_df = pd.concat(concat_ls,axis=1)

            if anomaly.lower() == 'true': #calculate an anomaly from a long term mean (normal)
                #wy_df['mean'] = wy_df.T.max(axis=1)
                #anom_df = wy_df.transpose() #commented out the transpose to add stat and anom as cols
                anom_df = wy_df
                if self.parameter == 'PRCP': #PREC is a cumulative variable, change to avg will require redownloading. PRCP is a step variable
                    anom_df['stat_PRCP'] = anom_df.mean(axis=1) #get the peak value
                    #the next line and its equivalent will calculate the anomaly from the dataset mean
                    #anom_df = anom_df.subtract(anom_df['stat_PREC'],axis=0) 
                    #anom_df['anomaly'] = (anom_df['stat']-int(anom_df['stat'].mean()))    
                elif self.parameter == 'TAVG': #temp needs to be converted to deg C and then the thawing degrees are calculated
                    anom_df = anom_df.apply(np.vectorize(self.convert_f_to_c))
                    anom_df['stat_TAVG'] = anom_df[anom_df > 0].sum(axis=1) #calculate thawing degrees as per Dierauer et al. 
                    #anom_df = anom_df.subtract(anom_df['stat_TAVG'],axis=0)
                    #anom_df['stat_TD'] = anom_df['stat'].max(axis=1) #get the peak value
                elif self.parameter == 'SNWD': #snow depth and swe these should probably be converted to cm but currently in inches
                    anom_df['stat_SNWD'] = anom_df.mean(axis=1) #get the peak value
                    #anom_df = anom_df.subtract(anom_df['stat_SNWD'],axis=0) 
                    #anom_df['anomaly'] = anom_df['stat']-int(anom_df['stat'].mean())
                #wy_df.loc['max'] = wy_df.max()#(int(wy_df['max'].mean())-wy_df.max())
                elif self.parameter == 'WTEQ':
                    anom_df['stat_WTEQ'] = anom_df.mean(axis=1)
                    #anom_df = anom_df.subtract(anom_df['stat_WTEQ'],axis=0)
                station_dict.update({station_id:anom_df}) #removed the transpose
            
            else: 
                station_dict.update({station_id:wy_df})
                #station_dict.update({station_id:wy_df}) #removed the transpose
        #pickled = pickle.dump(station_dict, open( f'{filename}', 'ab' ))
        #print(station_dict)
        #print(num_years_dict)
        #years_df = pd.DataFrame.from_dict(num_years_dict,orient='index')
        #years_df.to_csv(f'/vol/v1/general_files/user_files/ben/excel_files/{state}_year_counts.csv')
        return station_dict #pct_change_wy #removed season mean

class PrepPlottingData(): 
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
        
        return df,(df['date_time'].iloc[-2]).year #get the second to last element in case there is a nan. This will still be the water year 

    def clean_gee_data(self): 
        df = self.gee_data #df of gee data that was read in externally with csv_to_df
        # print('the intital df is: ',df)
        # print(df['site_num'])
        # print(type(df['site_num'][0]))
        try: 
            df = df.loc[df['site_num']==self.station_id]
            #print('df here is: ',df)
        except: 
            raise KeyError('Doule check your column headers. This should be the site id and the default is site_num.')
        
        #the sentinel images in GEE are split into two tiles on a day there is an overpass, combine them. 
        try: 
            df1 = df.groupby([df['date_time'].dt.date])['filter'].sum().reset_index() #filter is the default column name 
        except:
            print('something went wrong with the groupby') 
            raise KeyError('Please double check the column header you are using. The defualt from GEE is filter.')
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
            raise
            print(f'That df seems to be empty. The start date is {df["date"][0]} and the id is {self.station_id}')
        #test the std for the sentinel data
        #df1['filter'] = df1['filter'].rolling(2).std()
        return df1

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


    #currently working 
    # def clean_gee_data(self): 
    #     df = self.gee_data #df of gee data that was read in externally with csv_to_df

    #     try: 
    #         df = df.loc[df['site_num']==self.station_id]
    #     except: 
    #         raise KeyError('Doule check your column headers. This should be the site id and the default is site_num.')
        
    #     #the sentinel images in GEE are split into two tiles on a day there is an overpass, combine them. 
    #     try: 
    #         df1 = df.groupby([df['date_time'].dt.date])['filter'].sum().reset_index() #filter is the default column name 
    #     except:
    #         print('something went wrong with the groupby') 
    #         raise KeyError('Please double check the column header you are using. The defualt from GEE is filter.')
    #     #df = df.groupby([df['Date_Time'].dt.date])['B'].mean()

    #     #df1 = (df.set_index('date').resample('D')['filter'].sum()).reset_index()
    #     #print('second df is: ', df1)
    #     try: 
    #         #get the week of year
    #         df1['date_time'] = pd.to_datetime(df1['date_time'])
    #         df1['week_of_year'] = df1['date_time'].dt.week 
            
    #         #add a month col
    #         df1['month'] = df1['date_time'].dt.month

    #         #get the week of year where October 1 falls for the year of interest 
    #         base_week = datetime.datetime(df1.date_time[0].to_pydatetime().year,10,1).isocalendar()[1]
    #         df1.loc[df1.month >= 10,'week_of_year'] = df1.week_of_year-base_week
    #         #adjust values that are after the first of the calendar year 
    #         df1.loc[df1.month < 10, 'week_of_year'] = df1.week_of_year + 12
    #     except IndexError: 
    #         print(f'That df seems to be empty. The start date is {df["date"][0]} and the id is {self.station_id}')
    #     return df1

    # def make_plot_dfs(self,year): 
    #     #get sentinel data 
    #     sentinel_df = self.clean_gee_data()
    #     #get list of weeks that had a sentinel obs 
    #     sentinel_weeks = sentinel_df.week_of_year.tolist()
    #     #get snotel station that aligns with the sentinel data 
    #     snotel_df = self.input_df#self.df_dict[f'{self.station_id}']
    #     #create a week of year col
    #     snotel_df['week_of_year'] = range(1,(len(snotel_df.index))+1)
    #      #select only the weeks of the snotel data that coincide with sentinel visits 
    #     snotel_df=snotel_df[snotel_df['week_of_year'].isin(sentinel_weeks)].reset_index()
    #     #make a dataset that combines the snotel and sentinel data for ease of plotting
    #     df = pd.concat(list([snotel_df.loc[:, snotel_df.columns.str.contains(str(year))],
    #         snotel_df.loc[:,snotel_df.columns.str.contains('stat')],
    #         sentinel_df['filter']]),axis=1) 
    #     df = df.fillna(value={year:0})
        
    #     return df
class LinearRegression(): 
    """Functions to carry out simple and multiple linear regression analysis from a df."""

    def __init__(self,input_df,param_list): 
        self.input_df = input_df
        self.param_list = param_list

    def simple_lin_reg_plot(self): 
        fig,ax = plt.subplots(1,1,figsize=(15,15))
        linreg = sp.stats.linregress(df[year],df['filter'])
        #The regression line can then be added to your plot: -

        
        #df['anomaly'].plot.line(ax=ax,legend=False,color='darkblue',lw=2)
        ax.scatter(df[year],df['filter'])
        ax.plot(np.unique(df[year]), np.poly1d(np.polyfit(df[year], df['filter'], 1))(np.unique(df[year])))
        ax.set_xlabel('Snow Water Equivalent (in SWE)')
        ax.set_ylabel('30m Sentinel 1 pixels classified as wet snow')
        ax.set_title('SNOTEL SWE vs Sentinel 1 wet snow area '+year+' water year')
        X2 = sm.add_constant(df[year])
        est = sm.OLS(df['filter'], X2)
        print(est.fit().f_pvalue)
        #Similarly the r-squared value: -
        plt.text(10, 1000, 'r2 = '+str(linreg.rvalue))
        
        plt.show()
        return df

    def multiple_lin_reg(self): 
        """Do multiple linear regression analysis of snotel and sentinel 1. Return results as df."""
        df = self.input_df
        #df = df[np.isfinite(df).all(1)]

        #df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]  # .astype(np.float64)
        #df = self.input_df.fillna(0)
        # df = self.input_df.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        # df = df.replace([np.inf, -np.inf], np.nan)
        # df = df.fillna(0.0)
        #df = df.dropna(0, how="all")
        print('df is: ', df)
        cols = list(df.columns)
        cols.remove('filter') #always remove filter because this will be the dependent variable 
        x_cols =  [x for x in cols if not x.startswith('WTEQ') and not x.startswith('PREC')]#only remove a few changing variables to see what's up with multicolinearity [i for i in cols if i not in list(['filter','WTEQ','PREC'])]#[i for i in cols if subs in i] #filter(lambda i: i not in ['filter','*WTEQ','*PREC'], cols) 
        print(x_cols)

        x = self.input_df[x_cols]  
        x = x.fillna(0)
        X_constant = sm.add_constant(x)

        print(x)
        y = self.input_df['filter']
        y = y.fillna(0)
        model = sm.OLS(y, x).fit()
        residuals=model.resid.mean()
        print(residuals)
        #predictions = model.predict(x)
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        vif["features"] = x.columns
        print(vif)
        print(model.summary())
        #visualize 
        sns.pairplot(df)
        plt.show()
        # fitted_vals = model.predict()
        # resids = model.resid

        # fig, ax = plt.subplots(1,2)
        
        # sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
        # ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
        # ax[0].set(xlabel='Predicted', ylabel='Observed')

        # sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
        # ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
        # ax[1].set(xlabel='Predicted', ylabel='Residuals')
        # plt.show()
    def linearity_test(model, y):
        '''
        Function for visually inspecting the assumption of linearity in a linear regression model.
        It plots observed vs. predicted values and residuals vs. predicted values.
        
        Args:
        * model - fitted OLS model from statsmodels
        * y - observed values
        '''
        fitted_vals = model.predict()
        resids = model.resid

        fig, ax = plt.subplots(1,2)
        
        sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
        ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
        ax[0].set(xlabel='Predicted', ylabel='Observed')

        sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
        ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
        ax[1].set(xlabel='Predicted', ylabel='Residuals')
        
    #linearity_test(lin_reg, y)    
    def vis_relationship(self,year,station_id): 
        df = self.input_df
        #add a week of year col based on index
        df['week_of_year'] = range(1,(len(df.index))+1)
        plot_size = int(len(self.param_list))
        fig,ax = plt.subplots(2,2)
        ax = ax.flatten()
         
        for i in range(len(self.param_list)): #['WTEQ','PREC','TAVG']
            
            ax[i].plot(df['week_of_year'],df[f'{self.param_list[i]}_{year}'],color='darkblue',lw=2)
            #this plots the mean values but it is uncessary if plotting anomaly from the mean 
            ax[i].plot(df['week_of_year'],df[f'stat_{self.param_list[i]}'],color='forestgreen',lw=2,ls='--')
            ax[i].set_title(f'{self.param_list[i]} vs Sentinel 1 wet snow')
            ax[i].set(xlabel='Week of year', ylabel='SWE (in)')
            ax0=ax[i].twinx()
            ax0.plot(df['week_of_year'],df['filter'],color='darkred',lw=2)
            ax0.set(ylabel='Sentinel 1 wet snow (30m pixels)')
            #add a legend to the first plot
            if i == 0: #add the legend only to the first plot  
                blue_line = mlines.Line2D([], [], color='darkblue', label='SNOTEL')
                reds_line = mlines.Line2D([], [], color='darkred', label='Sentinel 1')
                green_line = mlines.Line2D([], [], color='forestgreen', label='Mean')
                plt.legend(handles=[blue_line, reds_line, green_line])
            elif i == 3: #add a pearson correlation coefficient to the final plot (snow depth)
                r, p = stats.pearsonr(df.dropna()[f'{self.param_list[i]}_{year}'], df.dropna()['filter'])
                ax[i].annotate(f"Pearson r = {np.round(r,2)} \nand p = {np.round(p,2)}",xy=(0.8,0.35),xycoords='figure fraction')
            else: 
                pass 
        fig.suptitle(f'Sentinel 1 vs SNOTEL HUC04 {station_id} for year {year}', fontsize=12)        
        plt.tight_layout()
        plt.show()
        plt.close('all')



        # #add swe
        # sns.lineplot(x=df['week_of_year'],y=df[f'WTEQ_{year}'],ax=ax[0],color='darkblue',lw=2)
        # #this plots the mean values but it is uncessary if plotting anomaly from the mean 
        # sns.lineplot(x=df['week_of_year'],y=df['stat_WTEQ'],ax=ax[0],color='forestgreen',lw=2,style=True, dashes=[(2,2)])
        # ax[0].set_title('SWE vs Sentinel 1 wet snow')
        # ax[0].set(xlabel='Week of year', ylabel='SWE (in)')
        # ax0=ax[0].twinx()
        # sns.lineplot(x=df['week_of_year'],y=df['filter'],ax=ax0,color='darkred',lw=2)
        # ax0.set(ylabel='Sentinel 1 wet snow (30m pixels)')
        # #r, p = stats.pearsonr(df.dropna()[f'WTEQ_{year}'], df.dropna()['filter'])
        # #ax[0].annotate(f"Overall Pearson r = {np.round(r,2)} and p = {np.round(p,2)}",xy=(0.8,0.8),xycoords='figure points')
        # #add a legend to the first plot
        # blue_line = mlines.Line2D([], [], color='darkblue', label='SNOTEL')
        # reds_line = mlines.Line2D([], [], color='darkred', label='Sentinel 1')
        # plt.legend(handles=[blue_line, reds_line])
        
        # #add prec
        # sns.lineplot(x=df['week_of_year'],y=df[f'PREC_{year}'],ax=ax[1],color='darkblue',lw=2)
        # ax[1].set_title('PREC vs Sentinel 1 wet snow')
        # ax[1].set(xlabel='Week of year', ylabel='PREC (in)')
        # ax1=ax[1].twinx()
        # sns.lineplot(x=df['week_of_year'],y=df['filter'],ax=ax1,color='darkred',lw=2)
        # ax1.set(ylabel='Sentinel 1 wet snow (30m pixels)')

        # #add temp
        # sns.lineplot(x=df['week_of_year'],y=df[f'TAVG_{year}'],ax=ax[2],color='darkblue',lw=2)
        # ax[2].set_title('TAVG vs Sentinel 1 wet snow')
        # ax[2].set(xlabel='Week of year', ylabel='TAVG (in)')
        # ax2=ax[2].twinx()
        # sns.lineplot(x=df['week_of_year'],y=df['filter'],ax=ax2,color='darkred',lw=2)
        # ax2.set(ylabel='Sentinel 1 wet snow (30m pixels)')

        # #add snow depth
        # sns.lineplot(x=df['week_of_year'],y=df[f'SNWD_{year}'],ax=ax[3],color='darkblue',lw=2)
        # ax[3].set_title('SNWD vs Sentinel 1 wet snow')
        # ax[3].set(xlabel='Week of year', ylabel='SNWD (in)')
        # ax3=ax[3].twinx()
        # sns.lineplot(x=df['week_of_year'],y=df['filter'],ax=ax3,color='darkred',lw=2)
        # ax3.set(ylabel='Sentinel 1 wet snow (30m pixels)')
        # #print(f"Scipy computed Pearson r: {r} and p-value: {p}")
        # #fig.legend(loc="upper right")
        # #import matplotlib.pyplot as plt
        # # defining legend style and data
        


        
##################################################################################
##################################################################################
##################################################################################
