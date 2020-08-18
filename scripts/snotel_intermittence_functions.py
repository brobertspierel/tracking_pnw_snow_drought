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

#register_matplotlib_converters()


################################################################################################
################################################################################################
################################################################################################
def make_site_list(input_csv): 
        """Get a list of all the snotel sites from input csv."""
        sites=pd.read_csv(input_csv) #read in the list of snotel sites by state
        try: 
            sites['site num']=sites['site name'].str.extract(r"\((.*?)\)") #strip the site numbers from the column with the name and number
            site_ls= sites['site num'].tolist()
            #print('try')
            #print(site_ls)
        except KeyError:  #this was coming from the space instead of _. Catch it and move on. 
            sites['site num']=sites['site_name'].str.extract(r"\((.*?)\)") #strip the site numbers from the column with the name and number
            site_ls= sites['site num'].tolist()
            print('except')
        
        return (sites,site_ls)

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
            pickle_data=pickle.dump(df_ls, open(filename,'ab'))
            #print('left if statement')
        else: 
            print('did not write data to pickle')
            return df_ls
        return filename

    def pickle_opener(self,filename): 
        """If the 'True' argument is specified for snotel_compiler you need this function to read that pickled
        object back in."""
        #filename = self.output_filepath+f'{self.state}_{self.parameter}_{self.start_date}_{self.end_date}_snotel_data_list'
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
    def __init__(self,input_ls,parameter,new_parameter,start_date,end_date,season): 
            self.input_ls=input_ls
            self.parameter=parameter
            self.new_parameter=new_parameter
            self.start_date=start_date
            self.end_date=end_date
            self.season = season 
    def scaling(self,df):
        """Select a subset of the data and/or resample."""
        #df[new_parameter] = df[parameter].rolling(7).mean()

        #select core winter months
        if self.season.lower() == 'core_winter': 
            df = df[df['month'].isin(['12','01','02'])]
        #select spring months
        elif self.season.lower() == 'spring': 
            df = df[df['month'].isin(['03','04','05'])]

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
            df = df.dropna()
            #print(df)
            df = df.reset_index()
        else: 
            print('That is not a valid parameter for season. Choose one of: resample, core_winter or spring')

        return df

    def prepare_data(self,current_param):#input_ls,parameter,new_parameter,start_date,end_date,season): 
        """This should change the water year lists into dataframes so that they can be fed in as dataframes with every year, one for every station."""
        station_dict = {}
        #print('df is ',input_ls[2])

        for df in self.input_ls: #the input_ls should be the list of dataframes from results. NOTE: change this when you're ready to run the whole archive 
            station_id=df['id'][0]
            print(df)
            #prep input data 
            df1=self.scaling(df)#self.parameter,self.new_parameter,self.season)
            wy_ls=water_years(df1,self.start_date,self.end_date) #list of dicts
            print('wy example is: ',wy_ls)
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
            #wy_df = wy_df.pct_change() * 100           
            station_dict.update({station_id:wy_df}) #removed the transpose
        #pickled = pickle.dump(station_dict, open( f'{filename}', 'ab' ))
        #print(station_dict)
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
                print(df)
            except: 
                print('Something is wrong with the format of your df it looks like: ')
                df = pd.read_csv(self.input_csv)
                print(df.head())
        
        return df,(df['date_time'].iloc[-2]).year #get the second to last element in case there is a nan. This will still be the water year 

    def clean_gee_data(self): 
        df = self.gee_data #df of gee data that was read in externally with csv_to_df
        try: 
            df = df.loc[df['site_num']==self.station_id]
           
        except: 
            raise KeyError('Doule check your column headers. This should be the site id and the default is site_num.')
        
        #the sentinel images in GEE are split into two tiles on a day there is an overpass, combine them. 
        try: 
            df1 = df.groupby([df['date_time'].dt.date])['filter'].sum().reset_index() #filter is the default column name 
            #print('df1 is: ', df1)
        except:
            print('something went wrong with the groupby') 
            raise KeyError('Please double check the column header you are using. The defualt from GEE is filter.')
        #df = df.groupby([df['Date_Time'].dt.date])['B'].mean()

        #df1 = (df.set_index('date').resample('D')['filter'].sum()).reset_index()
        #print('second df is: ', df1)

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

        return df1

    def make_plot_dfs(self,year): 
        #get sentinel data 
        sentinel_df = self.clean_gee_data()
        #get list of weeks that had a sentinel obs 
        sentinel_weeks = sentinel_df.week_of_year.tolist()
        #get snotel station that aligns with the sentinel data 
        snotel_df = self.input_df#self.df_dict[f'{self.station_id}']
        #create a week of year col
        snotel_df['week_of_year'] = range(1,(len(snotel_df.index))+1)
         #select only the weeks of the snotel data that coincide with sentinel visits 
        snotel_df=snotel_df[snotel_df['week_of_year'].isin(sentinel_weeks)].reset_index()
        #make a dataset that combines the snotel and sentinel data for ease of plotting
        df = pd.concat(list([snotel_df.loc[:, snotel_df.columns.str.contains(str(year))],sentinel_df['filter']]),axis=1) 
        df = df.fillna(value={year:0})
        
        return df
class LinearRegression(): 
    """Functions to carry out simple and multiple linear regression analysis from a df."""

    def __init__(self,input_df): 
        self.input_df = input_df

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
        #plt.show()
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
##################################################################################
##################################################################################
##################################################################################
#likely depreceated 
    #functions used for time series clustering and parallelization
        
def fastDTW_implement(x,y): 
    """Run the python fastDTW function/package."""
    # calculate the distance 
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    #from dtw import dtw 
    #print('x,y are ',x,y)
    distance, path = fastdtw(x,y,dist=euclidean)#TimeSeriesScalerMeanVariance().fit_transform(x), TimeSeriesScalerMeanVariance().fit_transform(y), dist=euclidean)
    
    return distance  

def faster(class_dict,cols): 
    """Prepare training data for fastDTW run."""
    #create the function that will be applied to each column in the dataframe in clustering 
    return pd.Series(list({k:fastDTW_implement(v,np.nan_to_num(cols.to_numpy(copy=True))) for (k,v) in class_dict.items()}.values()))

def clustering(args):
    """Compare station time series against reference dataset and determine time series type. 
    """
    key,x,y = args #x is the station dataframe and y is the dictionary of training dataframes
    return [key,x[key].apply(lambda column: faster(y,column),axis=0).reset_index()]

def cluster_parallel(train_dict,classify_data,filepath,njobs): 
    """Paralellizes the clustering function above."""
    #get data from pickled dictionary of dataframes
    # if from_pickle: 
    #     df_dict = pickle_opener(1,None,filepath,filename_in)
    # else: 
    df_dict = classify_data
    print('working...')
    #run wrapper pyParz for multiprocessing pool function 
    results_classes=pyParz.foreach(list(df_dict.keys()),clustering,args=[df_dict,train_dict],numThreads=njobs)
    #print(type(results_classes))
    #print(type(results_classes[0]))

    return results_classes

################################################################################################
################################################################################################
################################################################################################
#get data ready for visualization
def plot_distances(dry_year,wet_year,avg_year,start_year,end_year,output_filepath,clusters): 
    """Plot results of distance to cluster centroids calculations."""
    #fig,ax = plt.subplots(nrows=8,ncols=4)
    #inputs are lists of lists. The inner list is like ['station_id',df of distances]
    dry_year_dict = {i[0]:i[1] for i in dry_year}
    wet_year_dict = {i[0]:i[1] for i in wet_year}
    #print(dry_year)
    #print(dry_year_dict)
    nrows=1
    ncols=2
    #axs = axs.ravel()
    #yr_list = list(range(start_year,end_year+1)) 
    print('making figures, please wait...')
    #for i in nrows*ncols: 
    for (k,v), (k2,v2) in zip(sorted(dry_year_dict.items()),sorted(wet_year_dict.items())): 
        fig,(ax,ax1) = plt.subplots(nrows,ncols,figsize=(24,10))
          
        print('current stations are: ',k,k2)
        if k == k2: 
            dy_df = v.drop(['index'],axis=1)
            wy_df = v2.drop(['index'],axis=1)
            ax = dy_df.T.plot.line(ax=ax,color=['darkblue','forestgreen','firebrick','black','peru'])
            ax1 = wy_df.T.plot.line(ax=ax1,color=['violet','purple','darkred',])
            ax.set_ylabel('DTW distance to cluster centroid')
            ax.set_title(f'Station {k} dry year {clusters} clusters')
            ax1.set_ylabel('DTW distance to cluster centroid')
            ax1.set_title(f'Station {k2} wet year {clusters} clusters')
            plt.savefig(output_filepath+k+'_distances_to_cluster_centroid_combined_years_'+f'{clusters}'+'_clusters')
            plt.close('all')
        else:
            print('the stations do not match, stopping')
            break
#def plot_anomolies(df_dict): 
    
def get_shp_extent(input_shape): 
    """Get shapefile extent and return a tuple."""
    source = ogr.Open(input_shape, update=True)
    #layer = source.GetLayer()
    inLayer = source.GetLayer()
    #create extent tuple 
    extent = inLayer.GetExtent()
    return extent

def reproject(input_shape,epsg): 
    """Emulate command line ogr reprojection tool."""
    reprojected_filename = input_shape[:-4]+'_reprojected.shp'
    subprocess.call(['ogr2ogr', '-f','ESRI Shapefile', '-t_srs', 'EPSG:{epsg}'.format(epsg=epsg), '-s_srs', 'EPSG:{epsg}'.format(epsg=epsg), reprojected_filename , input_shape])
    return reprojected_filename


def create_fishnet_vectors(shp_path,output_filepath,gridsize): 
    """Takes a boundary shapefile and creates a fishnet version. Requires make_fishnet_grid.py."""
    grid_extent = get_shp_extent(shp_path)
    print(grid_extent)
    #create gee file output name
    output_filename = output_filepath+'fishnet_{gridsize}_grid.shp'.format(gridsize=int(gridsize))
    try: 
        #check if file has already been created
        if os.path.exists(output_filename):
            print ('cnd file already exists' )
        else:
            print ('making file')
            #call make_fishnet_grid and assign extents from get_shp_extent tuple 
            subprocess.call(['python', 'make_fishnet_grid.py', output_filename, str(grid_extent[0]), str(grid_extent[1]), str(grid_extent[2]), str(grid_extent[3]), str(gridsize), str(gridsize)])
    except RuntimeError:
        print ('That file does not exist' )
        pass
    return output_filename


def grid_mapping(shapefile,kmeans_output,cluster_centers,cluster): 
    shape = gpd.read_file(shapefile)
    shape['site num'] = shape['site num'].fillna(0,inplace=True)
    #shape = shape['site num'].astype('int32').dtypes
    gdf = gpd.GeoDataFrame(kmeans_output,geometry=gpd.points_from_xy(kmeans_output.lon,kmeans_output.lat))
    #print(gdf.head())
    #mapping_data = pd.concat([shape,kmeans_output],axis=1)#shape.merge(kmeans_output, on='site num')
    mapping_data = gpd.sjoin(gdf,shape,how='right',op='within')
    #print(mapping_data)
    #print(mapping_data)
    #print(mapping_data['count'].sum())
    centers_df = pd.DataFrame(cluster_centers,columns=['centers'])
    

    fig, ax1 = plt.subplots()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right",size="5%",pad=0.1)
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.125, 0.47, 0.25, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])

    ax1=mapping_data.plot(column='count',ax=ax1,legend=True,cax=cax)#.plot(mapping_data['count'])
    ax1.set_title(f'Cluster {cluster}')
    ax2.plot(centers_df, color='red')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)



    # fig,(ax,ax2) = plt.subplots(1,2,figsize=[5.5,2.8])
    # axins = inset_axes(ax, width=1.3, height=0.9)
    # ax=centers_df.plot()
    # ax2=mapping_data.plot(column='count',ax=ax2,legend=True)

    plt.show()
    plt.clf()
    plt.close('all')
    # ax = mapping_data.dropna().plot(column='Sum of Revenue', cmap =    
    #                             'YlGnBu', figsize=(15,9),   
    #                              scheme='quantiles', k=3, legend =  
    #                               True);
def transform_data(input_data,year_of_interest,season): 
    arrs = []
    
    for k,v in input_data.items(): 
        #print('k is: ', k)
        #print('v is: ', v.columns)
        #select (or not) a year of interest 
        #add decimal place
        if len(k) == 3: 
            k = '.00000'+k 
        elif len(k) == 4: 
            k='.0000'+k
        else: 
            print(f'something is wrong with that key which is {k}')
            break 
        #print('v shape is: ', v.shape)
        try: 
            if year_of_interest.isnumeric(): 
                df_slice = pd.DataFrame(v[year_of_interest])
                if not season == 'resample': 
                    df = v.append(pd.Series([float(k+str(i)) for i in v.columns],index=df_slice.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
                elif season == 'resample': 
                    rows = list(range(53-df_slice.shape[0]))
                    df = df_slice.loc[df_slice.index.tolist() + rows]
                    df = df.append(pd.Series([float(k+str(i)) for i in df.columns],index=df.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
                else:   
                    print('that is not a valid parameter for "season". Valid params are: core_winter, spring or resample')
                    break
                arrs.append(df.to_numpy())
            else: 
                if not season == 'resample': 
                    df = v.append(pd.Series([float(k+str(i)) for i in v.columns],index=v.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
                elif season == 'resample': 
                    rows = list(range(53-v.shape[0]))
                    df = v.loc[v.index.tolist() + rows]
                    df = df.append(pd.Series([float(k+str(i)) for i in df.columns],index=df.columns),ignore_index=True)#pd.Series(v.columns, index=v.columns)
                else:   
                    print('that is not a valid parameter for "season". Valid params are: core_winter, spring or resample')
                    break
                arrs.append(df.to_numpy())
        except KeyError: 
            print('that station does not have a year that early (or late)')
            continue
        # if not season.lower() == 'core_winter' or season.lower() == 'spring': 
        #   print(df.shape)
            
        #   #df = np.pad(df, ((0,rows),(0,cols)),mode='constant',constant_values=np.nan)
        #   arrs.append(df.to_numpy())#np.pad(df.to_numpy(), ((0,rows),(0,0)),mode='constant',constant_values=np.nan))
        # else: 
        #   print(df.shape)
        #   arrs.append(df.to_numpy())
        
    arr_out= np.concatenate(tuple(arrs),axis=1)
    return arr_out

    ###################################
def plot_anomalies(input_dict): 
    #from the main function- not currently working but used to plot anomolies 
    for k,v in water_years[0].items(): 
        #print(v)
        print(v.iloc[:-1,:-1])
        df = v.iloc[:-1,:-1]
        df['average'] = df.mean(axis=1)
        df['anomaly'] = df['average']-water_years[1] #get the full time series mean for the season of interest
        # yr_min=int(v.index.min())
        # yr_max=int(v.index.max())+1
        # year_list = range(yr_min,yr_max,1)
        #get start and end year
        # try:  
        fig,(ax,ax1) = plt.subplots(2,1, figsize=(5,5))
        #plt.figure()
        #   years = [int(i) for i in v.columns if i.isnumeric()]
        #   print(years)
        # except:
        #   print('that one is not a number')
        #   continue
        # fig,ax = plt.subplots(figsize=(5,5))
        # v[v.columns[len(years):]].plot.line(ax=ax,legend=False)
        #plt.plot(df['anomaly'])


        palette = sns.light_palette('Navy', len(df.T.columns.unique()))
    # for i in range(rows*cols):
    #   try: 
    #       count = 0 
    #       df_slice = df[df['climate_region']==region_list[i]].sort_values('year')
    #       for j in year_list: 
        #       df_slice[df_slice['year']==j].sort_values('low_bound').plot.line(x='low_bound',y=variable,ax=axes[i],legend=False,color=list(palette)[count]) #variable denotes first or last day. Can be first_day_mean or last_day_mean
        #       count +=1

        #   axes[i].set_title(string.capwords(str(region_list[i]).replace('_',' ')))
        #   axes[i].xaxis.label.set_visible(False)
            
        # except IndexError: 
        #   continue
        year_list = sorted(df.T.columns.unique())

        norm = mpl.colors.Normalize(vmin=min(year_list),vmax=max(year_list))
        cmap = sns.light_palette('Navy',len(year_list),as_cmap=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
    
        df['anomaly'].plot.line(ax=ax,legend=False,color='darkblue',lw=2)
        v.T.plot.line(ax=ax1,legend=False,lw=0.5)#color=list(palette))
    
        # put colorbar at desire position
        #cbar_ax = fig.add_axes([0.95, 0.1, .01, .8])
        #add colorbar
        #fig.colorbar(sm,ticks=np.linspace(int(min(year_list)),int(max(year_list))+1,int(len(year_list))+1),boundaries=np.arange(int(min(year_list)),int(max(year_list))+1,1),cax=cbar_ax)

        index_vals=list(df.index.values)
        ax.set_title(f'SNOTEL station {k} {season} months')
        ax.set_ylabel(f'Anomaly from {min(index_vals)} to {max(index_vals)} mean (inches SWE)')
        ax.set_xlabel('Water year')
        ax1.set_xlabel('Day of winter (Dec-Feb)')
        ax1.set_ylabel('Daily total SWE (in)')
        #ax.set_xticks(year_list)
        #ax.set_xticklabels(year_list)
        plt.tight_layout()
        plt.show()
        plt.close('all')
   