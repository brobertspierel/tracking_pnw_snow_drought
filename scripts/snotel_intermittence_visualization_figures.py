import pandas as pd 
import matplotlib.pyplot as plt 
import os
import sys
import numpy as np 
from climata.snotel import StationDailyDataIO 
from dateutil.parser import parse
from pathlib import Path
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from scipy.stats import linregress 
import pickle 
import geopandas as gpd
from shapely.geometry import Point # Shapely for converting latit
from scipy.stats import linregress 
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from calendar import monthrange
import contextily as ctx 
import collections
from matplotlib.lines import Line2D
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from dateutil.parser import parse
import random 
from sklearn import preprocessing
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy import copy
import matplotlib
from colour import Color
from pylab import cm
import multiprocessing as mp 
import pyParz 
import time
from statistics import mean
register_matplotlib_converters()


def point_mapping(station_list,station_csv,version,state,label,site_label,par_dir,colors,color_dict,diffs,from_year,to_year):
    """Create a simple map with selected stations of interest just to see where they are geographically."""
    #stolen from: httpsstackoverflow.comquestions44488167plotting-lat-long-points-using-basemap
    #first get the list of stations
    print('entered point mapping')
    station_csv.columns = [c.replace(' ', '_') for c in station_csv.columns]
    station_csv['site_num'] = station_csv['site_num'].astype(int)
    df=station_csv[station_csv['site_num'].isin(station_list)]
    df=df.sort_values(by=['site_num'])
    #print(df.head())
    #print(df)
    # creating a geometry column 
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]

    # Coordinate reference system : WGS84 system was set to 4326
    crs = {'init': 'epsg:4326'}

    # Creating a Geographic data frame 
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    gdf = gdf.to_crs(epsg=3857) #3857
    ###########################################
    states = gpd.read_file(par_dir/'states.shp')
    state_map=states[states.STATE_NAME.str.contains(state)] 
    state_map=state_map.to_crs(epsg=3857)
    #oregon = states[~states.STATE_NAME.isin(['Oregon'])]
    #print(oregon)

    #make the figure 
    ##############################################
    fig, ax = plt.subplots(figsize=(24,22))
    #use for the scale bar at some point
    #fig = plt.figure(1)
    #ax=fig.add_subplot(111,projection=ccrs.UTM(zone='10N'))
    #ax.arrow(0.5,0.5, 0.5,0.5,head_width=3, head_length=6)#, fc='k', ec='k')

    state_map.plot(ax=ax,color='None',edgecolor='black',alpha=0.75,linewidth=3)

    #make geo dataframe

    gdf.plot(ax=ax, color=colors,markersize=200,edgecolor='#4c4c4c')
    ctx.add_basemap(ax, zoom = 9)

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    #get the colors for the legend that occur in that set of years
    id_list = set(list(diffs['diffs']))
    try: 
        color_dict_ids = {item:color_dict.get(item) for item in id_list} 
        print('color dict ids',color_dict_ids)
    except KeyError: 
        print('no intersecting ids')
    ############################################

    #add a north arrow
    x, y, arrow_length = 0.98, 0.98, 0.08
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=26,
            xycoords=ax.transAxes)
    #add a title
    plt.title(f'Oregon {from_year}-{to_year} snow persistence shifts',fontsize=35)

    #make a colorbar

    #im = ax.imshow(np.arange(100).reshape((10,10)))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="10%", pad=0.05)

    #plt.colorbar(im, cax=cax)
    img = plt.imshow(np.array([[0,1]]), cmap="coolwarm_r")#,aspect='auto')
    img.set_visible(False)
    
    cbar=plt.colorbar(img, orientation="vertical",ticks=[0,1],fraction=0.0325, pad=0.04) #get the colorbar to be the same size as the subplot
    cbar.ax.set_yticklabels(['Low \npersistence', 'High \npersistence'],fontsize=20)  # vertically oriented colorbar
    #add a legend 
    # snotel = [plt.Line2D([0,0], [0,0], marker='o',color=color, label='Snotel stations',linestyle='',markersize=8) for color in color_dict_out.values()]
    # plt.legend(snotel,manual_color_subset.keys(),numpoints=1,loc='upper left')
    #uncomment the for loop below to have labels
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf[label]):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    plt.savefig(par_dir/'stats'/'outputs'/'final_figs'/f'{state}_{from_year}_{to_year}_map_final_w_labels.png',bbox_inches = 'tight',
    pad_inches = 0) #save the output. Right now the extension is hard coded which isn't ideal
    
    # plt.show()
    # plt.close()