import os 
import sys 
import numpy as np
import h5py
import matplotlib.pyplot as plt 
from osgeo import gdal
from osgeo import osr


# the RefMatrix numbers specify the grid size in the x and y directions, and coordinates of the the upper left corner of the raster.
# col 1 row 1-2 are the dimensions of the x pixel (edited) 
# col 2 row 1-2 are the dimensions of the y pixel
# row 3 col1 = X coordinate of upper left part of the full spatial image
# row 3 col2= Y coordinate of the upper left

def read_mat_file(file): 
	"""Reads a .mat file and extracts the gridded snow data and geospatial info."""
	f = h5py.File(mod_file,'r')
	data = f.get('snow_fraction')
	coords = f.get('RefMatrix')
	return np.array(data),np.array(coords)

def create_geotransform(arr): 
	"""Takes a numpy array from mat file and creates a geostransform tuple for gdal."""
	x_res = arr[0,1]
	y_res = arr[1,0]
	ul_x = arr[0,2]
	ul_y = arr[1,2]

	return ul_x, x_res, 0, ul_y, 0, y_res

def create_raster(arr,output_file,width,height,geotrans): 
	#needs geotransform info as: (upper_left_x, x_resolution, x_skew, upper_left_y, y_skew, y_resolution)
	dst_ds = gdal.GetDriverByName("GTiff").Create(output_file, width, height, 1, gdal.GDT_Float32) #here is the number of bands
	
	dst_ds.SetGeoTransform(geotrans)
	srs = osr.SpatialReference()
	srs.ImportFromEPSG(4326)
	dst_ds.SetProjection(srs.ExportToWkt())
	dst_ds.GetRasterBand(1).WriteArray(arr)   # write r-band to the raster
	dst_ds.FlushCache()                     # write to disk
	dst_ds = None

mod_file = "/vol/v1/general_files/user_files/ben/rasters/MODSCAG/westernUS_Terra_20170401_mindays10_minthresh05_ndsimin0.00_zthresh08000800.mat"
out_fn = os.path.join("/vol/v1/general_files/user_files/ben/rasters/MODSCAG/",'test.tif')
snow_frac,geo_info = read_mat_file(mod_file)
transform = create_geotransform(geo_info)
height, width = snow_frac.shape
create_raster(snow_frac, out_fn, width, height,transform)
print(snow_frac.shape)
print(geo_info.shape)
print(geo_info)
