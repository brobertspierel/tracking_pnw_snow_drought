import ee
import pprint 
import pandas as pd 
import numpy as np 
import sys 
import os 
#deal with authentication
#authenticate = ee.Authenticate()
# print('authenticate')
# print(authenticate)
ee.Initialize()

class CreateViirsData(): 
	def __init__(self,image): 
		self.image=image
	#use bitmasks to remove clouds, ocean and inland water 
	#from https:#code.earthengine.google.com/274c5d85621c8fc4142d87cbd867d187
	def getQABits(self, img_sel,start, end, newName): 
	    
		# Compute the bits we need to extract.
		pattern = 0;
		for i in range(start,end):#(def i = start; i <= end; i++) {
			pattern += i**2

		# Return a single band image of the extracted QA bits, giving the band
		# a new name.
		return img_sel.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)
	 
	# --- Mask out cloudy pixels - cloud state  from state QA band (bits 3 and 4)
	def cloud_mask(self): 
		# Select the QA band.
		QA = self.image.select('QF1')

		# Get the internal_cloud_algorithm_flag bit.
		cloud = self.getQABits(QA,2, 3, 'cloud_state').expression("b(0) == 2 || b(0) == 3") 
		# Return an image masking out cloudy areas.
		return self.image.updateMask(cloud.eq(0))#.internalCloud.eq(0));


	def water_mask(self): 
		QA = self.image.select('QF2'); 
		water = self.getQABits(QA,1,2,'water').expression("b(0) == 1 || b(0) == 2") 

		return self.image.updateMask(water.eq(0)); 

	#add a ndsi band to the VIIRS data
	def ndsi_calc(self):
		ndsi = self.image.normalizedDifference(['I1','I3']);
		pos_ndsi = ndsi.updateMask(ndsi.gte(0)).multiply(100).round(); 
		new_img = self.image.addBands(pos_ndsi); 
		return new_img.select(['nd','QF1'],['NDSI_Snow_Cover','NDSI_Snow_Cover_Class']); #rename to match the modis product


########################################/##make a combined imageCollection with the VIIRS and MODIS snow products 

def viirs_modis_snow(start_date,end_date,forest_thresh): 
	#get VIIRS data
	viirs = ee.ImageCollection("NOAA/VIIRS/001/VNP09GA"); 
	viirs_sel = viirs.filterDate(start_date,end_date); 
	# Map the cloud and water masking functions over the collection.
	viirs_cloud_masked = viirs_sel.map(lambda img: CreateViirsData(img).cloud_mask()) 
	viirs_cloud_masked = viirs_cloud_masked.map(lambda img: CreateViirsData(img).water_mask())
	#add the ndsi band
	viirs_snow = viirs_cloud_masked.map(lambda img: CreateViirsData(img).ndsi_calc())  
	#add an id band 
	viirs_snow = viirs_snow.map(lambda img: img.set('source','viirs'))

	#get the MODIS terra data
	terra_snow=ee.ImageCollection("MODIS/006/MOD10A1").filterDate(start_date,end_date).map(lambda img: img.set('source','modis'))
	                                                    
	#combine the two collections
	combined = terra_snow.select('NDSI_Snow_Cover').merge(viirs_snow.select('NDSI_Snow_Cover'));
	#cast to float and inherit source property 
	combined = combined.map(lambda img: img.float().set('source',img.get('source')))

	# combined = combined.map(function(img){

	#define a water mask 
	modis_water_mask=ee.Image("MODIS/006/MOD44W/2015_01_01").select('water_mask')
	#define a forest mask 
	forest_mask=ee.Image("USGS/NLCD_RELEASES/2016_REL/2016").select('percent_tree_cover').lte(forest_thresh)
	#apply a water mask
	combined = combined.map(lambda img: img.updateMask(modis_water_mask.eq(0))) 
	#apply a forest mask
	combined = combined.map(lambda img: img.updateMask(forest_mask.eq(0)))
	return combined #img.updateMask((ee.Image("USGS/NLCD_RELEASES/2016_REL/2016").select('percent_tree_cover').lte(forest_thresh))); 
 

def make_daily_composites(start_date,end_date,forest_thresh):
  #get the snow imageCollection
  ic = viirs_modis_snow(start_date,end_date,forest_thresh); 
  #get all of the dates in the collection as a list
  dates = ic.aggregate_array('system:index'); 
  dates = dates.map(lambda x: ee.Date.parse('yyyy_MM_dd',(ee.String(x).slice(2)))) #GEE adds an id (e.g. 1_date), remove it 
   
  dates = dates.distinct() #get unique elements 

  #there are two images for each day in the collection (one from MODIS and one from VIIRS). Take the median image and reset the date. 
  ndsi_combined = dates.map(lambda date: ic.filterDate(date).median().set('system:time_start',ee.Date(date).millis())) 
  
  return ee.ImageCollection.fromImages(ndsi_combined)
 
	#define a function to create the rolling composites 

def rolling_comps(img,ic):
	date = ee.Date(img.get('system:time_start')) 
	start = date.advance(-3,'day') #hardecoded 
	end = date.advance(3,'day') #hardcoded
	fil = ic.filterDate(start,end).median() #hardcoded stat method 
	return fil.set('system:time_start',date)

def binarize_snow(img): 
	return (img.select('NDSI_Snow_Cover')
	.gt(40).set('system:time_start',
	img.get('system:time_start')))  #band and threshold are hardcoded


class ExportStats(): 

	def __init__(self,ic,features,output_folder,scale=1000,dem=None,**kwargs):#,end_date,start_date='1980-10-01'): #make sure to define a different start date if you want something else 
		self.ic=ic
		self.scale=scale
		self.features = features#kwargs.get('features')
		self.dem = dem 
		self.output_folder=output_folder
		for key, value in kwargs.items():
			setattr(self, key, value)

	def calc_export_stats(self,feat,img): 
		"""Use a reducer to calc spatial stats."""
		# var get_ic_counts = comp_ic.map(function(img){ 

		if dem: 
			elev_mean = self.dem.reduceRegion( 
			reducer=ee.Reducer.mean(),
			geometry=feat.geometry(),
			scale=30,
			tileScale=4,
			maxPixels=1e13
			)
			elev_min = self.dem.reduceRegion( 
			reducer=ee.Reducer.min(),
			geometry=feat.geometry(),
			scale=30,
			tileScale=4,
			maxPixels=1e13
			)
			elev_max = self.dem.reduceRegion( 
			reducer=ee.Reducer.max(),
			geometry=feat.geometry(),
			scale=30,
			tileScale=4,
			maxPixels=1e13
			)
		pixel_ct_dict = img.reduceRegion(
			reducer=ee.Reducer.sum(), #maybe change to median? This get's the basin mean for a given day (image)
			geometry=feat.geometry(),
			scale=self.scale,
			tileScale=4,
			maxPixels=1e13
			)
		dict_out = (pixel_ct_dict.set('huc8',feat.get('huc8'))
			.set('date',ee.Date(img.get('system:time_start')))
			.set('elev_mean',elev_mean.get('elevation'))
			.set('elev_min',elev_min.get('elevation'))
			.set('elev_max',elev_max.get('elevation'))) 

		dict_feat = ee.Feature(None, dict_out)
		return dict_feat

	def generate_stats(self): 
				
		get_ic_stats=ee.FeatureCollection(self.features).map(lambda feat: self.ic.map(lambda img: self.calc_export_stats(feat,img)))

		return get_ic_stats
	

	def run_exports(self): 
		"""Export some data."""

		task=ee.batch.Export.table.toDrive(
			collection=ee.FeatureCollection(self.generate_stats()).flatten(),
			description= 'py_modis_viirs_std_stats_by_basin_'+self.timeframe+'_huc8',
			folder=self.output_folder,
			fileNamePrefix='py_modis_viirs_std_stats_by_basin_'+self.timeframe+'_huc8',
			fileFormat= 'csv'
			)
		#start the task in GEE 
		print(task)
		task.start()


def main(hucs,dem,forest_thresh=0): 
	
	dates = {12:31,2:28,4:30}
	for year in range(2000,2021): #years: this is exclusive 
		for m in [[11,12],[1,2],[3,4]]: 	
			#create input image collection 
			#first combine the modis and viirs datasets 
			ic = make_daily_composites(f'{year}-{m[0]}-1',f'{year}-{m[1]}-{dates.get(m[1])}',forest_thresh);
			#next make seven day rolling composites to account for some of the cloud pollution 
			ic = ic.map(lambda img: rolling_comps(img,ic))
			#next binarize the snow data at 0.4 in keeping with literature 
			ic = ic.map(lambda img: binarize_snow(img))
			#generate the exports 
			exports = ExportStats(ic,output_folder='modis_viirs',scale=500,features=hucs,dem=dem,timeframe=f'start_date_{year}_{m[0]}_end_date_{year}_{m[1]}').run_exports()
	print('done generating exports')

if __name__ == '__main__':
	forest_thresh = int(sys.argv[1])

	pnw_snotel = ee.FeatureCollection("users/ak_glaciers/NWCC_high_resolution_coordinates_2019_hucs")
	hucs = ee.FeatureCollection("USGS/WBD/2017/HUC08").filterBounds(pnw_snotel)
	dem = ee.Image("USGS/SRTMGL1_003")
	forest_thresh=75

	main(hucs,dem,forest_thresh)


# //export some stats 
# var export_stats = hucs.map(function(feat){
#   var ic = make_daily_composites(start_date,end_date,forest_thresh); 
  
#   //define a function to create the rolling composites 
#   var rolling_comps = ic.map(function(img){
#   var date = ee.Date(img.get('system:time_start')); 
#   var start = date.advance(-3,'day'); //hardecoded 
#   var end = date.advance(3,'day'); //hardcoded
#   var fil = ic.filterDate(start,end).mean(); //hardcoded stat method 
#   return fil.set('system:time_start',date); 
#   }); 
  
#   var dem_clip = dem.clip(feat.geometry()); 
#   var elev_mean = dem_clip.reduceRegion({ 
#               reducer: ee.Reducer.mean(),
#               geometry: feat.geometry(),
#               scale: 30,
#               tileScale:4,
#               maxPixels: 1e13
#             });
#   var elev_min = dem_clip.reduceRegion({ 
#               reducer: ee.Reducer.min(),
#               geometry: feat.geometry(),
#               scale: 30,
#               tileScale:4,
#               maxPixels: 1e13
#             });
#   var elev_max = dem_clip.reduceRegion({ 
#               reducer: ee.Reducer.max(),
#               geometry: feat.geometry(),
#               scale: 30,
#               tileScale:4,
#               maxPixels: 1e13
#             });
  
#   var get_ic_counts = rolling_comps.map(function(img){ 
#       var pixel_ct_dict = img.reduceRegion({ 
#               reducer: ee.Reducer.mean(), //maybe change to median?  
#               geometry: feat.geometry(),
#               scale: 500,
#               tileScale:4,
#               maxPixels: 1e13
#             });
#     var dict_out = pixel_ct_dict.set('huc8',feat.get('huc8')).set('date',ee.Date(img.get('system:time_start'))).set('elev_mean',elev_mean.get('elevation'))
#     .set('elev_min',elev_min.get('elevation')).set('elev_max',elev_max.get('elevation')); 
#     var dict_feat = ee.Feature(null, dict_out);
#     return dict_feat; 
# }); 
#   return get_ic_counts; 
# }); 


# // Map.addLayer(comp_ic.first(),{},'test')

# Export.table.toDrive({
#   collection:ee.FeatureCollection(export_stats).flatten(),
#   description: 'modis_viirs_mean_stats_by_basin_'+start_date+'_'+(end_date)+'_huc8'+'_timestep_'+
#   ee.String(ee.Number(comp_time_step)).getInfo()+'_days',
#   folder:"modis_viirs",
#   fileNamePrefix:'modis_viirs_mean_stats_by_basin_'+start_date+'_'+(end_date)+'_huc8'+'_timestep_'+
#   ee.String(ee.Number(comp_time_step)).getInfo()+'_days',
#   fileFormat: 'csv'
# });

# test = make_daily_composites('2016-11-01','2017-04-30',75)
# #Map.addLayer(test.first(),{},'aslkjhg')
# print(test.getInfo(),'test')
#print(ee.Date(test.first().get('system:time_start')))


# def viirs_modis_snow(start_date,end_date,forest_thresh): 
# 	#get VIIRS data
# 	viirs = ee.ImageCollection("NOAA/VIIRS/001/VNP09GA"); 
# 	viirs_sel = viirs.filterDate(start_date,end_date); 
# 	# Map the cloud and water masking functions over the collection.
# 	viirs_cloud_masked = viirs_sel.map(lambda img: CreateViirsData(img).cloud_mask()) 
# 	viirs_cloud_masked = viirs_cloud_masked.map(lambda img: water_mask(img))
# 	#add the ndsi band
# 	viirs_snow = viirs_cloud_masked.map(lambda img: ndsi_calc(img))  
# 	#add an id band 
# 	viirs_snow = viirs_snow.map(lambda img: img.set('source','viirs'))

# 	#get the MODIS terra data
# 	terra_snow=ee.ImageCollection("MODIS/006/MOD10A1").filterDate(start_date,end_date).map(lambda img: img.set('source','modis'))
	                                                    
# 	#combine the two collections
# 	combined = terra_snow.select('NDSI_Snow_Cover').merge(viirs_snow.select('NDSI_Snow_Cover'));
# 	#cast to float and inherit source property 
# 	combined = combined.map(lambda img: img.float().set('source',img.get('source')))

# 	# combined = combined.map(function(img){

# 	#define a water mask 
# 	modis_water_mask=ee.Image("MODIS/006/MOD44W/2015_01_01").select('water_mask')
# 	#define a forest mask 
# 	forest_mask=ee.Image("USGS/NLCD_RELEASES/2016_REL/2016").select('percent_tree_cover').lte(forest_thresh)
# 	#apply a water mask
# 	combined = combined.map(lambda img: img.updateMask(modis_water_mask.eq(0))) 
# 	#apply a forest mask
# 	combined = combined.map(lambda img: img.updateMask(forest_mask.eq(0)))
# 	return combined #img.updateMask((ee.Ima