import ee
import pprint 
import pandas as pd 
import numpy as np 

#deal with authentication
# authenticate = ee.Authenticate()
# print('authenticate')
# print(authenticate)
ee.Initialize()


class GetDaymet(): 
	def __init__(self,aoi,start_year,start_month,end_month,**kwargs): 
		self.start_year=start_year
		self.start_month=start_month
		self.end_month=end_month
		self.aoi=aoi
		#set the rest of the vars from kwargs (end dates)

	def get_data(self): 
		daymet_ic = (ee.ImageCollection("NASA/ORNL/DAYMET_V4").filterBounds(self.aoi)
															  .filter(ee.Filter.calendarRange(self.start_year,self.start_year,'year'))
															  .filter(ee.Filter.calendarRange(self.start_month,self.end_month,'month')))
															  #.filter(ee.Filter.calendarRange(self.start_day,self.end_day,'day_of_month')))
		return daymet_ic

	def average_temp(self,img): 
		"""Take the average of tmax and tmin."""
		tavg = (img.select('tmax').add(img.select('tmin'))).divide(ee.Image(2))#daymet_ic.map(lambda img: ((img.select('tmax').add(img.select('tmin'))).divide(ee.Image(2))))

		return img.addBands(ee.Image(tavg)) 


	def get_ics(self): 
		ic = self.get_data()
		#tavg is a bit more complicated, do that one next 
		output = ic.map(lambda img: self.average_temp(img))
		#get a list of the band names in an image of the tavg ic
		bands=output.first().bandNames()
		
		try: 
			if bands.contains('tmin_1').getInfo(): 
				output=output.select(bands,bands.replace('tmin_1','tavg'))
			elif bands.contains('tmax_1').getInfo(): 
				output=output.select(bands,bands.replace('tmax_1','tavg'))
		except KeyError as e: 
			print('The default col tmin_1 does not exist.')
			print(f'The bands in the tavg ic look like: {bands}')
		return output



class ExportStats(): 

	def __init__(self,ic,features,scale=1000,**kwargs):#,end_date,start_date='1980-10-01'): #make sure to define a different start date if you want something else 
		self.ic=ic
		self.scale=scale
		self.features = features#kwargs.get('features')

		for key, value in kwargs.items():
			setattr(self, key, value)

	def calc_export_stats(self,feat,img): 
		"""Use a reducer to calc spatial stats."""
		# var get_ic_counts = comp_ic.map(function(img){ 
		pixel_ct_dict = img.reduceRegion(
			reducer=ee.Reducer.mean(), #maybe change to median? This get's the basin mean for a given day (image)
			geometry=feat.geometry(),
			scale=self.scale,
			tileScale=4,
			maxPixels=1e13
			)
		dict_out = pixel_ct_dict.set('huc8',feat.get('huc8')).set('date',ee.Date(img.get('system:time_start')))
		dict_feat = ee.Feature(None, dict_out)
		return dict_feat

	def generate_stats(self): 
				
		get_ic_stats=ee.FeatureCollection(self.features).map(lambda feat: self.ic.map(lambda img: self.calc_export_stats(feat,img)))

		return get_ic_stats
	

	def run_exports(self): 
		"""Export some data."""

		task=ee.batch.Export.table.toDrive(
			collection=ee.FeatureCollection(self.generate_stats()).flatten(),
			description= 'py_daymet_mean_stats_by_basin_'+self.timeframe+'_huc8',
			folder="daymet_outputs",
			fileNamePrefix='py_daymet_mean_stats_by_basin_'+self.timeframe+'_huc8',
			fileFormat= 'csv'
			)
		#start the task in GEE 
		print(task)
		task.start()


def main(hucs): 

	for year in range(1980,2021): #years: this is exclusive 
		for m in [[11,12],[1,2],[3,4]]: 
			try: 
				ic = GetDaymet(hucs.first().geometry(),start_year=year,start_month=m[0],end_month=m[1]).get_ics()
			except IndexError as e: 
				pass 
			exports = ExportStats(ic,features=hucs,timeframe=f'start_date_{year}_{m[0]}_end_date_{year}_{m[1]}').run_exports()
if __name__ == '__main__':
	
	pnw_snotel = ee.FeatureCollection("users/ak_glaciers/NWCC_high_resolution_coordinates_2019_hucs")
	hucs = ee.FeatureCollection("USGS/WBD/2017/HUC08").filterBounds(pnw_snotel)
	
	main(hucs)

########################################################################
#composites
# class MakeComposites(): 
# 	"""Create 7 or 12 day composites depending on other data."""

# 	def __init__(self,ic,start_date,end_date,aoi,comp_time_step=12): 
# 		self.ic=ic
# 		self.start_date=start_date
# 		self.end_date=end_date
# 		self.comp_time_step=comp_time_step
# 		#self.aoi=aoi 

# 	def week_difference(self): 
# 		return ee.Date(self.start_date).advance(self.comp_time_step, 'days').millis().subtract(ee.Date(self.start_date).millis())

# 	def list_map(self): 
# 		return ee.List.sequence(ee.Date(self.start_date).millis(), 
# 							ee.Date(self.end_date).millis(), 
# 							self.week_difference())

# 	# def date_list(self): #might need to change the way the dates work on this because its set up for just one year at the moment  
# 	# 	return self.list_map()

# 	def make_composites(self,date):
# 		date=ee.Date(date)
# 		output = (ee.ImageCollection(ic).filterDate(date, date.advance(self.comp_time_step, 'days'))
# 									   .filterBounds(self.aoi))
# 		return output.median().set('system:time_start',ee.Date(date)) #note that the stat here is hardcoded and could be changed

# 	def run_comps(self,aoi): 
# 		#date = ee.Date(date);
# 		return ee.ImageCollection.fromImages(self.list_map().map(lambda dateMillis: self.make_composites(dateMillis)))
		#return self.make_composites(date)


	# pprint.pprint(ic.first().bandNames().getInfo())

	# #make multi-day composites 
	# comps = MakeComposites(ic,'2000-10-01','2001-04-30',hucs.first().geometry())
	
	# exports = ExportStats(ic,'2000-10-01','2001-04-30',hucs.first().geometry())

	# exports.run_exports()
	# collection=test.generate_comps()
	# print(collection.size().getInfo())


	#date_list = comps.list_map()
	#script to call the compositing functions 
#   	var optical_ic = optical_funcs.make_daily_composites(obs_date_start,obs_date_end,forest_thresh); 
# //create the composites 
	#comp_ic = ee.ImageCollection.fromImages(date_list.map(lambda dateMillis: comps.run_comps(dateMillis)))
	#print(comp_ic.size().getInfo())
# optical_ic = ee.ImageCollection.fromImages(date_list.map(function(dateMillis){
#   var date = ee.Date(dateMillis);
#   return optical_funcs.make_sca_composite(date,comp_time_step,hucs.first().geometry(),optical_ic,forest_thresh); //change the function here to generate SP or SCA
#   }));


#   	exports.make_sca_composite = function (date,time_step,aoi,ic) {
#   date = ee.Date(date);
#   var output = ee.ImageCollection(ic)
#                       .filterDate(date, date.advance(time_step, 'days'))
#                       .filterBounds(aoi)
#                       .select('NDSI_Snow_Cover')
#   var ndsi_ic = output.map(function(img){
#     var binary = (img.where(img.select('NDSI_Snow_Cover').gte(40), 1)).where(img.select('NDSI_Snow_Cover').lt(40),0); 
#     return binary; 
#   }); 
  
#   return ndsi_ic.mode().set('system:time_start',ee.Date(date)); //note that the choice of composite stats here could impact the output image 
# }; 





  	# def clean_dates(self,date):
  	# 	 return ee.Date.parse('yyyy_MM_dd',(ee.String(x).slice(2)));  #GEE adds an id (e.g. 1_date), remove it 

  	# def make_daily_composites(self)

  	# 	#get a list of dates from ic 
  	# 	dates = ic.aggregate_array('system:index')

  	# 	#remove some extra stuff from the dates
  	# 	dates = dates.map(lambda img: self.clean_dates(img))

  	# 	dates = dates.distinct() #get unique elements 


  		# dates = dates.map(function(x){
    # return ee.Date.parse('yyyy_MM_dd',(ee.String(x).slice(2)));  //GEE adds an id (e.g. 1_date), remove it 

  # 	exports.make_daily_composites = function(start_date,end_date,forest_thresh){
  # //get the snow imageCollection
  # var ic = viirs_modis_snow(start_date,end_date,forest_thresh); 
  # //get all of the dates in the collection as a list
  # var dates = ic.aggregate_array('system:index'); 
  # dates = dates.map(function(x){
  #   return ee.Date.parse('yyyy_MM_dd',(ee.String(x).slice(2)));  //GEE adds an id (e.g. 1_date), remove it 
  # }); 
  # dates = dates.distinct(); //get unique elements 

#   //there are two images for each day in the collection (one from MODIS and one from VIIRS). Take the median image and reset the date. 
#   var ndsi_combined = dates.map(function(date){
#     return ic.filterDate(date).median().set('system:time_start',ee.Date(date).millis()); 
#   });
#   return ee.ImageCollection.fromImages(ndsi_combined); 
# }; 

# //make a collection at a user-defined number of days of composite sp images
# exports.make_sp_composite = function (date,time_step,aoi,ic) {
#   date = ee.Date(date);
#   var output = ee.ImageCollection(ic)
#                       .filterDate(date, date.advance(time_step, 'days'))
#                       .filterBounds(aoi)
#                       .select('NDSI_Snow_Cover')
#   var ndsi_ic = output.map(function(img){
#     var binary = (img.where(img.select('NDSI_Snow_Cover').gte(40), 1)).where(img.select('NDSI_Snow_Cover').lt(40),0); //threshold hardcoded at 40 (0.4)
#     return binary; 
#   }); 
#   var sp = ndsi_ic.sum().divide(output.count()); 
#   return sp.set('system:time_start',ee.Date(date)); 
# }; 

# //this is a little cumbersome and should just be split into one function and a second that calls it but it was left for ease of use
# //make a collection of composite images at a user-defined number of days 


# var list_map = function(startDate,endDate){
  # return ee.List.sequence(ee.Date(startDate).millis(), ee.Date(endDate).millis(), week_difference(startDate))};
 	# week_difference = function(startDate){
 #  return ee.Date(startDate).advance(comp_time_step, 'days').millis().subtract(ee.Date(startDate).millis())};  
#//create list of start dates for composites
# var date_list = list_map(obs_date_start,obs_date_end); 