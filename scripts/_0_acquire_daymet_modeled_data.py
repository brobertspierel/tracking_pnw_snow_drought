import ee

"""
Use the Google Earth Engine Python API to create tasks for generating yearly/seasonal Daymet estimates 
for HUCx river basins in the US Pacific Northwest (PNW). NOTE that you need to have an active GEE and associated 
Google Drive account for this script to work. You will be prompted to log into your account in the ee.Authenticate call. 

Inputs: 
Specify the HUC level 
Include any other inputs to spatially bound the river basins- here we use SNOTEL stations. 

Outputs: 
Script will start spatial statistics tasks (GEE command Export.table.toDrive) on your GEE account and export 
to the associated Google Drive account and specified folder. 
"""

#deal with authentication
try: 
	ee.Initialize()
except Exception as e: 
	ee.Authenticate()
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
		tavg = (img.select('tmax').add(img.select('tmin'))).divide(ee.Image(2))

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

	def __init__(self,ic,features,scale=1000,**kwargs): #make sure to define a different start date if you want something else 
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
		dict_out = pixel_ct_dict.set(self.huc,feat.get(self.huc)).set('date',ee.Date(img.get('system:time_start')))
		dict_feat = ee.Feature(None, dict_out)
		return dict_feat

	def generate_stats(self): 
				
		get_ic_stats=ee.FeatureCollection(self.features).map(lambda feat: self.ic.map(lambda img: self.calc_export_stats(feat,img)))

		return get_ic_stats

	def run_exports(self,output_folder): 
		"""Export some data."""

		task=ee.batch.Export.table.toDrive(
			collection=ee.FeatureCollection(self.generate_stats()).flatten(),
			description= f'py_daymet_mean_stats_by_basin_{self.timeframe}_{self.huc}', #these filenames are hardcoded 
			folder=output_folder,
			fileNamePrefix=f'py_daymet_mean_stats_by_basin_{self.timeframe}_{self.huc}',
			fileFormat= 'csv'
			)
		#start the task in GEE 
		print(task)
		task.start()

def main(hucs): 

	for year in range(1980,2021): #years: this is exclusive 
		for m in [[11,12],[1,2],[3,4]]: #these are the seasonal periods (months) used in analysis 
			try: 
				ic = GetDaymet(hucs.first().geometry(),start_year=year,start_month=m[0],end_month=m[1]).get_ics()
			except IndexError as e: 
				pass 
			#run the exports 
			exports = ExportStats(ic,features=hucs,timeframe=f'start_date_{year}_{m[0]}_end_date_{year}_{m[1]}',huc='huc6').run_exports(output_folder)

if __name__ == '__main__':
	
	pnw_snotel = ee.FeatureCollection("users/ak_glaciers/NWCC_high_resolution_coordinates_2019_hucs")
	hucs = ee.FeatureCollection("USGS/WBD/2017/HUC06").filterBounds(pnw_snotel)
	
	main(hucs)

