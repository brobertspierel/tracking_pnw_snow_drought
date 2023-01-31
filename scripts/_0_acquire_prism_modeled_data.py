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

class GetPrism(): 
	"""Sets up the Daymet imageCollection for GEE. Calculates an average 
	temperature band from tmin and tmax and adds that to the collection. 
	Performs spatial and temporal filtering. 
	"""

	def __init__(self,start_year,start_month,end_month,**kwargs): 
		self.start_year=start_year
		self.start_month=start_month
		self.end_month=end_month
		#set the rest of the vars from kwargs (end dates)

	def get_data(self): 
		prism_ic = (ee.ImageCollection("OREGONSTATE/PRISM/AN81d")#.filterBounds(self.aoi) #not 100% sure why this was filtering to the first aoi 9/6/2021
																.filter(ee.Filter.calendarRange(self.start_year,self.start_year,'year'))
																.filter(ee.Filter.calendarRange(self.start_month,self.end_month,'month'))
																.select(['ppt','tmean']))
															  #.filter(ee.Filter.calendarRange(self.start_day,self.end_day,'day_of_month')))
		return prism_ic

class ExportStats(): 

	def __init__(self,ic,features,scale=4000,**kwargs): #make sure to define a different start date if you want something else 
		self.ic=ic
		self.scale=scale
		self.features = features#kwargs.get('features')

		for key, value in kwargs.items():
			setattr(self, key, value)

	def calc_export_stats(self,feat,img): 
		"""Use a reducer to calc spatial stats."""
		# var get_ic_counts = comp_ic.map(function(img){ 
		pixel_ct_dict = img.reduceRegion(
			reducer=self.reducer, #maybe change to median? This get's the basin mean for a given day (image)
			geometry=feat.geometry(),
			scale=self.scale,
			crs='EPSG:4326', 
			tileScale=4,
			maxPixels=1e13
			)
		dict_out = pixel_ct_dict.set(self.huc,feat.get(self.huc)).set('date',ee.Date(img.get('system:time_start')))
		dict_feat = ee.Feature(None, dict_out)
		return dict_feat

	def generate_stats(self): 
		"""Iterator function for the calc_export_stats function. 
		This emulates the nested functions approach in GEE Javascript API."""

		get_ic_stats=ee.FeatureCollection(self.features).map(lambda feat: self.ic.map(lambda img: self.calc_export_stats(feat,img)))
		return get_ic_stats

	def run_exports(self): 
		"""Export some data."""

		task=ee.batch.Export.table.toDrive(
			collection=ee.FeatureCollection(self.generate_stats()).flatten(),
			description= f'proj_prism_mean_stats_for_{self.modifier}_{self.timeframe}', 
			folder=self.output_folder,
			fileNamePrefix=f'proj_prism_mean_stats_for_{self.modifier}_{self.timeframe}',
			fileFormat= 'csv'
			)
		#start the task in GEE 
		print(task)
		task.start()

def main(hucs): 

	for wy in range(1990,2021): #years: this is exclusive 
		
		for m in [[11,12],[1,2],[3,4]]: #these are the seasonal periods (months) used in analysis 	
			#generate the water year name first because that should be agnostic of the month 
			timeframe = f'start_month_{m[0]}_end_month_{m[1]}_WY{wy}'
			try: 
				if m[1] == 12: 
					#for the fall years, set the water year back one integer year. 
					#Note that this will still be labeled as the water year (preceeding year)
					amended_wy = wy-1
				else: 
					#for the winter and spring months reset the water year 
					#to the original value because we don't want the previous year for that one
					amended_wy = wy 
				ic = GetPrism(
					start_year=amended_wy,
					start_month=m[0],
					end_month=m[1]
					).get_data()
			except IndexError as e: 
				pass 
			
			#run the exports- note that default is to generate stats for a HUC level (e.g. 6,8) but this can be run as points (e.g. snotel). 
			#you should change the reducer to first and then make sure to change the huc variable to whatever the point dataset id col is. 
			exports = ExportStats(ic,features=hucs,
								timeframe=timeframe,
								huc='site_num',
								reducer=ee.Reducer.first(), #change to mean for running basins, first for points
								output_folder='prism_outputs', 
								modifier='SNOTEL'
								).run_exports()

if __name__ == '__main__':
	#note that the default setup for running this is to use HUCS (polygons) which demands the mean reducer. 
	#one should also be able to run this in point mode 
	pnw_snotel = ee.FeatureCollection("users/ak_glaciers/NWCC_high_resolution_coordinates_2019_hucs")
	hucs = ee.FeatureCollection("USGS/WBD/2017/HUC06").filterBounds(pnw_snotel)
	
	main(pnw_snotel)

