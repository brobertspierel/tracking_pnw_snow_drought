import ee
import geopandas as gpd
#there must be a better way to deal with this just once 
#deal with authentication
try: 
  ee.Initialize()
except Exception as e: 
  authenticate = ee.Authenticate()
  ee.Initialize()

###################################################################################
###################################################################################
##viirs snow product https:##modis-snow-ice.gsfc.nasa.gov#uploads#VIIRS-snow-products-user-guide-version-1.pdf

##use bitmasks to remove clouds, ocean and inland water 
##from https:##code.earthengine.google.com#274c5d85621c8fc4142d87cbd867d187

def getQABits(image, start, end, newName): 
    ## Compute the bits we need to extract.
    pattern = 0 
    for i in range(start,end): 
      pattern += 2**i
    ## Return a single band image of the extracted QA bits, giving the band
    ## a new name.
    return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

## --- Mask out cloudy pixels - cloud state  from state QA band (bits 3 and 4)
def cloud_mask(image): 
  ## Select the QA band.
  QA = image.select('QF1') 
  ## Get the internal_cloud_algorithm_flag bit.
  cloud = getQABits(QA, 2, 3, 'cloud_state').expression("b(0) == 2 || b(0) == 3")
  ## Return an image masking out cloudy areas.
  return image.updateMask(cloud.Not()) ##.internalCloud.eq(0)) 

def water_mask(image):
  QA = image.select('QF2')  
  water = getQABits(QA,1,2,'water').expression("b(0) == 1 || b(0) == 2")

  return image.updateMask(water.Not())   

##add a ndsi band to the VIIRS data
def ndsi_calc(img):
  ndsi = img.normalizedDifference(['I1','I3']) 
  pos_ndsi = ndsi.updateMask(ndsi.gte(0)).multiply(100).round()
  new_img = img.addBands(pos_ndsi)  
  return new_img.select(['nd','QF1'],['NDSI_Snow_Cover','NDSI_Snow_Cover_Class']) ##rename to match the modis product

def viirs_only(start_date,end_date): 
  ##get VIIRS data
  viirs = ee.ImageCollection("NOAA/VIIRS/001/VNP09GA")
  viirs_sel = viirs.filterDate(start_date,end_date)  
  ## Map the cloud and water masking functions over the collection.
  viirs_cloud_masked = viirs_sel.map(cloud_mask).map(water_mask) 
  ##add the ndsi band
  viirs_snow = viirs_cloud_masked.map(ndsi_calc) 
  viirs_snow = viirs_snow.map(lambda img: img.set('source','viirs')) 
  return viirs_snow

def viirs_modis_snow(start_date,end_date): 
  ##get VIIRS data
  viirs = ee.ImageCollection("NOAA/VIIRS/001/VNP09GA")
  viirs_sel = viirs.filterDate(start_date,end_date)  
  ## Map the cloud and water masking functions over the collection.
  viirs_cloud_masked = viirs_sel.map(cloud_mask).map(water_mask) 
  ##add the ndsi band
  viirs_snow = viirs_cloud_masked.map(ndsi_calc) 
  viirs_snow = viirs_snow.map(lambda img: img.set('source','viirs')) 

  ##get the MODIS terra data
  terra_snow=ee.ImageCollection("MODIS/006/MOD10A1").filterDate(start_date,end_date).map(lambda img: img.set('source','modis')) 
                                                        
  ##combine the two collections
  combined = terra_snow.select('NDSI_Snow_Cover').merge(viirs_snow.select('NDSI_Snow_Cover')) 
  combined = combined.map(lambda img: img.float().set('source',img.get('source')))
  return combined 

##combine the two datasets where they have overlapping dates with a median daily composite 
def make_daily_composites(start_date,end_date): 
  ##get the snow imageCollection
  ic = viirs_modis_snow(start_date,end_date)  
  ##get all of the dates in the collection as a list
  dates = ic.aggregate_array('system:index')  
  dates = dates.map(lambda x: ee.Date.parse('yyyy_MM_dd',(ee.String(x).slice(2)))) 
  dates = dates.distinct()  ##get unique elements 

  ##there are two images for each day in the collection (one from MODIS and one from VIIRS). Take the median image and reset the date. 
  ndsi_combined = dates.map(lambda date: ic.filterDate(date).median().set('system:time_start',ee.Date(date).millis()).set('date',ee.Date(date))) 
  return ee.ImageCollection.fromImages(ndsi_combined)  

def fill_gaps(date, ic): 
  """Infill pixels with days closest to the target day, moving backwards in time one day then forwards one day. 
  in this manner preference is given to days in the past but higher preference is given to days closer to the target day. 
  there is a cleaner way to do this
  """
  date = ee.Date(date)  
  ##get the central image, the day in question 
  center =  ic.filterDate(date.format('YYYY-MM-dd')).first().selfMask()   
  
  ##move one day forward and one day back 
  min_1 =  ic.filterDate(date.advance(-1,'days').format('YYYY-MM-dd')).first() 
  center = center.unmask(min_1)  
  plus_1 =  ic.filterDate(date.advance(1,'days').format('YYYY-MM-dd')).first() ##,date.format('YYYY-MM-dd')) 
  center = center.unmask(plus_1)  
  ##move two days forward and two back 
  min_2 =  ic.filterDate(date.advance(-2,'days').format('YYYY-MM-dd')).first() ##,date.format('YYYY-MM-dd')) 
  center = center.unmask(min_2)  
  plus_2 =  ic.filterDate(date.advance(2,'days').format('YYYY-MM-dd')).first() ##,date.format('YYYY-MM-dd')) 
  center = center.unmask(plus_2) 
  ##move three days forward and three back 
  min_3 =  ic.filterDate(date.advance(-3,'days').format('YYYY-MM-dd')).first() ##,date.format('YYYY-MM-dd')) 
  center = center.unmask(min_3)  
  plus_3 =  ic.filterDate(date.advance(3,'days').format('YYYY-MM-dd')).first() ##,date.format('YYYY-MM-dd')) 
  center = center.unmask(plus_3) 
  ##move four days forward and four back 
  min_4 =  ic.filterDate(date.advance(-4,'days').format('YYYY-MM-dd')).first() ##,date.format('YYYY-MM-dd')) 
  center = center.unmask(min_4)  
  plus_4 =  ic.filterDate(date.advance(4,'days').format('YYYY-MM-dd')).first() ##,date.format('YYYY-MM-dd')) 
  center = center.unmask(plus_4) 
  
  return center

def binarize(img,band_name='NDSI_Snow_Cover'): 
  """Convert the NDSI band from a 0-100 scale to binary 0/1 based on 0.4 threshold."""
  return img.select('NDSI_Snow_Cover').gte(40).set('system:time_start',img.get('system:time_start')) #make sure the time gets inherited from the image

####################################################################
#do the exports 
def calc_export_stats(feat,img): 
	"""Use a reducer to calc spatial stats."""
	# var get_ic_counts = comp_ic.map(function(img){ 
	pixel_ct_dict = img.reduceRegion(
		reducer=ee.Reducer.mean(), #TESTING the mean values to determine bias btwn MODIS and VIIRS (8/22/2021) gets the basin sum for a binary raster 
		geometry=feat.geometry(),
		scale=500,
		tileScale=4,
		maxPixels=1e13
		)
	dict_out = pixel_ct_dict.set('basin',feat.get('STAID')).set('date',ee.Date(img.get('system:time_start')))
	dict_feat = ee.Feature(None, dict_out) 
	return dict_feat

def generate_stats(fc, ic): 
	"""Iterator for calc_export_stats."""
	get_ic_stats=ee.FeatureCollection(fc).map(lambda feat: ic.map(lambda img: calc_export_stats(feat,img)))

	return get_ic_stats

def run_exports(fc, ic, start_date, end_date): 
	"""Export some data."""

	task=ee.batch.Export.table.toDrive(
		collection=ee.FeatureCollection(generate_stats(fc, ic)).flatten(),
		description= f'VIIRS_daily_center_comps_{start_date}_{end_date}_USGS_ref_basins_no_forest_cor_mean',
		folder="chapter_3",
		fileNamePrefix=f'VIIRS_daily_center_comps_{start_date}_{end_date}_USGS_ref_basins_no_forest_cor_mean',
		fileFormat= 'csv'
		)
	#start the task in GEE 
	print(task)
	task.start()

def main(ref_gages): 
  #get the basins of interest 
  wus = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.inList('NAME',['Washington','Oregon','California','Arizona','Nevada','Idaho','Montana','Utah','New Mexico','Colorado','Wyoming'])) 

  ref_basins = ee.FeatureCollection('users/ak_glaciers/bas_ref_all')
  ref_basins = ref_basins.filterBounds(wus).filter(ee.Filter.inList('GAGE_ID', ref_gages))

  ###############################################################################
  ##create an imageCollection from the full time period 
  ###############################################################################

  for year in range(2000,2021): 
    start_date = f'{year}-10-01'
    end_date = f'{year+1}-09-30' #need to add a year to span the start of the year 
    if year + 1 <= 2020: 
      #make the combined MODIS/VIIRS imageCollection 
      full_ic = make_daily_composites(start_date, end_date).filterBounds(ref_basins)
      #test an ic that is just MODIS (8/17/2021)
      terra_snow=ee.ImageCollection("MODIS/006/MOD10A1").filterDate(start_date,end_date).map(lambda img: img.set('source','modis'))
      viirs_snow = viirs_only(start_date,end_date)
      #get a list of dates from the inputs and cast to ee.Date type 
      ic_dates = viirs_snow.aggregate_array('system:time_start') 
      #move the time series forward four days so there are no empty days 
      #ic_dates = ic_dates.slice(4,-5)
      ic_dates = ic_dates.map(lambda date: ee.Date(date))
      #print(ic_dates.getInfo())
      #gap fill by stepping back a day, then forward a day then two days back, two days forward etc. 
      gap_filled = ic_dates.map(lambda date: fill_gaps(date,viirs_snow))
      #gap filled imageCollection for the full period (currently a year as of 7/20/2021)
      gap_filled = ee.ImageCollection.fromImages(gap_filled)
       #binarize the ndsi band to make SCA 
      #gap_filled = gap_filled.map(lambda img: binarize(img))
      exports = run_exports(ref_basins, gap_filled, start_date, end_date)
    else: 
      break
if __name__ == '__main__':
  forest_thresh = 75

  # ref_gages = ee.List(['9334000', '9489100', '12209500', '12483800', '13046680', '13340500', '9226500', '9220500', '6092500', '6297000', 
  # '9505300', '13092000', '6197500', '12097000', '8321500', '9307500', '9060500', '13334700', '9419610', '9278500', '10183900', '14101500', 
  # '14232500', '8236000', '9052000', '12209000', '14134000', '9245000', '12096500', '6102500', '9415460', '12044900', '13334450', '12207850', 
  # '8386505', '8315480', '12206900', '12207750', '12209490', '12413875', '13162225', '6187915', '6190540', '12097850', '14231000', 
  # '8277470', '10243260', '10336674', '10336770', '14216000', '12392155', '12456500', '6187950', '9107000', '10348850', '12143600', 
  # '13339500', '6746095', '12144000', '9312600', '13340600', '12095000', '12097500', '10205030', '12094000', '12143400', '6191000', 
  # '9492400', '9494000', '12092000', '6280300', '9081600', '8267500', '8380500', '10109001', '10234500', '10242000', '10316500', '10329500', 
  # '10396000', '12082500', '12175500', '12358500', '12413000', '12414500', '12451000', '13185000', '13337000', '14158500', '12210000', '14136500',
  # '9217900', '9444200', '10336676', '13310700', '14159200', '9505200', '6188000', '9404450', '13161500', '10173450', '13338500', '13340000', '14216500', 
  # '13296500', '9505350', '10308200', '10172700', '12452800', '6218500', '6043500', '6632400', '9223000', '12411000'])

  # ref_gages = ee.List(['10109001', '10172700', '10173450', '10205030', '10234500', '10242000', '10308200', '10316500', '10329500', 
  # '10336676', '10348850', '10396000', '12082500', '12092000', '12095000', '12097500', '12143400', '12143600', '12144000', 
  # '12175500', '12358500', '12392155', '12411000', '12413000', '12414500', '12451000', '12452800', '13161500', '13185000', 
  # '13296500', '13310700', '13337000', '13338500', '13339500', '13340000', '13340600', '14158500', '14216500']) 

  filtered_gages = "/vol/v1/general_files/user_files/ben/chapter_3/streamflow_data/shapefiles/gage_basin_data/snowy_basins/filtered_snotel_stations_ratio_10_percent.shp"
  ref_gages = gpd.read_file(filtered_gages)
  print(ref_gages.shape)
  ref_gages = ee.List(list(ref_gages['STAID'].astype('str')))

  main(ref_gages)