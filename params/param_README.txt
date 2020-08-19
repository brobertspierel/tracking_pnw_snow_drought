state_shapefile":"/vol/v1/general_files/user_files/ben/k_means/temp_files/oregon_state_grid_reprojected_epsg_4326.shp" #used to restrict the study area 
pnw_shapefile":"None" #background for mapping 
epsg": "4326" #output crs
output_filepath": "/vol/v1/general_files/user_files/ben/sentinel_1_snow/pickles/", 
season": one of either 'resample', 'core_winter','full_winter', or 'spring' these are to determine a subset of the snotel data 
read_from_pickle":"true", #depreceated? 
pickle_it": "true", #depreceated? 
csv_dir": used to get the sentinel 1 outputs from GEE
stations":"/vol/v1/general_files/user_files/ben/oregon_snotel_sites.csv" #csv file outlining the station specific data for snotel stations 
parameter":"WTEQ" #snotel station param- can be one of WTEQ (SWE in inches), TAVG (average daily temp in F), SNWD (snow depth in inches), SNOW (snowfall inches) more info can be found here: https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#elementCodes
start_date":"1985-10-01" 
end_date":"2019-09-30"
write_out":"true" #whether or not to pickle the snotel data. If you have already acquired the snotel data from the API there is no need for this to be set to true. Valid args are 'true' or anything else will be evaluated as false.  