state_shapefile":path/to/state_shapefile 
pnw_shapefile":"None" #background for mapping 
epsg": "4326" #output crs
output_filepath": "path/to/output", 
season": one of either 'resample', 'core_winter','full_winter', or 'spring' these are to determine a subset of the snotel data 
read_from_pickle":"true", #depreceated? 
csv_dir": used to get the sentinel 1 outputs from GEE
stations":"path/to/snotel_sites/shapefile" #csv file outlining the station specific data for snotel stations 
parameter":"WTEQ" #snotel station param- can be one of WTEQ (SWE in inches), TAVG (average daily temp in F), SNWD (snow depth in inches), SNOW (snowfall inches) more info can be found here: https://www.wcc.nrcs.usda.gov/web_service/AWDB_Web_Service_Reference.htm#elementCodes
start_date":"1985-10-01" 
end_date":"2019-09-30"
anom_start_date: #this is for getting a long term (climate normal) mean. Should be treated independently from the start and end dates that align with sentinel and/or modis
anom_end_date: #see the anom start date description
write_out":"true" #whether or not to pickle the snotel data. If you have already acquired the snotel data from the API there is no need for this to be set to true. Valid args are 'true' or anything else will be evaluated as false.  
anom_bool: this is to tell the prepare_data function whether to calculate the long term anomalies from the mean. This should be the default behavior. I am leaving the option to skip that for any future work that might require it. 