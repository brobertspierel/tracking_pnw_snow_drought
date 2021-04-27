Calculate annual and sub-annual snow drought from SNOTEL and Daymet
===================================================================

SNOTEL
######

1. Acquire daily data from NRCS API using Climata for the 1981-2020 water years for: 
	
	a. Cumulative precipitation (in)
	b. Daily SWE (in)
	c. Average daily temperature (deg F)

2. Join the three datasets into one for later use. 
3. Add the huc ids by joining on the SNOTEL ID column
4. Take the daily mean for each basin for each time period to deal with basins that have multiple SNOTEL stations
5. 

