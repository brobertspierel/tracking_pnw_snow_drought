Calculate annual and sub-annual snow drought from SNOTEL and Daymet
===================================================================

Basic definitions for snow drought are based on work from Dierauer et al., 2019. "Snow Drought Risk and Susceptibility in the Western United States and Southwestern Canada". Means of calculating have been considerably altered. The work from Dierauer et al., as well as other work on the topic, uses a binary classification scheme where a snow drought determination is made principally on whether temperature, precip and peak SWE (or April 1 SWE) are above or below a long-term mean (climate normal). Here we try to increase the nuance of that definition by using the Dierauer et al., definitions to calculate intitalization centroids for a kmeans algorithm. This approach gives a classification of snow drought type (or no snow drought) and it also gives a distance to the cluster centroid. Therefore, this is not a binary classification but rather gives some indication of the relationship between a given snow drought unit (basin/season/year) and a baseline condition (long-term mean). The approach is tested for a point-based dataset (SNOTEL) and a distributed meteorological dataset (Daymet). Sample runs are included in the scripts listed below. 


Dependencies: 
		*snow_drought_definition_revision.py
		*_1_calculate_revised_snow_drought.py
		*_2_plot_sd_counts_comparison.py 
		*scikit-learn

