

Documentation for low snow/snow drought in the PNW
==================================================
Guide: 
^^^^^^
Documentation for the body of python scripts which acquire SNOTEL data, MODIS/VIIRS optical remote sensing data and Sentinel 1 remote sensing data. The RS data was processed in Google Earth Engine and downloaded to a local server for analysis. 

Dependencies: 
		* `climata <https://pypi.org/project/climata/>`_
		* numpy 
		* pandas
		* `geopandas <https://geopandas.org/>`_
		* matplotlib

This document outlines some workflows and processes used to generate data and figures for this project. 

.. toctree::
   :maxdepth: 1
   :caption: Script examples:
   Acquire SNOTEL data from NRCS <get_snotel>
   Calculate snow droughts from SNOTEL <calc_snow_drought>
   Plot snow drought on decadal scale <calc_plot_long_term_snow_drought>
   Collect all data for analysis <collect_all_data_class>
   Plot remote sensing data <calc_plot_rs_data>
   Merge remote sensing data and calculate stats <merge_rs_data_calc_stats>


.. toctree:: 
   :maxdepth: 1
   :titlesonly:
   
   license
   help
   

Indices and tables 
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
