

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

.. toctree::
   :maxdepth: 1
   :caption: Script examples:
   1. Acquire SNOTEL data from NRCS <get_snotel>
   2. Calculate snow droughts from SNOTEL <calc_snow_drought>
   3. Plot snow drought on decadal scale <calc_plot_long_term_snow_drought>
   4. Collect all data for analysis <collect_all_data_class>
   5. Plot remote sensing data <calc_plot_rs_data>
   6. Merge remote sensing data and calculate stats <merge_rs_data_calc_stats>


Indices and tables 
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
