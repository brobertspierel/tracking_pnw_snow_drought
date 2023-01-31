

Documentation for low snow/snow drought in the PNW
==================================================
Guide: 
^^^^^^
Documentation for the body of python scripts which acquire SNOTEL data and MODIS/VIIRS optical remote sensing data. The RS data was processed in Google Earth Engine and downloaded to a local server for analysis. These scripts and the associated figure generation code is associated with Roberts-Pierel, B., Raleigh, M., Kennedy, R., (XXXX) "Tracking the evolution of snow drought in the Pacific Northwest at variable scales." [Working title]

Dependencies: 
		* `climata <https://pypi.org/project/climata/>`_
		* numpy 
		* pandas
		* `geopandas <https://geopandas.org/>`_
		* matplotlib

This document outlines some workflows and processes used to generate data and figures for this project. Although the associated repo includes scripts for generating figures, this documentation focuses 
primarily on data acquisition and the process for calculating snow drought occurrences and types. 

.. toctree::
   :maxdepth: 2
   :caption: Script examples:

   Acquire SNOTEL data from NRCS <get_snotel.rst>
   Acquire Daymet data via GEE <get_daymet.rst>
   Define and calculate snow droughts <calc_snow_drought.rst>


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
