## Description
Custom python package that facilitates the creation of Google Earth Engine image composite, compute phenological spectral indices and segment images for classification in Earth Engine for Sentinel-2 and Landsat 5, 7, 8 sensors.

## Functions and Classes
The function marked as being bith function and method are functions that can be either called or used in a .map() method.
- `mask_sentinel_clouds()` : FUNCTION/ METHOD - masks clouds present in an image captured by Sentinel-2 sensor.
- `mask_landsat_clouds()` : FUNCTION/ METHOD - masks clouds present in an image captured by Landsat 5, 7 or 8 sensors.
- `sentinel2_spectral_indices()` : FUNCTION/ METHOD - compute phecological spectralindices for images capture by Sentinel-2 sensor.
- `landsat57_spectral_indices()` : FUNCTION/ METHOD - compute phecological spectralindices for images capture by Landsat 5 or 7 sensors.
- `landsat8_spectral_indices()` : FUNCTION/ METHOD - compute phecological spectralindices for images capture by Landsat 8 sensor.
- `segment_image()` : FUNCTION - segment the input image to help classifiers better distinguish between objects. 
- `buffer_size()` : METHOD - generates a buffer of input size around the centroid of an object.
- `get_metrics()` : FUNCTION - convert the input pre-processed TFRecord dataset into Bacthes Dataset ready to be fed to Kears deep models

## Tests
- `test_cloud_mask` - test the **mask_sentinel_clouds()** and **mask_landsat_clouds()** functions
- `test_compute_indices` - test the **sentinel2_spectral_indices()**, **landsat57_spectral_indices()**, and **landsat8_spectral_indices()** functions
- `test_image_segmentation` - test the **segment_image()** function

- No test were implemented for the **buffer_size()** function due to it being a very flexible method that only requires an integer as input.
- No test were implemented for the **get_metrics** function as it is a standalone that merely request numerical data from the Gogle server.