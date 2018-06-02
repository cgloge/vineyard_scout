# Vineyard Scout
A **Keras / TensorFlow** model that predicts whether a piece of land is suitable for growing wine grapes. It's currently limited to California, but could easily be expanded to other locations.

## The dataset
The california_vineyards.csv file includes the addresses of [vineyards in California](http://www.discovercaliforniawines.com/discover-california/wine-map-winery-directory/) along with land for sale on [Zillow](https://www.zillow.com/).

| Data source                       | Records |
| --------------------------------- | --------|
| Zillow land (California)          | 461     |
| Vineyards (California)            | 438     |

## Collecting weather & elevation data
If you want to generate your own data using the [Google Maps API](https://developers.google.com/api-client-library/python/start/get_started) & [WeatherBit API](https://www.weatherbit.io/api/weather-history-daily-bulk), you'll need to create a file called **config.py** which includes the following variables:

```
#Google maps API key
gmap_key = ''

#Weatherbit  API key
wbit_key = ''
```

The detailed weather & elevation data is created by the [address_to_land_data notebook](/address_to_land_data.ipynb) and is  exported as [a csv (for viewing purposed)](/california_vineyards_elevation_weather.csv) and as a [pkl file](/california_vineyards_elevation_weather.pkl) for importing into the [Keras / TensorFlow model](/build_vineyard_scout.ipynb) and [exploratory data analysis](/address_to_land_data.ipynb) scripts.

## Exploratory data analysis (EDA)
The [EDA notebook](/vineyard_eda.ipynb) imports the detailed elevation and weather data created by the [address_to_land_data notebook](/address_to_land_data.ipynb) and plots the locations of vineyards vs non-vineyards on the map of California, weather patterns over time, etc.

##  Keras / TensorFlow predictive neural network
The Keras [build_vineyard_scout](/build_vineyard_scout.ipynb) imports the [pkl file](/california_vineyards_elevation_weather.pkl) which contains the elevation & weather data as a pandas dataframe, and feeds in the various variables as inputs to the model:

* Latitude & longitude
* Matrix of elevation points based on 1km area around latitude & longitude coordinates
* Wind direction
* Wind speed
* Precipitation
* Average temperature
* Minimum temperature
* Max temperature
* Cloud coverage
* GHI (Global Horizontal Irradiance) - aka solar radiation
* RH (Relative humidity)

Hyperparameters and variables can be easily tweaked. Each setting is stored in a pandas dataframe for tracking with parameters have the strongest impact on the model. Regularization and dropout helped the model perform with such a small sample size.

After testing a few times, I typically saw **accuracy scores of 80-90%** for predicting if a location was a vineyard or not on the blind test set.
