# Vineyard Scout
A **Keras / TensorFlow** model that predicts whether a piece of land is suitable for growing wine grapes. It's currently limited to California, but could easily be expanded to other locations.

## The dataset
The california_vineyards.csv file includes the addresses of [vineyards in California](http://www.discovercaliforniawines.com/discover-california/wine-map-winery-directory/) along with land for sale on [Zillow](https://www.zillow.com/).

| Data source                       | Records |
| --------------------------------- | --------|
| Zillow land (California)          | 461     |
| Vineyards (California)            | 438     |

## Weather & elevation data
If you want to generate your own data using the Google Maps API & WeatherBit API, you'll need to create a file called **config.py** which includes the following variables:

```
#Google maps API key
gmap_key = ''

#Weatherbit  API key
wbit_key = ''
```


