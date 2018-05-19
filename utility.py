#Update the config.py file with your own Google Maps & WeatherBit API keys to run
from config import *

#import Google Maps API modules
import googlemaps
from googlemaps import convert
from datetime import datetime
import requests

#Create the googlemaps object and pass it the API key stored in config.py
gmaps = googlemaps.Client(key=gmap_key)

#Default retry sleep - if API returns an error
gmaps_sleep = 1
weather_sleep = 1 


#Helper function which spits out latitude & longitude given a written address
def lat_lng(address):
    try:
    # Geocode an address
        geocode_result = gmaps.geocode(address)

        # Grab the location values from the returned dictionary
        location = geocode_result[0].get('geometry').get('location')

        #split in to lat & long coordinates
        lat = location.get('lat')
        lng = location.get('lng')

        #Reset the sleep value to 1 second if the API fails in the future
        gmaps_sleep = 1
        return(lat, lng)

    except Exception:
        print('Google Maps geocode failed - retrying')
        print(Exception)
        
        # sleep for a bit before trying again - increment the delay for each try
        time.sleep(gmaps_sleep)
        gmaps_sleep = gmaps_sleep + 2
        
        # try again
        return lat_lng(address)
    
#Get elevation for one set of latitude & longitude coordinates
def get_elevation(lat, lng):
    try:
        elevation = gmaps.elevation((lat, lng))[0].get('elevation')
        return elevation
                  
    except Exception:
        #Try again - I didn't add an additional sleep here since the lat, long geocoding handles that
        return get_elevation(lat, lng)
        
    
#Takes latitude and longitude coordinates and returns a 5x5 matrix of elevation points over roughly a 1 square km area 
def elevation_matrix(lat, lng):
    import numpy as np
    
    #Create a 5 x 5 matrix of elevation points using lat long points equivelant to a 1 kilometer area
    cols = 5
    rows = 5
    lat_lng_increment = .0002

    #Minus our lat & long starting coordinates by .005 to help us center our data
    lat = lat - (lat_lng_increment * cols / 2)
    lng = lng - (lat_lng_increment * rows / 2)

    array1 = []
    array2 = []
    for j in range(cols):
        #if we're on the first row, set the latitute back to initial value
        if (j == 0):
            lng_j = lng

        for i in range(rows):
            #if we're on the first row, set the latitute back to initial value
            if (i == 0):
                lat_i = lat   

            #get elevation for incremented latitude & longitude point
            elevation = get_elevation(lat_i, lng_j)
            array1.append(elevation)
            lat_i = lat_i + lat_lng_increment

        lng_j = lng_j + lat_lng_increment
        array2.append(array1)
        array1 = []

    altitude_matrix = np.array(array2)
    return(altitude_matrix)

#Returns the following historical weather data that includes the following for the dates passed to it:
def weather_hist(start_date, end_date, lat, lng):
    try:
        wbit_url = 'http://api.weatherbit.io/v1.0/history/daily?key=' + wbit_key + '&lat=' + str(lat) + '&lon=' + str(lng) + '&start_date=' + start_date + '&end_date=' + end_date
        r = requests.get(wbit_url).json().get('data')[0] 
                  
        #Reset the sleep value to 1 second if the API fails in the future         
        weather_sleep = 1          
        return r
    
    except Exception:
        print('Weather history failed - retrying')
        print(wbit_url)
        print(Exception)
        
        # sleep for a bit in case that helps
        time.sleep(weather_sleep)
        weather_sleep = weather_sleep + 2
        
                  
        # try again
        return weather_hist(start_date, end_date, lat, lng)
