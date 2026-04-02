import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def save_data():
    h_data, f_data = prepare_data()

    print("Saving data as treated_historic_data.csv")
    h_data.to_csv('data/treated_historic_data.csv')

    print("Saving data as treated_forecast_data.csv")
    f_data.to_csv('data/treated_forecast_data.csv')

def prepare_data():
    # Read historic data
    print("Reading historic data...")
    h_data = pd.read_csv("data/historic_data.csv")

    # Read forecast data
    print("Reading forecast data...")
    f_data = pd.read_csv("data/forecast_data.csv")

    # Set date type to date column
    print("Adding datetime type to date column")
    h_data["date"] = pd.to_datetime(h_data["date"], utc=True)
    f_data["date"] = pd.to_datetime(f_data["date"], utc=True)

    # Set description of columns
    print("Adding feature descriptions to data")
    set_historic_feature_description(h_data)
    set_forecast_feature_description(f_data)

    # Write zone columns
    locations = ['seville', 'cadiz', 'huelva', 'malaga', 'cordoba', 'badajoz', 'lisbon', 'c_real']
    print("Adding location to the data")
    write_zones_to_data(h_data, locations)
    write_zones_to_data(f_data, locations)

    # Unstack data
    features = ['date','latitude', 'temperature_2m', 'relative_humidity_2m',
                'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
                'cloud_cover', 'precipitation']
    
    print("Widening historic data...")
    new_h_data = convert_wide_data(h_data, features)
    new_f_data = convert_wide_data(f_data, features)

    # Add periodic variables
    print("Adding sin and cos features to data")
    add_sin_cos_hour_data(new_h_data)
    add_sin_cos_hour_data(new_f_data)

    # Add lags
    print("Adding lags...")
    target = 'temperature_2m_seville'
    add_lags_to_data(new_h_data, target)
    add_lags_to_data(new_f_data, target)

    # Remove NaN's
    print("Removing NaN values from data")
    new_h_data = new_h_data.dropna()
    new_f_data = new_f_data.dropna()
    
    return new_h_data, new_f_data


def set_historic_feature_description(data):
    data.attrs['description'] = {
        'date': 'Time and date',
        'latitude': 'Geographical WGS84 coordinates of the location. Multiple coordinates can be comma separated. E.g. &latitude=52.52,48.85&longitude=13.41,2.35. To return data for multiple locations the JSON output changes to a list of structures. CSV and XLSX formats add a column location_id. For North and South America locations use negative longitudes, because they lie west of Greenwich.',
        'longitude': 'Geographical WGS84 coordinates of the location. Multiple coordinates can be comma separated. E.g. &latitude=52.52,48.85&longitude=13.41,2.35. To return data for multiple locations the JSON output changes to a list of structures. CSV and XLSX formats add a column location_id. For North and South America locations use negative longitudes, because they lie west of Greenwich.',
        'elevation': 'The elevation used for statistical downscaling. Per default, a 90 meter digital elevation model is used. You can manually set the elevation to correctly match mountain peaks. If &elevation=nan is specified, downscaling will be disabled and the API uses the average grid-cell height. For multiple locations, elevation can also be comma separated.',
        'temperature_2m': 'Air temperature at 2 meters above ground (Celsius degrees)',
        'relative_humidity_2m': 'Relative humidity at 2 meters above ground',
        'dew_point_2m': 'Dew point temperature at 2 meters above ground (Celsius degrees)',
        'apparent_temperature': 'Apparent temperature is the perceived feels-like temperature combining wind chill factor, relative humidity and solar radiation (Celsius degrees)',
        'precipitation': 'Total precipitation (rain, showers, snow) sum of the preceding hour (mm)',
        'rain': 'Rain from large scale weather systems of the preceding hour in millimeter (mm)',
        'snowfall': 'Snowfall amount of the preceding hour in centimeters. For the water equivalent in millimeter, divide by 7. E.g. 7 cm snow = 10 mm precipitation water equivalent (cm)',
        'snow_depth': 'Snow depth on the ground (meters)',
        'weather_code': 'Weather condition as a numeric code. Follow WMO weather interpretation codes. See table below for details. (WMO code)',
        'pressure_msl': 'Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface. Typically pressure on mean sea level is used in meteorology. Surface pressure gets lower with increasing elevation. (hPa)',
        'surface_pressure': 'Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface. Typically pressure on mean sea level is used in meteorology. Surface pressure gets lower with increasing elevation. (hPa)',
        'cloud_cover': 'Total cloud cover as an area fraction (%)',
        'cloud_cover_low': 'Low level clouds and fog up to 3 km altitude (%)',
        'cloud_cover_mid': 'Mid level clouds from 3 to 8 km altitude (%)',
        'cloud_cover_high': 'High level clouds from 8 km altitude (%)',
        'et0_fao_evapotranspiration': 'ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants. (mm)',
        'vapour_pressure_deficit': 'Vapour Pressure Deficit (VPD) in kilopascal (kPa). For high VPD (>1.6), water transpiration of plants increases. For low VPD (<0.4), transpiration decreases (kPa)',
        'wind_speed_10m': 'Wind speed at 10 meters above ground. Wind speed on 10 meters is the standard level. (km/h)',
        'wind_speed_100m': 'Wind speed at 100 meters above ground. Wind speed on 10 meters is the standard level. (km/h)',
        'wind_direction_10m': 'Wind direction at 10 meters above ground (degrees)',
        'wind_direction_100m': 'Wind direction at 100 meters above ground (degrees)',
        'wind_gusts_10m': 'Gusts at 10 meters above ground as a maximum of the preceding hour (km/h)',
        'soil_temperature_0_to_7cm': 'Average temperature of different soil levels below ground.',
        'soil_temperature_7_to_28cm': 'Average temperature of different soil levels below ground.',
        'soil_temperature_28_to_100cm': 'Average temperature of different soil levels below ground.',
        'soil_temperature_100_to_255cm': 'Average temperature of different soil levels below ground.',
        'soil_moisture_0_to_7cm': 'Average soil water content as volumetric mixing ratio at 0-7 cm depth. (m^3/m^3)',
        'soil_moisture_7_to_28cm': 'Average soil water content as volumetric mixing ratio at 7-28 cm depth. (m^3/m^3)',
        'soil_moisture_28_to_100cm': 'Average soil water content as volumetric mixing ratio at 28-100 cm depth. (m^3/m^3)',
        'soil_moisture_100_to_255cm': 'Average soil water content as volumetric mixing ratio at 100-255 cm depth. (m^3/m^3)'
    }

def set_forecast_feature_description(data):
    data.attrs['description'] = {
        'date': 'Time and date',
        'latitude': 'Geographical WGS84 coordinates of the location. Multiple coordinates can be comma separated. E.g. &latitude=52.52,48.85&longitude=13.41,2.35. To return data for multiple locations the JSON output changes to a list of structures. CSV and XLSX formats add a column location_id. For North and South America locations use negative longitudes, because they lie west of Greenwich.',
        'longitude': 'Geographical WGS84 coordinates of the location. Multiple coordinates can be comma separated. E.g. &latitude=52.52,48.85&longitude=13.41,2.35. To return data for multiple locations the JSON output changes to a list of structures. CSV and XLSX formats add a column location_id. For North and South America locations use negative longitudes, because they lie west of Greenwich.',
        'elevation': 'The elevation used for statistical downscaling. Per default, a 90 meter digital elevation model is used. You can manually set the elevation to correctly match mountain peaks. If &elevation=nan is specified, downscaling will be disabled and the API uses the average grid-cell height. For multiple locations, elevation can also be comma separated.',
        'temperature_2m': 'Air temperature at 2 meters above ground (Celsius degrees)',
        'relative_humidity_2m': 'Relative humidity at 2 meters above ground',
        'dew_point_2m': 'Dew point temperature at 2 meters above ground (Celsius degrees)',
        'apparent_temperature': 'Apparent temperature is the perceived feels-like temperature combining wind chill factor, relative humidity and solar radiation (Celsius degrees)',
        'precipitation_probability': 'Probability of precipitation with more than 0.1 mm of the preceding hour. Probability is based on ensemble weather models with 0.25° (~27 km) resolution. 30 different simulations are computed to better represent future weather conditions.',
        'precipitation': 'Total precipitation (rain, showers, snow) sum of the preceding hour (mm)',
        'rain': 'Rain from large scale weather systems of the preceding hour in millimeter (mm)',
        'showers': 'Showers from convective precipitation in millimeters from the preceding hour (mm)',
        'snowfall': 'Snowfall amount of the preceding hour in centimeters. For the water equivalent in millimeter, divide by 7. E.g. 7 cm snow = 10 mm precipitation water equivalent (cm)',
        'snow_depth': 'Snow depth on the ground (meters)',
        'weather_code': 'Weather condition as a numeric code. Follow WMO weather interpretation codes. See table below for details. (WMO code)',
        'pressure_msl': 'Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface. Typically pressure on mean sea level is used in meteorology. Surface pressure gets lower with increasing elevation. (hPa)',
        'surface_pressure': 'Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface. Typically pressure on mean sea level is used in meteorology. Surface pressure gets lower with increasing elevation. (hPa)',
        'cloud_cover': 'Total cloud cover as an area fraction (%)',
        'cloud_cover_low': 'Low level clouds and fog up to 3 km altitude (%)',
        'cloud_cover_mid': 'Mid level clouds from 3 to 8 km altitude (%)',
        'cloud_cover_high': 'High level clouds from 8 km altitude (%)',
        'visibility': 'Viewing distance in meters. Influenced by low clouds, humidity and aerosols. (meters)',
        'evapotranspiration': 'Evapotranspration from land surface and plants that weather models assumes for this location. Available soil water is considered. 1 mm evapotranspiration per hour equals 1 liter of water per spare meter. (mm)',
        'et0_fao_evapotranspiration': 'ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants. (mm)',
        'vapour_pressure_deficit': 'Vapour Pressure Deficit (VPD) in kilopascal (kPa). For high VPD (>1.6), water transpiration of plants increases. For low VPD (<0.4), transpiration decreases (kPa)',
        'wind_speed_10m': 'Wind speed at 10 meters above ground. Wind speed on 10 meters is the standard level. (km/h)',
        'wind_speed_80m': 'Wind speed at 80 meters above ground. Wind speed on 10 meters is the standard level. (km/h)',
        'wind_speed_120m': 'Wind speed at 120 meters above ground. Wind speed on 10 meters is the standard level. (km/h)',
        'wind_speed_180m': 'Wind speed at 180 meters above ground. Wind speed on 10 meters is the standard level. (km/h)',
        'wind_direction_10m': 'Wind direction at 10 meters above ground (degrees)',
        'wind_direction_80m': 'Wind direction at 80 meters above ground (degrees)',
        'wind_direction_120m': 'Wind direction at 120 meters above ground (degrees)',
        'wind_direction_180m': 'Wind direction at 180 meters above ground (degrees)',
        'wind_gusts_10m': 'Gusts at 10 meters above ground as a maximum of the preceding hour (km/h)',
        'temperature_80m': 'Air temperature at 80 meters above ground (Celsius degrees)',
        'temperature_120m': 'Air temperature at 120 meters above ground (Celsius degrees)',
        'temperature_180m': 'Air temperature at 180 meters above ground (Celsius degrees)',
        'soil_temperature_0cm': 'Temperature in the soil at 0 cm depth. 0 cm is the surface temperature on land or water surface temperature on water. (Celsius degrees)',
        'soil_temperature_6cm': 'Temperature in the soil at 6 cm depth. 0 cm is the surface temperature on land or water surface temperature on water. (Celsius degrees)',
        'soil_temperature_18cm': 'Temperature in the soil at 18 cm depth. 0 cm is the surface temperature on land or water surface temperature on water. (Celsius degrees)',
        'soil_temperature_54cm': 'Temperature in the soil at 54 cm depth. 0 cm is the surface temperature on land or water surface temperature on water. (Celsius degrees)',
        'soil_moisture_0_to_1cm': 'Average soil water content as volumetric mixing ratio at 0-1 cm depth. (m^3/m^3)',
        'soil_moisture_1_to_3cm': 'Average soil water content as volumetric mixing ratio at 1-3 cm depth. (m^3/m^3)',
        'soil_moisture_3_to_9cm': 'Average soil water content as volumetric mixing ratio at 3-9 cm depth. (m^3/m^3)',
        'soil_moisture_9_to_27cm': 'Average soil water content as volumetric mixing ratio at 9-27 cm depth. (m^3/m^3)',
        'soil_moisture_27_to_81cm': 'Average soil water content as volumetric mixing ratio at 27-81 cm depth. (m^3/m^3)'
}
    
def convert_wide_data(data, features):
    """
    Unstack the data as we have several rows corresponding to the same DATE

    Args:
        data: data with repeated dates
        features: values we want in the future dataset
    """
    wide_data = data.pivot_table(
        index='date',
        columns='zone',
        values=[x for x in features if x not in ['date', 'latitude', 'longitude']],
        aggfunc='first'
        )

    wide_data.columns = [f'{feat}_{zone}' for feat, zone in wide_data.columns]

    return wide_data

def write_zones_to_data(data, locations_in_order):
    """
    Assign zones to the latitudes and longitudes (we expect to have unique latitudes in the dataset)

    Args:
        data: data with unique latitude values
        locations_in_order: locations corresponding to the latitudes written in order of appearance

    Return: add a column 'zone' with the name of the corresponding location
    """
    loc_dict = {lat: loc for (lat, loc) in zip(data['latitude'].unique(), locations_in_order)}

    data['zone'] = data['latitude'].map(loc_dict)

def add_sin_cos_hour_data(data):
    """
    This function add new features to reproduce the periodic values of the data
    """
    data_date = pd.to_datetime(data.index)
    data['hour'] = data_date.hour
    data['sin_hour'] = np.sin(2*np.pi*data['hour']/24)
    data['cos_hour'] = np.cos(2*np.pi*data['hour']/24)

def add_lags_to_data(data, target):
    # Build the lags
    lags = [1, 3, 6, 12, 24]
    for lag in lags:
        data[f'temp_lag_{lag}h'] = data[target].shift(lag)

    # Rolling means
    data['temp_rolling_mean_6h']  = data[target].shift(1).rolling(6).mean()
    data['temp_rolling_mean_24h'] = data[target].shift(1).rolling(24).mean()

def main():
    save_data()

if __name__ == "__main__":
    main()