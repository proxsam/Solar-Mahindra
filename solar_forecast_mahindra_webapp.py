import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.graph_objects as go
import requests
import pytz
from pvlib import solarposition
import pvlib
from sklearn.preprocessing import  OrdinalEncoder, StandardScaler

# Load the model and scalers
@st.cache_resource
def load_model():
    model = joblib.load('xgb.pkl')
    return model


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_future_weather_forecast(latitude, longitude, start_date, num_forecast_steps):
    # Predictors to fetch
    Predictors = [
        'temperature_2m', 'relativehumidity_2m', 'dew_point_2m',
        'surface_pressure', 'cloud_cover', 'wind_speed_10m',
        'wind_direction_10m', 'shortwave_radiation',
        'direct_radiation', 'diffuse_radiation'
    ]

    # Calculate end date based on start date and forecast steps
    end_date = pd.to_datetime(start_date) + pd.Timedelta(hours=num_forecast_steps-1)

    # Fetch forecast
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date.strftime('%Y-%m-%d'),
        'hourly': Predictors
    }).json()

    # Create DataFrame
    time = pd.to_datetime(np.array(r['hourly']['time']))
    weather_forecast_df = pd.DataFrame(index=time)

    for p in Predictors:
        weather_forecast_df[p] = np.array(r['hourly'][p])

    # weather_forecast_df = weather_forecast_df.between_time('06:00', '23:00')

    return weather_forecast_df

def nonlinear_features(df,latitude,longitude,altitude):
    
    # Ensure the dataframe has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.set_index('datetime', inplace=True)

    # 2. Temperature-Humidity Index (THI)
    df['temperature_f'] = df['temperature_2m'] * 9/5 + 32  # Convert to Fahrenheit
    df['THI'] = df['temperature_f'] - (0.55 - 0.0055 * df['relativehumidity_2m']) * (df['temperature_f'] - 58)

    # 3. Wind Chill Factor
    df['wind_speed_mph'] = df['wind_speed_10m'] * 2.237  # Convert to mph
    df['wind_chill'] = 35.74 + 0.6215*df['temperature_f'] - 35.75*(df['wind_speed_mph']**0.16) + 0.4275*df['temperature_f']*(df['wind_speed_mph']**0.16)

    # 4. Heat Index
    df['heat_index'] = -42.379 + 2.04901523*df['temperature_f'] + 10.14333127*df['relativehumidity_2m'] - 0.22475541*df['temperature_f']*df['relativehumidity_2m'] - 6.83783e-3*df['temperature_f']**2 - 5.481717e-2*df['relativehumidity_2m']**2 + 1.22874e-3*df['temperature_f']**2*df['relativehumidity_2m'] + 8.5282e-4*df['temperature_f']*df['relativehumidity_2m']**2 - 1.99e-6*df['temperature_f']**2*df['relativehumidity_2m']**2

    # 5. Solar Zenith Angle
    df['solar_zenith_angle'] = calculate_solar_zenith_angle(df, latitude, longitude, altitude)

    # 6. Air Mass
    df['air_mass'] = 1 / np.cos(np.radians(df['solar_zenith_angle']))

    # 9. Day of Year Sine and Cosine
    df['day_of_year'] = df.index.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # 10. Hour of Day Sine and Cosine
    df['hour_of_day'] = df.index.hour + df.index.minute / 60
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # 11. Lagged Variables
    for col in ['Active_Power']:
        df[f'{col}_lag_1h'] = df[col].shift(1)
        df[f'{col}_lag_2h'] = df[col].shift(2)
        df[f'{col}_lag_3h'] = df[col].shift(3)
        df[f'{col}_lag_4h'] = df[col].shift(4)
        df[f'{col}_lag_21h'] = df[col].shift(21)
        df[f'{col}_lag_22h'] = df[col].shift(22)
        df[f'{col}_lag_23h'] = df[col].shift(23)
        df[f'{col}_lag_24h'] = df[col].shift(24)

    # 11. Lagged Variables
    for col in ['temperature_2m', 'cloud_cover', 'wind_speed_10m']:
        df[f'{col}_lag_1h'] = df[col].shift(1)
        df[f'{col}_lag_24h'] = df[col].shift(24)

    # 12. Rolling Statistics
    for col in ['temperature_2m', 'cloud_cover', 'wind_speed_10m']:
        df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
        df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24).std()

    # 13. Interaction Terms
    df['temp_wind_interaction'] = df['temperature_2m'] * df['wind_speed_10m']
    df['cloud_radiation_interaction'] = df['cloud_cover'] * df['direct_radiation']

    # 15. Weather Stability Index
    # Calculate the change in key weather variables
    for col in ['temperature_2m', 'cloud_cover', 'wind_speed_10m']:
        df[f'{col}_change'] = df[col].diff()

    # Create a composite weather stability index
    df['weather_stability_index'] = (df['temperature_2m_change'].abs() +
                                     df['cloud_cover_change'].abs() +
                                     df['wind_speed_10m_change'].abs())

    return df

def calculate_solar_zenith_angle(df, latitude, longitude, altitude):
  site = pvlib.location.Location(latitude, longitude, altitude=altitude)

  solar_position = site.get_solarposition(df.index)
  solar_zenith_angle = solar_position['zenith']

  return solar_zenith_angle


def standardize_future_data(new_test,predictor_scaler_fit,encoder):

    standardize_predictor_list = ['temperature_2m', 'relativehumidity_2m', 'dew_point_2m',
       'surface_pressure', 'cloud_cover', 'wind_speed_10m',
       'wind_direction_10m', 'shortwave_radiation', 'direct_radiation',
       'diffuse_radiation', 'season', 'temperature_f', 'THI',
       'wind_speed_mph', 'wind_chill', 'heat_index', 'solar_zenith_angle',
       'air_mass', 'day_of_year', 'day_of_year_sin', 'day_of_year_cos',
       'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos',
       'Active_Power_lag_1h','Active_Power_lag_2h','Active_Power_lag_3h','Active_Power_lag_4h','Active_Power_lag_21h','Active_Power_lag_22h','Active_Power_lag_23h','Active_Power_lag_24h',
       'temperature_2m_lag_1h','temperature_2m_lag_24h', 'cloud_cover_lag_1h', 'cloud_cover_lag_24h',
       'wind_speed_10m_lag_1h', 'wind_speed_10m_lag_24h',
       'temperature_2m_rolling_mean_24h', 'temperature_2m_rolling_std_24h',
       'cloud_cover_rolling_mean_24h', 'cloud_cover_rolling_std_24h',
       'wind_speed_10m_rolling_mean_24h', 'wind_speed_10m_rolling_std_24h',
       'temp_wind_interaction', 'cloud_radiation_interaction',
       'temperature_2m_change', 'cloud_cover_change', 'wind_speed_10m_change',
       'weather_stability_index']
    
    X_new_test = new_test[standardize_predictor_list]
    X_new_test = predictor_scaler_fit.transform(X_new_test)
    new_stand_test = pd.DataFrame(X_new_test, index=new_test[standardize_predictor_list].index, columns=new_test[standardize_predictor_list].columns)

    categorical_columns = ['time_interval']
    encoded_features_test = encoder.transform(new_test[categorical_columns])
    encoded_test = pd.DataFrame(encoded_features_test, columns=categorical_columns, index=new_test.index)

    new_stand_test = pd.concat([new_stand_test, encoded_test], axis = 1)

    return new_stand_test

def detect_time_interval_future(df):
    df_time_detect = df.copy()

    df_time_detect.index = pd.to_datetime(df_time_detect.index)

    intervals = {'zeroth_interval':(0,6), 'first_interval': (6, 9), 'second_interval': (9, 11), 'third_interval': (11, 13),
                'fourth_interval': (13, 15), 'fifth_interval': (15, 17), 'sixth_interval': (17, 20)}

    df_time_detect['time_interval'] = pd.cut(df_time_detect.index.hour, bins=[interval[0] for interval in intervals.values()] + [24],
                                labels=[interval_name for interval_name in intervals.keys()],
                                include_lowest=True, right=False)

    return df_time_detect

def future_forecast(model, num_forecast_steps, features_list, predictor_scaler_fit, encoded_features,
                    latitude, longitude,altitude, weather_forecast , initial_data_no_linear):

    forecast_data_no_linear = initial_data_no_linear.copy()
    weather_forecast_df = weather_forecast.copy()

    future_predictions = []

    for i in range(num_forecast_steps):

        future_data = pd.concat([forecast_data_no_linear,weather_forecast_df])

        forecast_data = nonlinear_features(future_data, latitude, longitude, altitude)

        forecast_data = forecast_data.between_time('06:00', '23:00')

        forecast_data.dropna(inplace=True)

        forecast_data_standard = standardize_future_data(forecast_data,predictor_scaler_fit,encoded_features)

        current_features = forecast_data_standard[features_list].iloc[i:i+1]

        current_hour = forecast_data.index.hour[i]

        if current_hour >= 18 or current_hour <= 6:
            next_prediction = 0.0
        else:
          next_prediction = model.predict(xgb.DMatrix(current_features))[0]
          next_prediction = next_prediction.clip(min=0.0)

        future_predictions.append(next_prediction)

        weather_forecast_df['Active_Power'].iloc[i+6] = next_prediction

    return future_predictions

def add_season(df):
  def season(month):
    if month in [12,1,2]:
      return 'Summer'
    elif month in [3,4,5]:
      return 'Autumn'
    elif month in [6,7,8]:
      return 'Winter'
    else:
      return 'Spring'

  df['season'] = df['date'].dt.month.apply(season)
  return df


def main():

    st.title('Solar Power Generation Forecasting')
    st.write('Using Open-Meteo API for weather data')
    
    # Yulara coordinates (default)
    lat = 28.009465
    lon = 72.980845
    altitude = 217

    features_importance_mahindra = ['hour_of_day_cos',
        'Active_Power_lag_1h',
        'Active_Power_lag_2h',
        'Active_Power_lag_24h',
        'cloud_cover_rolling_mean_24h',
        'hour_of_day',
        'Active_Power_lag_21h',
        'Active_Power_lag_23h',
        'cloud_cover_lag_1h',
        'temperature_2m_lag_1h',
        'temperature_2m_rolling_std_24h',
        'Active_Power_lag_22h',
        'air_mass',
        'hour_of_day_sin',
        'Active_Power_lag_4h',
        'temperature_2m_rolling_mean_24h',
        'Active_Power_lag_3h',
        'cloud_cover_lag_24h',
        'temperature_2m_lag_24h',
        'relativehumidity_2m',
        'cloud_cover',
        'temperature_f',
        'time_interval',
        'shortwave_radiation',
        'heat_index',
        'temperature_2m_change',
        'wind_speed_10m_rolling_std_24h',
        'temp_wind_interaction',
        'wind_speed_10m',
        'solar_zenith_angle',
        'day_of_year_cos',
        'weather_stability_index',
        'wind_chill',
        'dew_point_2m',
        'cloud_cover_change',
        'wind_speed_10m_rolling_mean_24h',
        'THI',
        'surface_pressure',
        'day_of_year',
        'wind_speed_10m_change',
        'cloud_cover_rolling_std_24h',
        'wind_speed_10m_lag_24h',
        'cloud_radiation_interaction',
        'wind_speed_10m_lag_1h',
        'wind_speed_mph',
        'day_of_year_sin',
        'diffuse_radiation',
        'wind_direction_10m',
        'temperature_2m',
        'direct_radiation',
        'season']
    
    # Main content area
    st.subheader(f"Forecast for Next 1 day")
    
    # Display selected parameters
    st.write(f"""
    **Selected Parameters:**
    - Location: {lat:.4f}°N, {lon:.4f}°E
    - Date: "11-28-2024" (Next day w.r.t Training Data)
    """)

    # Fetch weather data button
    if st.button('Generate Forecast'):
        with st.spinner('Fetching weather data and generating forecast...'):

            train = pd.read_csv("train_11_27_2024.csv")

            actual_power_28_11 = pd.read_csv('Real_Power_28_11_24.csv', skiprows=5, nrows=18)

            actual_power_28_11.rename(columns={"0.0": "Active_Power"}, inplace=True)

            initial_forecast_data_no_linear = train.tail(24)

            initial_forecast_data_no_linear['date'] = pd.to_datetime(initial_forecast_data_no_linear['date'])

            initial_forecast_data_no_linear = initial_forecast_data_no_linear.set_index('date')
            
            print(initial_forecast_data_no_linear)

            start_date = (initial_forecast_data_no_linear.index[-1] + pd.Timedelta(hours=1)).strftime('%Y-%m-%d')

            weather_forecast = fetch_future_weather_forecast(
                latitude=28.009465,
                longitude=72.980845,
                start_date=start_date,
                num_forecast_steps=24
            )
            
            model = joblib.load("xgb.pkl")
            season_ord = joblib.load("season.pkl")
            predictor_scaler_fit = joblib.load("predictor_scaler_fit.pkl")
            encoded_features = joblib.load("encoded_features.pkl")

            features_list = features_importance_mahindra[:25]

            weather_forecast = weather_forecast.reset_index()  
            weather_forecast.rename(columns={"index": "date"}, inplace=True)
        
            # Process weather data
            weather_forecast['date'] = pd.to_datetime(weather_forecast['date'])
            weather_forecast = add_season(weather_forecast)
            weather_forecast = weather_forecast.set_index('date')
            season_train = season_ord.transform(np.array(weather_forecast['season']).reshape(-1,1))
            weather_forecast['season'] = season_train
            weather_forecast = detect_time_interval_future(weather_forecast)
            weather_forecast['Active_Power']=0.0

            # Forecast next 18 hours
            future_predictions = future_forecast(
                model=model,
                num_forecast_steps=18,
                features_list=features_list,
                latitude=28.009465,
                longitude=72.980845,
                altitude = 217,
                predictor_scaler_fit=predictor_scaler_fit, 
                encoded_features=encoded_features,
                weather_forecast=weather_forecast,
                initial_data_no_linear = initial_forecast_data_no_linear
            )

            future_dates = pd.date_range(
                start=initial_forecast_data_no_linear.index[-1] + pd.Timedelta(hours=7),
                periods=18,
                freq='H'
            )

            df_daylight = pd.DataFrame({
                    'date': future_dates,
                    'predicted_power': future_predictions
                })
            df_daylight['Active_Power'] = actual_power_28_11['Active_Power']

            # Create tabs for different views
            tab1, tab2 = st.tabs(["Forecast Plot", "Plots"])
            
            with tab1:
                # Create plot
                fig = go.Figure()

                # Add power prediction line
                fig.add_trace(go.Scatter(
                    x=df_daylight['date'],
                    y=df_daylight['predicted_power'],
                    name='Predicted Power',
                    line=dict(color='green')  
                ))

                fig.add_trace(go.Scatter(
                    x=df_daylight['date'],
                    y=df_daylight['Active_Power'],
                    name='Actual Power',
                    line=dict(color='red')  
                ))

                # Update layout
                fig.update_layout(
                    title='Solar Power Predictions for Future Dates',
                    xaxis_title='Date',
                    yaxis_title='Predicted Active Power',
                    height=600
                )

                # Display in Streamlit
                st.plotly_chart(fig)

                actual_values = df_daylight['Active_Power']
                predicted_values = df_daylight['predicted_power']

                point_wise_rmse = np.sqrt((actual_values - predicted_values)**2)
                df_daylight['Hour_Wise_RMSE'] = point_wise_rmse
            
                st.dataframe(df_daylight)
                fig2 = go.Figure()

                # Add point-wise RMSE line
                fig2.add_trace(go.Scatter(
                    x=df_daylight['date'],
                    y=point_wise_rmse,
                    name='Point-wise RMSE',
                    line=dict(color='blue')  
                ))

                # Update layout
                fig2.update_layout(
                    title='Hour-wise Root Mean Square Error',
                    xaxis_title='Date',
                    yaxis_title='RMSE',
                    height=500
                )
                
                # Display in Streamlit
                st.plotly_chart(fig2)

            with tab2:
                # Display 3 images in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image('val_data_preds.png', caption='Validation Data Predictions', use_container_width=True)
                    st.image('how_early_train.png',caption='How earlier dates we use to Train Vs RMSE ', use_container_width=True)
                    st.image('top_25_correlation.png',caption='Top 25 Features Correlation with Active Power ', use_container_width=True)
                with col2:
                    st.image('mahindra_flow_chart.png', caption='Predictons Flowchart', use_container_width=True)

if __name__ == '__main__':
    main()
