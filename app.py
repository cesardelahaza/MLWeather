import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import json

# Page configuration
st.set_page_config(page_title="MLWeather", layout="wide")

# --- Sidebar (lateral menu) ---
with st.sidebar:
    st.title("Menu")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "Visualizations", "Predictions"]
    )
    
    st.markdown("---")
    st.caption("My first Streamlit app")

# --- Principal Content ---
st.title("MLWeather")

if page == "Home":
    st.header("Welcome 👋")
    st.write("Add map of Seville...")

elif page == "Visualizations":
    st.header("Visualization of Data")

    # Load CSV
    df = pd.read_csv("data/treated_historic_data.csv")
    
    tab_sev, tab_all = st.tabs(["Seville", "All the locations"])
    
    historic_df = pd.read_csv("data/treated_historic_data.csv")
    forecast_df = pd.read_csv("data/treated_forecast_data.csv")

    historic_df['d_type'] = 'Historic'
    forecast_df['d_type'] = 'Forecast'
    df_combined = pd.concat([historic_df, forecast_df])
    
    with tab_sev:
        # Variables in Seville
        fig_temp_sev = px.line(df_combined, x='date', y='temperature_2m_seville',
                               title= 'Temperature in Seville', color='d_type',
                               color_discrete_map={"Historic": "blue", "Forecast": "red"},
                               labels={
                                   'date': 'Date',
                                   'temperature_2m_seville': 'Temperature (ºC)',
                                   'd_type': ''
                                   })
        st.plotly_chart(fig_temp_sev)
        
        col1_prec_sev, col2_prec_sev = st.columns([1,1])
        with col1_prec_sev:
            fig_prec_sev = px.line(df_combined, x='date', y='precipitation_seville',
                                   title='Precipitation in Seville', color='d_type',
                                   color_discrete_map={"Historic": "blue", "Forecast": "red"},
                                   labels={
                                       'date': 'Date',
                                       'precipitation_seville': 'Precipitation',
                                       'd_type': ''
                                       })
            st.plotly_chart(fig_prec_sev)
        with col2_prec_sev:
            fig_viol_sev = px.violin(df_combined, y='cloud_cover_seville', points='all', box=True,
                                     title='Cloud cover in Seville', color='d_type',
                                     color_discrete_map={"Historic": "blue", "Forecast": "red"},
                                     labels={
                                         'date': 'Date',
                                         'cloud_cover_seville': 'Cloud cover (%)',
                                         'd_type': ''
                                   })
            st.plotly_chart(fig_viol_sev)
        
        col1_pres_sev, col2_humi_sev, col3_wind_sev = st.columns(3)
        with col1_pres_sev:
            fig_pres_sev = px.line(df_combined, x='date', y='pressure_msl_seville',
                                   title='Pressure in Sevilla', color='d_type',
                                   color_discrete_map={"Historic": "blue", "Forecast": "red"},
                                   labels={
                                       'date': 'Date',
                                       'pressure_msl_seville': 'Pressure',
                                       'd_type': ''
                                       })
            st.plotly_chart(fig_pres_sev)
        with col2_humi_sev:
            fig_humi_sev = px.line(df_combined, x='date', y='relative_humidity_2m_seville',
                                   title='Relative humidity in Sevilla', color='d_type',
                                   color_discrete_map={"Historic": "blue", "Forecast": "red"},
                                   labels={
                                       'date': 'Date',
                                       'relative_humidity_2m_seville': 'Humidity',
                                       'd_type': ''
                                       })
            st.plotly_chart(fig_humi_sev)
        with col3_wind_sev:
            fig_wind_sev = px.scatter_polar(df_combined, 
                                            r="wind_speed_10m_seville", theta="wind_direction_10m_seville", 
                                            color="d_type", opacity=0.5,
                                            title='Wind speed and direction',
                                            color_discrete_map={"Historic": "blue", "Forecast": "red"},
                                            labels={
                                                'd_type': ''
                                                })
            st.plotly_chart(fig_wind_sev)
        
    with tab_all:
        # Comparison in all regions
        # --- Feature selector (Y axis) ---
        vars = {'Temperature': 'temperature_2m', 
                'Pressure': 'pressure_msl', 
                'Humidity': 'relative_humidity_2m',
                'Precipitation': 'precipitation',
                'Wind': 'wind_speed_10m'} 
        
        var = st.selectbox("Variable", list(vars.keys()))

        # Plot of variables vs time
        fig = px.line(df, y=[x for x in df.columns if vars[var] in x], 
                    title=var,
                    labels={
                        'date': 'Date',
                        'value': var,
                        'variable': 'City'
                        })
        st.plotly_chart(fig) 

        h_df = pd.read_csv("data/historic_data.csv")

        locations = ['seville', 'cadiz', 'huelva', 'malaga', 'cordoba', 'badajoz', 'lisbon', 'c_real']
        loc_dict = {lat: loc for (lat, loc) in zip(h_df['latitude'].unique(), locations)}

        h_df['zone'] = h_df['latitude'].map(loc_dict)

        fig_map = px.scatter_mapbox(h_df, lat="latitude", lon="longitude",
                                color=vars[var], range_color=[h_df[vars[var]].min(), h_df[vars[var]].max()],
                                animation_frame="date", 
                                hover_name="zone",
                                size=vars[var],
                                zoom=5, height=500,
                                title= var + ' in time',
                                labels={
                                    'date': 'Date',
                                    'temperature_2m': 'Temperature',
                                    'pressure_msl': 'Pressure',
                                    'relative_humidity_2m': 'Humidity',
                                    'precipitation': 'Precipitation',
                                    'wind_speed_10m': 'Wind speed'
                                    })
                                
        
        fig_map.update_layout(mapbox_style="carto-positron")
        st.plotly_chart(fig_map, use_container_width=True)
  

elif page == "Predictions":
    st.header("Predictions")

    # --- Model selector ---
    models = {'Linear model': 'linear_model', 
              'XGBoost model': 'xgb_model'} 
    
    model = st.selectbox("Choose model from the list...", list(models.keys()))
    metrics = json.loads(open('evaluation/metrics.json').read())

    if models[model]=='linear_model':
        lr_metrics = metrics[models[model]]
        col1, col2, col3 = st.columns(3)
        with col1:
            mae = lr_metrics['mae']
            st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px'>
                        <div style='font-size: 0.85rem; color: gray'>MAE</div>
                        <div style='font-size: 1.8rem; font-weight: bold'>{mae}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True)
        with col2:
            rmse = lr_metrics['rmse']
            st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px'>
                        <div style='font-size: 0.85rem; color: gray'>RMSE</div>
                        <div style='font-size: 1.8rem; font-weight: bold'>{rmse}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True)
        with col3:
            r2 = lr_metrics['r2']
            st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px'>
                        <div style='font-size: 0.85rem; color: gray'>R2</div>
                        <div style='font-size: 1.8rem; font-weight: bold'>{r2}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True)

        lr_data = pd.read_csv('data/predictions/linear_model_predictions.csv', index_col=0)

        # Real and predicted in same plot
        fig1 = px.line(lr_data, x='date', y=['target', 'y_pred'], 
                   title=model + ' predictions',
                   labels={
                       'date': 'Date',
                       'value': 'Temperature',
                       'variable': 'Legend'
                       })
        fig1.for_each_trace(lambda trace: trace.update(name=trace.name.replace("target", "Real").replace("y_pred", "Predicted")))
        st.plotly_chart(fig1)  
        
        # Most important coefficients of the model
        coefs = pd.Series(lr_metrics['coefficients'], index=lr_metrics['features'])
        top_coefs = coefs.abs().nlargest(15).sort_values()

        fig = px.bar(
            x=top_coefs.values,
            y=top_coefs.index,
            orientation='h',
            title='Top 15 most important coefficients (Linear Regression)',
            labels={'x': 'Absolute coefficient', 'y': 'Feature'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatterplot predicted vs real | Residuals
        col1_vs, col2_res = st.columns(2)
        with col1_vs:

            fig = px.scatter(lr_data, x='target', y='y_pred',
                             title='Predicted vs Real temperature (Linear Regression)',
                             labels={'target': 'Real (°C)', 
                                     'y_pred': 'Predicted (°C)'})

            fig.add_shape(type='line',
                          x0=lr_data['target'].min(),
                          y0=lr_data['target'].min(),
                          x1=lr_data['target'].max(),
                          y1=lr_data['target'].max(),
                          line=dict(color='red', dash='dash'))
            
            st.plotly_chart(fig, use_container_width=True)

        with col2_res:
            fig = px.line(lr_data, x='date', y='residuals',
                          title='Residuals (Linear Regression)',
                          labels={'date': 'Date',
                                  'residuals': 'Residuals'})
            fig.add_hline(y=0, line=dict(color='red', dash='dash'))

            st.plotly_chart(fig, use_container_width=True)

        # Show dataframe of predictions
        st.dataframe(lr_data)
