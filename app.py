import streamlit as st
import pandas as pd
import plotly.express as px

import folium
from streamlit_folium import st_folium

import json

import ui.charts as charts

# Page configuration
st.set_page_config(page_title="MLWeather", layout="wide")

@st.cache_data
def load_data():
    historic = pd.read_csv("data/treated_historic_data.csv")
    forecast = pd.read_csv("data/treated_forecast_data.csv")
    historic['d_type'] = 'Historic'
    forecast['d_type'] = 'Forecast'
    return pd.concat([historic, forecast])

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
    st.subheader('ML Project to predict temperatures in Seville')
    m = folium.Map(location=[37.39, -5.99], zoom_start=12)  
    folium.Marker([37.39, -5.99], popup="Seville", tooltip='Seville').add_to(m)
    st_folium(m, use_container_width=True, height=400)

if page == "Visualizations":
    st.header("The Weather in Seville")

    # Load CSV
    df = pd.read_csv("data/treated_historic_data.csv")
    
    tab_sev, tab_all = st.tabs(["Seville", "All the locations"])
    
    df_combined = load_data()
    
    with tab_sev:
        # Variables in Seville
        col1_temp_sev, col2_boxs_sev = st.columns([1,1])
        with col1_temp_sev:
            st.plotly_chart(charts.line_chart(df_combined, 'temperature_2m_seville', 
                                            'Temperature in Seville', 'Temperature (ºC)'))
        with col2_boxs_sev:
            
            f = px.box(df_combined, x='hour', y='temperature_2m_seville',
                   title='Temperature boxplots for each hour', color='d_type',
                   color_discrete_map={"Historic": "blue", "Forecast": "red"},
                   labels={'hour': 'Hour',
                           'temperature_2m_seville': 'Temperature (ºC)',
                           'd_type': ''})
            
            st.plotly_chart(f)
        
        col1_prec_sev, col2_prec_sev = st.columns([1,1])
        with col1_prec_sev:
            st.plotly_chart(charts.line_chart(df_combined, 'precipitation_seville', 
                                              'Precipitation in Seville', 'Precipitation'))
        with col2_prec_sev:
            fig_viol_sev = px.violin(df_combined, y='cloud_cover_seville', points='all', box=True,
                                     title='Cloud cover in Seville', color='d_type',
                                     color_discrete_map={"Historic": "blue", "Forecast": "red"},
                                     labels={'date': 'Date',
                                             'cloud_cover_seville': 'Cloud cover (%)',
                                             'd_type': ''})
            st.plotly_chart(fig_viol_sev)
        
        col1_pres_sev, col2_humi_sev, col3_wind_sev = st.columns(3)
        with col1_pres_sev:
            st.plotly_chart(charts.line_chart(df_combined, 'pressure_msl_seville', 
                                              'Pressure in Seville', 'Pressure'))
        
        with col2_humi_sev:
            st.plotly_chart(charts.line_chart(df_combined, 'relative_humidity_2m_seville', 
                                              'Relative humidity in Seville', 'Humidity'))
        
        with col3_wind_sev:
            fig_wind_sev = px.scatter_polar(df_combined, 
                                            r="wind_speed_10m_seville", theta="wind_direction_10m_seville", 
                                            color="d_type", opacity=0.5,
                                            title='Wind speed and direction',
                                            color_discrete_map={"Historic": "blue", "Forecast": "red"},
                                            labels={'d_type': ''})
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
              'XGBoost model': 'xgboost_model',
              'Random Forest model': 'random_forest_model',
              'SARIMA model': 'sarima_model',
              'LSTM model': 'lstm_model'} 
    
    model = st.selectbox("Choose model from the list...", list(models.keys()))
    
    metrics = json.loads(open('evaluation/metrics.json').read())

    labels = {'mae': 'MAE', 'rmse': 'RMSE', 'r2': 'R²'}

    model_metrics = metrics[models[model]]
    cols = st.columns(len(labels))
    for col, key in zip(cols, model_metrics):
        with col:
            st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 8px'>
                <div style='font-size: 0.85rem; color: gray'>{labels[key]}</div>
                <div style='font-size: 1.8rem; font-weight: bold'>{model_metrics[key]}</div>
                </div>
            """, unsafe_allow_html=True)
        
    model_data = pd.read_csv('data/predictions/' + models[model] + '_predictions.csv', index_col=0)

    # Real and predicted in same plot
    st.plotly_chart(charts.model_predictions(model_data, model))

    # Most important features of the model
    if model not in ['SARIMA model', 'LSTM model']:
        coefs = pd.Series(model_metrics['importance'], index=model_metrics['features'])
        top_coefs = coefs.abs().nlargest(15).sort_values()

        st.plotly_chart(charts.most_imp_feats(top_coefs, model), use_container_width=True)

    # Scatterplot predicted vs real | Residuals
    col1_vs, col2_res = st.columns(2)
    with col1_vs:

        st.plotly_chart(charts.pred_vs_real(model_data, model), use_container_width=True)

    with col2_res:

        st.plotly_chart(charts.residuals(model_data,model), use_container_width=True)
    
    # Show dataframe of predictions
    st.markdown('---')
    st.subheader(f'Data obtained from the {model}')
    st.dataframe(model_data)