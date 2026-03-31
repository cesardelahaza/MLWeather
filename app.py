import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="MLWeather", layout="wide")

# --- Sidebar (lateral menu) ---
with st.sidebar:
    st.title("Menu")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "Visualizations"]
    )
    
    st.markdown("---")
    st.caption("My first Streamlit app")

# --- Principal Content ---
st.title("MLWeather")

if page == "Home":
    st.header("Welcome 👋")
    st.write("Add map of Seville...")

elif page == "Visualizations":
    st.header("Visualization 📊")

    # Load CSV
    df = pd.read_csv("data/historic_data.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    
    locations = ['seville', 'cadiz', 'huelva', 'malaga', 'cordoba', 'badajoz', 'lisbon', 'c_real']

    loc_dict = {lat: loc for (lat, loc) in zip(df['latitude'].unique(), locations)}

    df['zone'] = df['latitude'].map(loc_dict)

    wide_df = df.pivot_table(
        index='date',
        columns='zone',
        values= [x for x in df.columns if x not in ['date', 'latitude']],
        aggfunc='first'
    )

    wide_df.columns = [f'{feat}_{zone}' for feat, zone in wide_df.columns]

    st.subheader("Historic Data")
    
    # --- Variable selector (Y axis) ---
    vars = {'Temperature': 'temperature_2m', 
            'Pressure': 'pressure_msl', 
            'Humidity': 'relative_humidity_2m',
            'Precipitation': 'precipitation'} 
    
    var = st.selectbox("Variable", list(vars.keys()))

    fig1 = px.line(wide_df, y=[x for x in wide_df.columns if vars[var] in x], 
                   title=var,
                   labels={
                       'date': 'Date',
                       'value': var
                       })
    st.plotly_chart(fig1)  
    timestamps = wide_df.index

    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude",
                            color=vars[var], range_color=[df[vars[var]].min(), df[vars[var]].max()],
                            animation_frame="date", 
                            hover_name="zone",
                            size=vars[var],
                            color_continuous_scale="RdYlBu_r",
                            zoom=5, height=500
                            )
    fig.update_layout(mapbox_style="open-street-map")

    st.plotly_chart(fig, use_container_width=True)

    # Show dataframe   
    st.dataframe(wide_df)

