import streamlit as st
import plotly.express as px
import pandas as pd

def render_geographic_analysis(df):
    """Render the geographic analysis section"""
    st.markdown("## Geographic Analysis")
    
    # Global Distribution
    st.markdown("### Global Distribution")
    try:
        st.plotly_chart(plot_bubble_map(df), use_container_width=True)
        render_country_statistics(df)
        
    except Exception as e:
        st.error(f"Error creating geographic visualization: {str(e)}")
        st.info("Some geographic data might be missing or incorrect.")

def plot_bubble_map(df):
    """Create a bubble map showing global distribution of cars"""
    map_df = df.groupby('Country').agg({
        'Model': 'count',
        'Price': 'mean'
    }).reset_index()
    map_df.columns = ['Country', 'Cars_Count', 'Avg_Price']
    
    country_coords = {
        'Germany': [51.1657, 10.4515],
        'Italy': [41.8719, 12.5674],
        'United Kingdom': [55.3781, -3.4360],
        'United States': [37.0902, -95.7129],
        'Japan': [36.2048, 138.2529],
        'France': [46.2276, 2.2137],
        'Sweden': [60.1282, 18.6435],
        'South Korea': [35.9078, 127.7669],
        'China': [35.8617, 104.1954]
    }
    
    map_df['Latitude'] = map_df['Country'].map(lambda x: country_coords.get(x, [0, 0])[0])
    map_df['Longitude'] = map_df['Country'].map(lambda x: country_coords.get(x, [0, 0])[1])
    
    fig = px.scatter_mapbox(
        map_df,
        lat='Latitude',
        lon='Longitude',
        size='Cars_Count',
        color='Avg_Price',
        hover_name='Country',
        hover_data={
            'Cars_Count': True,
            'Avg_Price': ':,.0f',
            'Latitude': False,
            'Longitude': False
        },
        color_continuous_scale='Viridis',
        size_max=40,
        zoom=1.5,
        title='Global Distribution of Sports Cars'
    )
    
    fig.update_layout(
        mapbox_style='carto-positron',
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    return fig

def render_country_statistics(df):
    """Render country statistics section"""
    st.markdown("### Country Statistics")
    country_stats = df.groupby('Country').agg({
        'Model': 'count',
        'Price': ['mean', 'min', 'max'],
        'Horsepower': 'mean'
    }).round(2)
    
    country_stats.columns = ['Number of Models', 'Average Price', 'Min Price', 'Max Price', 'Average Horsepower']
    country_stats = country_stats.sort_values('Number of Models', ascending=False)
    
    for col in ['Average Price', 'Min Price', 'Max Price']:
        country_stats[col] = country_stats[col].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(country_stats, use_container_width=True)
