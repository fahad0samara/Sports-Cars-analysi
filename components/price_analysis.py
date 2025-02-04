import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def render_price_analysis(df):
    """Render the price analysis section"""
    st.markdown("## Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Range Selector
        price_range = st.slider(
            "Select Price Range ($)",
            min_value=float(df['Price'].min()),
            max_value=float(df['Price'].max()),
            value=(float(df['Price'].min()), float(df['Price'].max()))
        )
    
    with col2:
        # Brand Selector
        selected_brands = st.multiselect(
            "Select Brands",
            options=sorted(df['Brand'].unique()),
            default=df['Brand'].value_counts().head().index.tolist()
        )
    
    # Filter data based on selections
    filtered_df = df[
        (df['Price'].between(price_range[0], price_range[1])) &
        (df['Brand'].isin(selected_brands if selected_brands else df['Brand'].unique()))
    ]
    
    # Price Distribution
    st.markdown("### Price Distribution")
    fig = px.histogram(
        filtered_df,
        x='Price',
        color='Brand',
        title="Price Distribution by Brand",
        labels={'Price': 'Price ($)', 'count': 'Number of Models'},
        nbins=30
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Features
    st.markdown("### Price vs Features")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x='Horsepower',
            y='Price',
            color='Brand',
            size='Engine_Size',
            hover_data=['Model', 'Year'],
            title="Price vs Horsepower"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            filtered_df,
            x='Top_Speed',
            y='Price',
            color='Brand',
            size='Engine_Size',
            hover_data=['Model', 'Year'],
            title="Price vs Top Speed"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Price Trends
    st.markdown("### Price Trends")
    yearly_avg = filtered_df.groupby(['Brand', 'Year'])['Price'].mean().reset_index()
    fig = px.line(
        yearly_avg,
        x='Year',
        y='Price',
        color='Brand',
        title="Average Price Trends by Brand",
        labels={'Price': 'Average Price ($)', 'Year': 'Year'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price Statistics
    st.markdown("### Price Statistics by Brand")
    price_stats = filtered_df.groupby('Brand').agg({
        'Price': ['count', 'mean', 'std', 'min', 'max']
    }).round(2)
    
    price_stats.columns = ['Count', 'Mean Price', 'Std Dev', 'Min Price', 'Max Price']
    price_stats = price_stats.sort_values(('Mean Price'), ascending=False)
    
    # Format currency columns
    for col in ['Mean Price', 'Min Price', 'Max Price']:
        price_stats[col] = price_stats[col].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(price_stats, use_container_width=True)
