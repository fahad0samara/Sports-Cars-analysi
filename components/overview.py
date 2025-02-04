import streamlit as st
import plotly.express as px
import pandas as pd

def render_overview_section(df):
    """Render the overview section of the dashboard"""
    st.markdown('<h1 class="car-header">🏎️ Elite Sports Cars Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="author-name">Developed by Fahad</p>', unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="car-stat-card">
            <div class="car-stat-label">Total Models</div>
            <div class="car-stat-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_price = df['Price'].mean()
        st.markdown(f"""
        <div class="car-stat-card">
            <div class="car-stat-label">Average Price</div>
            <div class="car-stat-value">${avg_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_hp = df['Horsepower'].mean()
        st.markdown(f"""
        <div class="car-stat-card">
            <div class="car-stat-label">Average Horsepower</div>
            <div class="car-stat-value">{avg_hp:,.0f} HP</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        brands = len(df['Brand'].unique())
        st.markdown(f"""
        <div class="car-stat-card">
            <div class="car-stat-label">Unique Brands</div>
            <div class="car-stat-value">{brands}</div>
        </div>
        """, unsafe_allow_html=True)
    
    render_brand_analysis(df)
    render_price_distribution(df)
    render_performance_metrics(df)

def render_brand_analysis(df):
    """Render brand analysis section"""
    st.markdown('<h2 class="car-subheader">Top Performing Brands</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        top_price_brands = df.groupby('Brand')['Price'].mean().sort_values(ascending=False).head(5)
        fig = px.bar(
            top_price_brands,
            title="Top 5 Most Expensive Brands",
            labels={'value': 'Average Price ($)', 'Brand': 'Brand'},
            color_discrete_sequence=['#ff4e50']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_hp_brands = df.groupby('Brand')['Horsepower'].mean().sort_values(ascending=False).head(5)
        fig = px.bar(
            top_hp_brands,
            title="Top 5 Most Powerful Brands",
            labels={'value': 'Average Horsepower', 'Brand': 'Brand'},
            color_discrete_sequence=['#2d3a8c']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def render_price_distribution(df):
    """Render price distribution section"""
    st.markdown('<h2 class="car-subheader">Price Distribution</h2>', unsafe_allow_html=True)
    fig = px.histogram(
        df,
        x='Price',
        nbins=30,
        title="Price Distribution of Sports Cars",
        labels={'Price': 'Price ($)', 'count': 'Number of Models'},
        color_discrete_sequence=['#ff4e50']
    )
    fig.update_layout(
        bargap=0.1,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def render_performance_metrics(df):
    """Render performance metrics section"""
    st.markdown('<h2 class="car-subheader">Performance Metrics</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df,
            x='Horsepower',
            y='Price',
            color='Brand',
            title="Horsepower vs Price",
            labels={'Horsepower': 'Horsepower (HP)', 'Price': 'Price ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df,
            x='Brand',
            y='Horsepower',
            title="Horsepower Distribution by Brand",
            labels={'Brand': 'Brand', 'Horsepower': 'Horsepower (HP)'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
