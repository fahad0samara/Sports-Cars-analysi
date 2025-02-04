import streamlit as st
import pandas as pd
import plotly.express as px

def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('Elite Sports Cars in Data.csv')
        
        # Basic preprocessing
        numeric_columns = ['Price', 'Horsepower', 'Torque', 'Engine_Size', 
                         'Top_Speed', 'Acceleration_0_100', 'Fuel_Efficiency']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df['Safety_Rating'] = df['Safety_Rating'].fillna(df['Safety_Rating'].mean())
        df['Production_Units'] = df['Production_Units'].fillna(df['Production_Units'].median())
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def apply_custom_styling():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        .car-stat-card {
            background: linear-gradient(135deg, #1e2761 0%, #2d3a8c 100%);
            border-radius: 15px;
            padding: 20px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }
        .car-stat-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .car-stat-label {
            font-size: 14px;
            opacity: 0.8;
            text-transform: uppercase;
        }
        .car-header {
            background: linear-gradient(135deg, #ff4e50 0%, #f9d423 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-weight: bold;
        }
        .car-subheader {
            color: #2d3a8c;
            border-left: 4px solid #ff4e50;
            padding-left: 10px;
            margin: 15px 0;
        }
        .highlight-metric {
            color: #ff4e50;
            font-weight: bold;
        }
        .author-name {
            color: #ff4e50;
            font-size: 16px;
            font-style: italic;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_number(value):
    """Format number with commas"""
    return f"{value:,}"

def calculate_percentage_change(new_value, old_value):
    """Calculate percentage change"""
    return ((new_value - old_value) / old_value) * 100
