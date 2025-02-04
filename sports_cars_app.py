import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Elite Sports Cars Analytics by Fahad",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
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

# Title and description
st.title("Elite Sports Cars Analytics")
st.markdown('<p class="author-name">Developed by Fahad</p>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides an interactive analysis of elite sports cars, including price predictions, 
market segments, and brand analysis. Use the sidebar to navigate through different analyses.
""")

@st.cache_data
def load_data():
    df = pd.read_csv('Elite Sports Cars in Data.csv')
    return df

@st.cache_data
def detect_outliers(df, columns):
    """Detect outliers using Isolation Forest"""
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(df[columns])
    return outliers == -1

@st.cache_data
def preprocess_data(df):
    """Preprocess data and engineer features"""
    # Create label encoders for categorical variables
    le_dict = {}
    categorical_cols = ['Brand', 'Country', 'Condition', 'Fuel_Type', 'Drivetrain', 
                       'Transmission', 'Popularity', 'Market_Demand']
    
    df_processed = df.copy()
    
    for col in categorical_cols:
        le_dict[col] = LabelEncoder()
        df_processed[f'{col}_encoded'] = le_dict[col].fit_transform(df_processed[col])
    
    # Create advanced features
    df_processed['Age'] = 2025 - df_processed['Year']
    df_processed['Power_to_Weight'] = df_processed['Horsepower'] / df_processed['Weight']
    df_processed['Performance_Score'] = (df_processed['Horsepower'] * df_processed['Torque']) / df_processed['Weight']
    df_processed['Maintenance_Score'] = df_processed['Insurance_Cost'] * df_processed['Number_of_Owners']
    
    # Calculate Rarity Score (avoid division by zero)
    median_production = df_processed['Production_Units'].median()
    df_processed['Rarity_Score'] = 1 / df_processed['Production_Units'].replace(0, median_production)
    
    # Log transform appropriate numerical features
    numeric_cols = ['Price', 'Mileage', 'Insurance_Cost', 'Production_Units']
    for col in numeric_cols:
        df_processed[f'{col}_log'] = np.log1p(df_processed[col])
    
    # Detect outliers
    outlier_cols = ['Price', 'Horsepower', 'Mileage', 'Insurance_Cost']
    df_processed['is_outlier'] = detect_outliers(df_processed, outlier_cols)
    
    # Create price segments
    df_processed['PriceSegment'] = pd.qcut(df_processed['Price'], q=4, labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
    
    return df_processed, le_dict

# Load and preprocess data
df = load_data()
df_processed, le_dict = preprocess_data(df)

# Add new visualization functions
def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numerical_cols].corr()
    fig = px.imshow(corr, 
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu",
                    aspect="auto")
    return fig

def plot_3d_scatter(df):
    """Create 3D scatter plot"""
    fig = px.scatter_3d(df, 
                        x='Horsepower', 
                        y='Weight', 
                        z='Price',
                        color='Brand',
                        title="3D Price Analysis",
                        labels={'Price': 'Price ($)', 
                               'Horsepower': 'Horsepower (HP)', 
                               'Weight': 'Weight (kg)'},
                        hover_data=['Model', 'Year'])
    return fig

def plot_parallel_coordinates(df):
    """Create parallel coordinates plot"""
    features = ['Price', 'Horsepower', 'Torque', 'Top_Speed', 'Engine_Size']
    fig = px.parallel_coordinates(df, 
                                dimensions=features,
                                color='Brand',
                                title="Multi-dimensional Feature Analysis")
    return fig

def plot_radar_chart(df, brands):
    """Create radar chart comparing brands"""
    metrics = ['Horsepower', 'Top_Speed', 'Engine_Size', 'Fuel_Efficiency', 'Safety_Rating']
    brand_metrics = df[df['Brand'].isin(brands)].groupby('Brand')[metrics].mean()
    
    # Normalize the metrics
    brand_metrics_normalized = (brand_metrics - brand_metrics.min()) / (brand_metrics.max() - brand_metrics.min())
    
    fig = go.Figure()
    for brand in brands:
        values = brand_metrics_normalized.loc[brand].tolist()
        values.append(values[0])  # Complete the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=brand,
            fill='toself'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Brand Performance Comparison"
    )
    return fig

def plot_bubble_map(df):
    """Create a bubble map showing global distribution of cars"""
    # Group by country and count cars
    map_df = df.groupby('Country').agg({
        'Model': 'count',
        'Price': 'mean'
    }).reset_index()
    map_df.columns = ['Country', 'Cars_Count', 'Avg_Price']
    
    # Get country coordinates
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
    
    # Add coordinates to the dataframe
    map_df['Latitude'] = map_df['Country'].map(lambda x: country_coords.get(x, [0, 0])[0])
    map_df['Longitude'] = map_df['Country'].map(lambda x: country_coords.get(x, [0, 0])[1])
    
    # Create bubble map
    fig = px.scatter_mapbox(
        map_df,
        lat='Latitude',
        lon='Longitude',
        size='Cars_Count',  # Use Cars_Count for bubble size
        color='Avg_Price',  # Color by average price
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

def plot_performance_spider(df, car_models):
    """Create spider plot for car model comparison"""
    # Use only available metrics
    metrics = ['Horsepower', 'Top_Speed', 'Engine_Size', 'Fuel_Efficiency', 'Safety_Rating']
    
    model_metrics = df[df['Model'].isin(car_models)].set_index('Model')[metrics]
    
    # Normalize the metrics
    model_metrics_normalized = (model_metrics - model_metrics.min()) / (model_metrics.max() - model_metrics.min())
    
    fig = go.Figure()
    for model in car_models:
        if model in model_metrics_normalized.index:
            # Convert Series to list using values.tolist()
            values = model_metrics_normalized.loc[model].values.tolist()
            values.append(values[0])  # Complete the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                name=model,
                fill='toself'
            ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Comparison"
    )
    return fig

def plot_price_heatmap(df):
    """Create price heatmap by brand and year"""
    pivot_data = df.pivot_table(
        values='Price',
        index='Brand',
        columns=pd.qcut(df['Year'], q=5, labels=['Oldest', 'Old', 'Mid', 'New', 'Newest']),
        aggfunc='mean'
    ).round(2)
    
    fig = px.imshow(pivot_data,
                    labels=dict(x="Time Period", y="Brand", color="Average Price"),
                    aspect="auto",
                    title="Price Evolution by Brand")
    return fig

# Add new sections and enhance existing ones
# Sidebar navigation
analysis_type = st.sidebar.selectbox(
    "Choose Analysis",
    ["Overview", "Price Analysis", "Performance Analysis", "Brand Analysis", "Market Segments", "Price Prediction", "Market Trends", "Comparative Analysis", "Geographic Analysis"]
)

# Overview
if analysis_type == "Overview":
    st.markdown('<h1 class="car-header">🏎️ Elite Sports Cars Analytics</h1>', unsafe_allow_html=True)
    
    # Quick Stats in a grid
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
    
    # Top Brands Section
    st.markdown('<h2 class="car-subheader">Top Performing Brands</h2>', unsafe_allow_html=True)
    
    # Create two columns for brand statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Most expensive brands
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
        # Most powerful brands
        top_hp_brands = df.groupby('Brand')['Horsepower'].mean().sort_values(ascending=False).head(5)
        fig = px.bar(
            top_hp_brands,
            title="Top 5 Most Powerful Brands",
            labels={'value': 'Average Horsepower', 'Brand': 'Brand'},
            color_discrete_sequence=['#2d3a8c']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price Distribution
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
    
    # Performance Metrics
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

# Price Analysis
elif analysis_type == "Price Analysis":
    st.header("Price Analysis")
    
    # Price Distribution
    st.subheader("Price Distribution")
    fig = px.histogram(df, x="Price", nbins=50, title="Price Distribution")
    st.plotly_chart(fig)
    
    # Price by Brand (Top 10)
    st.subheader("Average Price by Brand (Top 10)")
    brand_avg_price = df.groupby('Brand')['Price'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(brand_avg_price, title="Top 10 Brands by Average Price")
    st.plotly_chart(fig)
    
    # Price vs Horsepower
    st.subheader("Price vs Horsepower")
    fig = px.scatter(df, x="Horsepower", y="Price", color="Brand",
                    title="Price vs Horsepower by Brand")
    st.plotly_chart(fig)

# Performance Analysis
elif analysis_type == "Performance Analysis":
    st.header("Performance Analysis")
    
    # Add tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Basic Analysis", "Advanced Metrics", "3D Analysis"])
    
    with tab1:
        st.subheader("Performance Distribution")
        metric = st.selectbox(
            "Select Performance Metric",
            ["Horsepower", "Torque", "Top_Speed", "Acceleration_0_100"],
            key="perf_metric"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=metric, nbins=50,
                             title=f"{metric} Distribution")
            st.plotly_chart(fig)
        
        with col2:
            fig = px.box(df, x="Brand", y=metric,
                        title=f"{metric} by Brand")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Advanced Performance Metrics")
        
        # Power-to-Weight Analysis
        fig = px.scatter(df_processed,
                        x="Power_to_Weight",
                        y="Price",
                        color="Brand",
                        size="Engine_Size",
                        hover_data=["Model", "Year"],
                        title="Power-to-Weight Ratio vs Price")
        st.plotly_chart(fig)
        
        # Performance Index Analysis
        fig = px.scatter(df_processed,
                        x="Performance_Score",
                        y="Price",
                        color="Brand",
                        size="Top_Speed",
                        hover_data=["Model", "Year"],
                        title="Performance Score vs Price")
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("3D Performance Analysis")
        fig = plot_3d_scatter(df)
        st.plotly_chart(fig)
        
        st.subheader("Multi-dimensional Analysis")
        fig = plot_parallel_coordinates(df)
        st.plotly_chart(fig)

# Brand Analysis
elif analysis_type == "Brand Analysis":
    st.header("Brand Analysis")
    
    # Brand Market Share
    st.subheader("Brand Market Share")
    brand_counts = df['Brand'].value_counts().head(10)
    fig = px.pie(values=brand_counts.values, names=brand_counts.index, 
                 title="Top 10 Brands Market Share")
    st.plotly_chart(fig)
    
    # Brand Performance Comparison
    st.subheader("Brand Performance Comparison")
    selected_brands = st.multiselect(
        "Select brands to compare (max 5):",
        df['Brand'].unique(),
        default=df['Brand'].value_counts().head(3).index.tolist(),
        max_selections=5,
        key="brand_comparison"
    )
    
    if selected_brands:
        brand_metrics = df[df['Brand'].isin(selected_brands)].groupby('Brand').agg({
            'Price': 'mean',
            'Horsepower': 'mean',
            'Top_Speed': 'mean'
        }).round(2)
        
        fig = px.bar(brand_metrics, barmode='group',
                    title="Brand Performance Metrics Comparison")
        st.plotly_chart(fig)

# Market Segments
elif analysis_type == "Market Segments":
    st.header("Market Segment Analysis")
    
    # Segment Distribution
    st.subheader("Price Segment Distribution")
    fig = px.pie(df_processed, names='PriceSegment', title="Market Segments Distribution")
    st.plotly_chart(fig)
    
    # Segment Characteristics
    st.subheader("Segment Characteristics")
    segment_stats = df_processed.groupby('PriceSegment').agg({
        'Price': ['mean', 'count'],
        'Horsepower': 'mean',
        'Top_Speed': 'mean'
    }).round(2)
    st.dataframe(segment_stats)
    
    # Segment Performance Comparison
    st.subheader("Segment Performance Comparison")
    metric = st.selectbox(
        "Select Performance Metric",
        ["Horsepower", "Top_Speed", "Acceleration_0_100", "Price"]
    )
    
    fig = px.box(df_processed, x='PriceSegment', y=metric,
                 title=f"{metric} Distribution by Segment")
    st.plotly_chart(fig)

# Price Prediction
elif analysis_type == "Price Prediction":
    st.header("Price Prediction Model")
    
    @st.cache_resource
    def train_model(df_processed):
        """Train the price prediction model"""
        features = [
            'Age', 'Engine_Size', 'Horsepower', 'Torque', 'Weight', 
            'Top_Speed', 'Acceleration_0_100', 'Fuel_Efficiency',
            'CO2_Emissions', 'Mileage_log', 'Safety_Rating',
            'Number_of_Owners', 'Insurance_Cost_log', 'Production_Units_log',
            'Power_to_Weight', 'Performance_Score', 'Maintenance_Score',
            'Rarity_Score', 'Brand_encoded', 'Country_encoded',
            'Condition_encoded', 'Fuel_Type_encoded', 'Drivetrain_encoded',
            'Transmission_encoded', 'Market_Demand_encoded'
        ]
        
        df_clean = df_processed[~df_processed['is_outlier']].copy()
        X = df_clean[features]
        y = df_clean['Price_log']
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        return model, scaler, features, X.mean(), X.median()

    model, scaler, features, feature_means, feature_medians = train_model(df_processed)
    
    st.subheader("Enter Car Specifications")
    st.markdown("Provide the specifications of the car to get a price estimate.")
    
    col1, col2 = st.columns(2)
    with col1:
        horsepower = st.number_input("Horsepower", min_value=100, max_value=1500, value=300, key="pred_hp")
        torque = st.number_input("Torque (Nm)", min_value=100, max_value=1500, value=400, key="pred_torque")
        weight = st.number_input("Weight (kg)", min_value=500, max_value=3000, value=1500, key="pred_weight")
    
    with col2:
        age = st.number_input("Age (years)", min_value=0, max_value=50, value=5, key="pred_age")
        mileage = st.number_input("Mileage", min_value=0, max_value=200000, value=50000, key="pred_mileage")
        condition = st.selectbox("Condition", df['Condition'].unique(), key="pred_condition")

    if st.button("Predict Price"):
        try:
            # Create sample input using feature means/medians for missing values
            sample = pd.DataFrame({
                'Age': [age],
                'Engine_Size': [feature_means['Engine_Size']],
                'Horsepower': [horsepower],
                'Torque': [torque],
                'Weight': [weight],
                'Top_Speed': [feature_means['Top_Speed']],
                'Acceleration_0_100': [feature_means['Acceleration_0_100']],
                'Fuel_Efficiency': [feature_means['Fuel_Efficiency']],
                'CO2_Emissions': [feature_means['CO2_Emissions']],
                'Safety_Rating': [feature_means['Safety_Rating']],
                'Number_of_Owners': [1],
                'Power_to_Weight': [horsepower / weight],
                'Performance_Score': [(horsepower * torque) / weight],
                'Maintenance_Score': [feature_means['Maintenance_Score']],
                'Rarity_Score': [feature_medians['Rarity_Score']],
                'Brand_encoded': [df_processed['Brand_encoded'].mode()[0]],
                'Country_encoded': [df_processed['Country_encoded'].mode()[0]],
                'Condition_encoded': [le_dict['Condition'].transform([condition])[0]],
                'Fuel_Type_encoded': [df_processed['Fuel_Type_encoded'].mode()[0]],
                'Drivetrain_encoded': [df_processed['Drivetrain_encoded'].mode()[0]],
                'Transmission_encoded': [df_processed['Transmission_encoded'].mode()[0]],
                'Market_Demand_encoded': [df_processed['Market_Demand_encoded'].mode()[0]],
                'Mileage_log': [np.log1p(mileage)],
                'Insurance_Cost_log': [feature_means['Insurance_Cost_log']],
                'Production_Units_log': [feature_means['Production_Units_log']]
            })
            
            # Scale features
            sample_scaled = pd.DataFrame(
                scaler.transform(sample[features]),
                columns=features
            )
            
            # Make prediction
            predicted_price = np.exp(model.predict(sample_scaled)[0])
            st.success(f"Estimated Price: ${predicted_price:,.2f}")
            
            # Show confidence range (±15%)
            st.info(f"Price Range: ${predicted_price * 0.85:,.2f} - ${predicted_price * 1.15:,.2f}")
            
            # Show feature values used for prediction
            if st.checkbox("Show detailed feature values"):
                st.write("Feature values used for prediction:")
                display_features = {
                    'Horsepower': horsepower,
                    'Torque': torque,
                    'Weight': weight,
                    'Power to Weight': round(sample['Power_to_Weight'][0], 2),
                    'Performance Score': round(sample['Performance_Score'][0], 2),
                    'Age': age,
                    'Mileage': mileage,
                    'Condition': condition
                }
                st.json(display_features)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.info("Please make sure all input values are within reasonable ranges.")
    
    # Add feature importance visualization
    with st.expander("View Feature Importance Analysis"):
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance.head(15), 
                    x='importance', 
                    y='feature',
                    title="Top 15 Most Important Features",
                    orientation='h')
        st.plotly_chart(fig)
        
        # Show correlation heatmap
        st.subheader("Feature Correlations")
        fig = plot_correlation_heatmap(df_processed)
        st.plotly_chart(fig)
    
    # Add similar cars feature
    with st.expander("Find Similar Cars"):
        st.subheader("Find Similar Cars")
        st.markdown("This will show existing cars with similar specifications to your input.")
        
        if 'last_prediction' in st.session_state:
            pred_price = st.session_state.last_prediction
            similar_cars = df[
                (df['Price'].between(pred_price * 0.8, pred_price * 1.2)) &
                (df['Horsepower'].between(horsepower * 0.8, horsepower * 1.2))
            ]
            
            if not similar_cars.empty:
                st.write(f"Found {len(similar_cars)} similar cars:")
                st.dataframe(
                    similar_cars[['Brand', 'Model', 'Year', 'Price', 'Horsepower', 'Condition']]
                )
            else:
                st.info("No similar cars found in the database.")

# Market Trends
elif analysis_type == "Market Trends":
    st.header("Market Trends Analysis")
    
    # Price trends over years
    st.subheader("Price Trends Over Years")
    yearly_avg = df.groupby('Year').agg({
        'Price': 'mean',
        'Horsepower': 'mean',
        'Production_Units': 'sum'
    }).reset_index()
    
    fig = px.line(yearly_avg, 
                  x='Year', 
                  y='Price',
                  title="Average Car Price by Year")
    st.plotly_chart(fig)
    
    # Market segment analysis
    st.subheader("Market Segment Analysis")
    segment_metrics = df_processed.groupby('PriceSegment').agg({
        'Price': ['count', 'mean', 'std'],
        'Horsepower': 'mean',
        'Production_Units': 'sum'
    }).round(2)
    
    fig = px.sunburst(df_processed, 
                      path=['PriceSegment', 'Brand'],
                      values='Price',
                      title="Market Segments and Brands")
    st.plotly_chart(fig)
    
    # Brand performance trends
    st.subheader("Brand Performance Trends")
    brand_metrics = df.groupby('Brand').agg({
        'Price': ['count', 'mean'],
        'Horsepower': 'mean'
    }).round(2)
    
    fig = px.scatter(brand_metrics.reset_index(), 
                    x=('Price', 'mean'),
                    y=('Horsepower', 'mean'),
                    size=('Price', 'count'),
                    hover_name='Brand',
                    title="Brand Position Map")
    st.plotly_chart(fig)

    # Add market overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_value = (df['Price'] * df['Production_Units']).sum()
        st.metric(
            "Total Market Value",
            f"${total_value/1e9:.1f}B",
            help="Total value of all cars in database"
        )
    
    with col2:
        avg_price_change = ((df[df['Year'] >= 2020]['Price'].mean() / 
                            df[df['Year'] < 2020]['Price'].mean()) - 1) * 100
        st.metric(
            "Price Growth",
            f"{avg_price_change:.1f}%",
            help="Average price change in recent years"
        )
    
    with col3:
        market_concentration = (df.groupby('Brand')['Production_Units'].sum().max() / 
                              df['Production_Units'].sum()) * 100
        st.metric(
            "Market Concentration",
            f"{market_concentration:.1f}%",
            help="Percentage of market held by largest manufacturer"
        )
    
    with col4:
        premium_share = (len(df[df['Price'] > df['Price'].quantile(0.75)]) / 
                        len(df)) * 100
        st.metric(
            "Premium Segment",
            f"{premium_share:.1f}%",
            help="Percentage of cars in premium segment"
        )
    
    # Add tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["💰 Price Trends", "🏢 Market Structure", "🌍 Regional Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price evolution heatmap
            st.plotly_chart(plot_price_heatmap(df), use_container_width=True)
        
        with col2:
            # Price trend by segment
            segment_trends = df.pivot_table(
                values='Price',
                index='Year',
                columns='PriceSegment',
                aggfunc='mean'
            ).fillna(method='ffill')
            
            fig = px.line(segment_trends,
                         title="Price Trends by Segment")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Market share analysis
        market_share = df.groupby('Brand')['Production_Units'].sum().sort_values(ascending=True)
        market_share_pct = (market_share / market_share.sum() * 100).round(2)
        
        fig = go.Figure(go.Bar(
            x=market_share_pct.values,
            y=market_share_pct.index,
            orientation='h'
        ))
        fig.update_layout(title="Market Share by Brand (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment distribution
        fig = px.sunburst(df,
                         path=['PriceSegment', 'Brand'],
                         values='Production_Units',
                         title="Market Segmentation")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Regional market analysis
        region_stats = df.groupby('Country').agg({
            'Price': ['mean', 'std'],
            'Production_Units': 'sum',
            'Model': 'count'
        }).round(2)
        
        region_stats.columns = ['Avg Price', 'Price Std', 'Total Production', 'Models']
        region_stats = region_stats.sort_values('Total Production', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(region_stats, height=400)
        
        with col2:
            fig = px.scatter(region_stats.reset_index(),
                           x='Total Production',
                           y='Avg Price',
                           size='Models',
                           color='Models',
                           hover_name='Country',
                           title="Regional Market Overview")
            st.plotly_chart(fig, use_container_width=True)

# Comparative Analysis
elif analysis_type == "Comparative Analysis":
    st.header("Comparative Analysis")
    
    # Brand Comparison
    st.subheader("Brand Comparison")
    selected_brands = st.multiselect(
        "Select brands to compare (max 5):",
        df['Brand'].unique(),
        default=df['Brand'].value_counts().head(3).index.tolist(),
        max_selections=5,
        key="brand_comparison"
    )
    
    if selected_brands:
        # Radar Chart
        st.plotly_chart(plot_radar_chart(df, selected_brands))
        
        # Detailed Metrics Table
        metrics = ['Price', 'Horsepower', 'Torque', 'Top_Speed', 'Engine_Size', 
                  'Fuel_Efficiency', 'Safety_Rating', 'Production_Units']
        
        comparison_df = df[df['Brand'].isin(selected_brands)].groupby('Brand')[metrics].agg([
            'mean', 'min', 'max', 'std'
        ]).round(2)
        
        st.write("Detailed Comparison:")
        st.dataframe(comparison_df)
        
        # Model Distribution
        st.subheader("Model Distribution by Year")
        fig = px.scatter(df[df['Brand'].isin(selected_brands)],
                        x='Year',
                        y='Price',
                        color='Brand',
                        size='Engine_Size',
                        hover_data=['Model', 'Horsepower'],
                        title="Model Distribution Over Time")
        st.plotly_chart(fig)

# Geographic Analysis
elif analysis_type == "Geographic Analysis":
    st.markdown("## Geographic Analysis")
    
    # Global Distribution
    st.markdown("### Global Distribution")
    try:
        st.plotly_chart(plot_bubble_map(df), use_container_width=True)
        
        # Add country statistics
        st.markdown("### Country Statistics")
        country_stats = df.groupby('Country').agg({
            'Model': 'count',
            'Price': ['mean', 'min', 'max'],
            'Horsepower': 'mean'
        }).round(2)
        
        # Format the statistics
        country_stats.columns = ['Number of Models', 'Average Price', 'Min Price', 'Max Price', 'Average Horsepower']
        country_stats = country_stats.sort_values('Number of Models', ascending=False)
        
        # Format currency columns
        for col in ['Average Price', 'Min Price', 'Max Price']:
            country_stats[col] = country_stats[col].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(country_stats, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating geographic visualization: {str(e)}")
        st.info("Some geographic data might be missing or incorrect.")

    # Country Comparison
    st.markdown("### Country Comparison")
    selected_countries = st.multiselect(
        "Select countries to compare:",
        df['Country'].unique(),
        default=df['Country'].value_counts().head(3).index.tolist(),
        key="geo_countries"
    )
    
    if selected_countries:
        fig = px.box(df[df['Country'].isin(selected_countries)],
                    x='Country',
                    y='Price',
                    color='Country',
                    points='all',
                    title="Price Distribution by Country")
        st.plotly_chart(fig)
        
        # Performance by Country
        fig = px.scatter(df[df['Country'].isin(selected_countries)],
                        x='Horsepower',
                        y='Price',
                        color='Country',
                        size='Engine_Size',
                        hover_data=['Brand', 'Model'],
                        title="Performance vs Price by Country")
        st.plotly_chart(fig)

# Add sidebar filters that apply to all sections
with st.sidebar:
    st.markdown("### Global Filters")
    with st.expander("Apply Filters"):
        # Year range filter
        year_range = st.slider(
            "Year Range",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            key="global_year_range"
        )
        
        # Price range filter
        price_range = st.slider(
            "Price Range ($)",
            min_value=float(df['Price'].min()),
            max_value=float(df['Price'].max()),
            value=(float(df['Price'].min()), float(df['Price'].max())),
            key="global_price_range"
        )
        
        # Brand filter
        selected_brands = st.multiselect(
            "Select Brands",
            df['Brand'].unique(),
            default=[],
            key="global_brands"
        )
        
        # Apply filters
        if selected_brands:
            df = df[df['Brand'].isin(selected_brands)]
        df = df[
            (df['Year'].between(year_range[0], year_range[1])) &
            (df['Price'].between(price_range[0], price_range[1]))
        ]

# Add help section in sidebar
with st.sidebar:
    st.markdown("---")
    with st.expander("Help & Information"):
        st.markdown("""
        ### How to Use This Dashboard
        
        1. **Navigation**: Use the dropdown above to switch between different analyses
        2. **Filters**: Apply global filters to focus on specific data
        3. **Interactivity**: Most charts are interactive - hover, zoom, and click for more details
        4. **Downloads**: Many sections allow you to download the filtered data
        
        ### Analysis Sections
        
        - **Overview**: Basic statistics and data explorer
        - **Price Analysis**: Detailed price trends and distributions
        - **Performance Analysis**: Car performance metrics and comparisons
        - **Brand Analysis**: Brand-specific insights
        - **Market Trends**: Historical trends and future predictions
        - **Geographic Analysis**: Country-wise distribution and comparison
        - **Comparative Analysis**: Side-by-side brand and model comparison
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("Created by Fahad")
st.sidebar.markdown("Data last updated: 2025-02-04")
