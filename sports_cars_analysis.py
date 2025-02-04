import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

print("Loading and preprocessing data...")
df = pd.read_csv('Elite Sports Cars in Data.csv')

def detect_outliers(df, columns):
    """Detect outliers using Isolation Forest"""
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(df[columns])
    return outliers == -1

def preprocess_data(df):
    # Create label encoders for categorical variables
    le_dict = {}
    categorical_cols = ['Brand', 'Country', 'Condition', 'Fuel_Type', 'Drivetrain', 
                       'Transmission', 'Popularity', 'Market_Demand']
    
    for col in categorical_cols:
        le_dict[col] = LabelEncoder()
        df[f'{col}_encoded'] = le_dict[col].fit_transform(df[col])
    
    # Create advanced features
    df['Age'] = 2025 - df['Year']
    df['Power_to_Weight'] = df['Horsepower'] / df['Weight']
    df['Performance_Index'] = (df['Horsepower'] * df['Torque']) / df['Weight']
    df['Maintenance_Score'] = df['Insurance_Cost'] * df['Number_of_Owners']
    df['Rarity_Score'] = 1 / df['Production_Units']
    
    # Log transform appropriate numerical features
    numeric_cols = ['Price', 'Mileage', 'Insurance_Cost', 'Production_Units']
    for col in numeric_cols:
        df[f'{col}_log'] = np.log1p(df[col])
    
    # Detect outliers
    outlier_cols = ['Price', 'Horsepower', 'Mileage', 'Insurance_Cost']
    df['is_outlier'] = detect_outliers(df, outlier_cols)
    
    # Create price segments
    df['PriceSegment'] = pd.qcut(df['Price'], q=4, labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
    
    return df, le_dict

def prepare_features(df):
    features = [
        'Age', 'Engine_Size', 'Horsepower', 'Torque', 'Weight', 
        'Top_Speed', 'Acceleration_0_100', 'Fuel_Efficiency',
        'CO2_Emissions', 'Mileage_log', 'Safety_Rating',
        'Number_of_Owners', 'Insurance_Cost_log', 'Production_Units_log',
        'Power_to_Weight', 'Performance_Index', 'Maintenance_Score',
        'Rarity_Score', 'Brand_encoded', 'Country_encoded',
        'Condition_encoded', 'Fuel_Type_encoded', 'Drivetrain_encoded',
        'Transmission_encoded', 'Market_Demand_encoded'
    ]
    
    X = df[features]
    y = df['Price_log']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    return X_scaled, y, features

print("Processing data and engineering features...")
df, le_dict = preprocess_data(df)

# Remove outliers for better model performance
df_clean = df[~df['is_outlier']].copy()
X, y, features = prepare_features(df_clean)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: ${np.exp(rmse):,.2f}")
print(f"MAE: ${np.exp(mae):,.2f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Create visualizations
plt.figure(figsize=(20, 15))

# 1. Price Distribution by Segment
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='PriceSegment', y='Price')
plt.xticks(rotation=45)
plt.title('Price Distribution by Segment')

# 2. Feature Importance
plt.subplot(2, 2, 2)
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features')
plt.xlabel('Feature Importance')

# 3. Actual vs Predicted Prices
plt.subplot(2, 2, 3)
plt.scatter(np.exp(y_test), np.exp(y_pred), alpha=0.5)
plt.plot([np.exp(y_test).min(), np.exp(y_test).max()], 
         [np.exp(y_test).min(), np.exp(y_test).max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

# 4. Price vs Horsepower (with outliers highlighted)
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Horsepower', y='Price', hue='is_outlier', alpha=0.6)
plt.title('Price vs Horsepower (Outliers Highlighted)')

plt.tight_layout()
plt.savefig('car_analysis.png')

# Save analysis report
print("\nSaving detailed analysis report...")
with open('car_analysis_report.txt', 'w') as f:
    f.write("=== Elite Sports Cars Analysis Report ===\n\n")
    
    f.write("1. Model Performance:\n")
    f.write(f"R² Score: {r2:.3f}\n")
    f.write(f"RMSE: ${np.exp(rmse):,.2f}\n")
    f.write(f"MAE: ${np.exp(mae):,.2f}\n\n")
    
    f.write("2. Top 10 Most Important Features:\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"{row['feature']}: {row['importance']:.3f}\n")
    
    f.write("\n3. Price Segment Statistics:\n")
    segment_stats = df.groupby('PriceSegment').agg({
        'Price': ['count', 'mean', 'std'],
        'Horsepower': 'mean',
        'Mileage': 'mean'
    }).round(2)
    f.write(segment_stats.to_string())
    
    f.write("\n\n4. Outlier Analysis:\n")
    f.write(f"Total outliers detected: {df['is_outlier'].sum()}\n")
    f.write(f"Percentage of outliers: {(df['is_outlier'].mean() * 100):.1f}%\n")
    
    f.write("\n5. Feature Correlations with Price:\n")
    correlations = df.select_dtypes(include=[np.number]).corr()['Price_log'].sort_values(ascending=False)
    f.write(correlations.head(10).to_string())
    
    f.write("\n\n6. Brand Analysis:\n")
    brand_stats = df.groupby('Brand').agg({
        'Price': ['mean', 'count'],
        'Horsepower': 'mean',
        'Market_Demand': lambda x: x.mode().iloc[0] if not x.empty else None
    }).round(2)
    f.write(brand_stats.head(10).to_string())

print("\nAnalysis complete! Check 'car_analysis.png' and 'car_analysis_report.txt' for detailed results.")
