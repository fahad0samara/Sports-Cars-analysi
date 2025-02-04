import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_performance_spider(df, car_models):
    """Create spider plot for car model comparison"""
    metrics = ['Horsepower', 'Top_Speed', 'Engine_Size', 'Fuel_Efficiency', 'Safety_Rating']
    
    model_metrics = df[df['Model'].isin(car_models)].set_index('Model')[metrics]
    model_metrics_normalized = (model_metrics - model_metrics.min()) / (model_metrics.max() - model_metrics.min())
    
    fig = go.Figure()
    for model in car_models:
        if model in model_metrics_normalized.index:
            values = model_metrics_normalized.loc[model].values.tolist()
            values.append(values[0])
            
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

def render_performance_comparison(df, selected_model):
    """Render performance comparison section"""
    try:
        model_data = df[df['Model'] == selected_model].iloc[0]
        similar_models = df[
            (df['Model'] != selected_model) &
            (df['Price'].between(model_data['Price'] * 0.8, model_data['Price'] * 1.2)) &
            (df['Horsepower'].between(model_data['Horsepower'] * 0.8, model_data['Horsepower'] * 1.2))
        ]
        
        if not similar_models.empty:
            comparison_df = pd.concat([
                df[df['Model'] == selected_model],
                similar_models.head(3)
            ])
            
            st.markdown("#### Specifications Comparison")
            comparison_metrics = ['Brand', 'Model', 'Year', 'Price', 'Horsepower', 
                                'Torque', 'Engine_Size', 'Top_Speed', 'Fuel_Efficiency']
            
            formatted_comparison = comparison_df[comparison_metrics].copy()
            formatted_comparison['Price'] = formatted_comparison['Price'].apply(lambda x: f"${x:,.2f}")
            formatted_comparison['Horsepower'] = formatted_comparison['Horsepower'].apply(lambda x: f"{x:,.0f} HP")
            formatted_comparison['Torque'] = formatted_comparison['Torque'].apply(lambda x: f"{x:,.0f} Nm")
            formatted_comparison['Top_Speed'] = formatted_comparison['Top_Speed'].apply(lambda x: f"{x:,.0f} km/h")
            formatted_comparison['Engine_Size'] = formatted_comparison['Engine_Size'].apply(lambda x: f"{x:.1f}L")
            formatted_comparison['Fuel_Efficiency'] = formatted_comparison['Fuel_Efficiency'].apply(lambda x: f"{x:.1f} mpg")
            
            st.dataframe(formatted_comparison.set_index('Model'), use_container_width=True)
            
            st.markdown("#### Performance Comparison")
            comparison_models = [selected_model] + similar_models['Model'].values.tolist()[:3]
            fig = plot_performance_spider(comparison_df, comparison_models)
            st.plotly_chart(fig, use_container_width=True)
            
            render_detailed_comparison(comparison_df, selected_model, similar_models)
        else:
            st.info("No similar models found in the selected price and performance range.")
    
    except Exception as e:
        st.error(f"Error comparing models: {str(e)}")
        st.info("Try adjusting the filters to find more comparable models.")

def render_detailed_comparison(comparison_df, selected_model, similar_models):
    """Render detailed comparison view"""
    with st.expander("View Detailed Comparison"):
        base_model = comparison_df[comparison_df['Model'] == selected_model].iloc[0]
        
        for similar_model in similar_models.head(3).itertuples():
            st.markdown(f"#### {similar_model.Brand} {similar_model.Model}")
            
            col1, col2 = st.columns(2)
            with col1:
                price_diff = ((similar_model.Price - base_model['Price']) / base_model['Price']) * 100
                hp_diff = ((similar_model.Horsepower - base_model['Horsepower']) / base_model['Horsepower']) * 100
                torque_diff = ((similar_model.Torque - base_model['Torque']) / base_model['Torque']) * 100
                
                st.metric("Price Difference", 
                        f"${similar_model.Price:,.2f}", 
                        f"{price_diff:+.1f}%")
                st.metric("Horsepower Difference", 
                        f"{similar_model.Horsepower:,.0f} HP", 
                        f"{hp_diff:+.1f}%")
                st.metric("Torque Difference", 
                        f"{similar_model.Torque:,.0f} Nm", 
                        f"{torque_diff:+.1f}%")
            
            with col2:
                speed_diff = ((similar_model.Top_Speed - base_model['Top_Speed']) / base_model['Top_Speed']) * 100
                efficiency_diff = ((similar_model.Fuel_Efficiency - base_model['Fuel_Efficiency']) / base_model['Fuel_Efficiency']) * 100
                
                st.metric("Top Speed Difference", 
                        f"{similar_model.Top_Speed:,.0f} km/h", 
                        f"{speed_diff:+.1f}%")
                st.metric("Efficiency Difference", 
                        f"{similar_model.Fuel_Efficiency:.1f} mpg", 
                        f"{efficiency_diff:+.1f}%")
