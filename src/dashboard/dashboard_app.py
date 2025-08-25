"""
Interactive dashboard application using Streamlit.

This module provides the DashboardApp class that creates an interactive
dashboard for visualizing business metrics and insights from the e-commerce
data processing pipeline.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import os
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dashboard.data_provider import DataProvider

logger = logging.getLogger(__name__)


class DashboardApp:
    """
    Main dashboard application using Streamlit.
    
    This class creates interactive visualizations including monthly revenue trends,
    top products, regional sales, and category discount analysis.
    """
    
    def __init__(self):
        """Initialize the DashboardApp."""
        self.data_provider = DataProvider()
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="E-commerce Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .section-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin: 1rem 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_filters(self, tables: Dict[str, pd.DataFrame], 
                      selected_months: Optional[List[str]] = None,
                      selected_regions: Optional[List[str]] = None,
                      selected_categories: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Apply interactive filters to analytical tables.
        
        Args:
            tables: Dictionary of analytical tables
            selected_months: List of selected months for filtering
            selected_regions: List of selected regions for filtering
            selected_categories: List of selected categories for filtering
            
        Returns:
            Dictionary of filtered tables
            
        Requirements: 4.7
        """
        filtered_tables = {}
        
        for table_name, df in tables.items():
            if df.empty:
                filtered_tables[table_name] = df
                continue
            
            filtered_df = df.copy()
            
            # Apply month filter
            if selected_months and 'month' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['month'].isin(selected_months)]
            
            # Apply region filter
            if selected_regions and 'region' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
            
            # Apply category filter
            if selected_categories and 'category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
            
            filtered_tables[table_name] = filtered_df
        
        return filtered_tables
    
    def _calculate_filtered_metrics(self, filtered_tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate metrics based on filtered data.
        
        Args:
            filtered_tables: Dictionary of filtered analytical tables
            
        Returns:
            Dictionary containing recalculated metrics
            
        Requirements: 4.7
        """
        try:
            metrics = {}
            
            # Calculate metrics from filtered monthly sales summary
            if 'monthly_sales_summary' in filtered_tables and not filtered_tables['monthly_sales_summary'].empty:
                monthly_data = filtered_tables['monthly_sales_summary']
                
                # Handle different possible column names
                revenue_col = 'total_revenue' if 'total_revenue' in monthly_data.columns else 'revenue'
                orders_col = 'unique_orders' if 'unique_orders' in monthly_data.columns else 'orders'
                aov_col = 'avg_order_value' if 'avg_order_value' in monthly_data.columns else 'aov'
                discount_col = 'avg_discount' if 'avg_discount' in monthly_data.columns else 'discount'
                
                if revenue_col in monthly_data.columns:
                    metrics['total_revenue'] = monthly_data[revenue_col].sum()
                else:
                    metrics['total_revenue'] = 0
                    
                if orders_col in monthly_data.columns:
                    metrics['total_orders'] = monthly_data[orders_col].sum()
                else:
                    metrics['total_orders'] = 0
                    
                if aov_col in monthly_data.columns:
                    metrics['avg_order_value'] = monthly_data[aov_col].mean()
                else:
                    metrics['avg_order_value'] = 0
                    
                if discount_col in monthly_data.columns:
                    metrics['avg_discount'] = monthly_data[discount_col].mean()
                else:
                    metrics['avg_discount'] = 0
                
                # Calculate growth metrics if we have multiple months
                if len(monthly_data) > 1 and revenue_col in monthly_data.columns:
                    monthly_data_sorted = monthly_data.sort_values('month')
                    
                    if len(monthly_data_sorted) >= 2:
                        last_revenue = monthly_data_sorted[revenue_col].iloc[-1]
                        prev_revenue = monthly_data_sorted[revenue_col].iloc[-2]
                        if prev_revenue > 0:
                            metrics['revenue_growth'] = f"{((last_revenue - prev_revenue) / prev_revenue * 100):+.1f}%"
                        else:
                            metrics['revenue_growth'] = "N/A"
                    else:
                        metrics['revenue_growth'] = "N/A"
                else:
                    metrics['revenue_growth'] = "N/A"
            else:
                # Set default values if no monthly data
                metrics['total_revenue'] = 0
                metrics['total_orders'] = 0
                metrics['avg_order_value'] = 0
                metrics['avg_discount'] = 0
                metrics['revenue_growth'] = "N/A"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating filtered metrics: {str(e)}")
            return {
                'total_revenue': 0,
                'total_orders': 0,
                'avg_order_value': 0,
                'avg_discount': 0,
                'revenue_growth': "N/A"
            }
    
    def create_revenue_trend_chart(self, monthly_data: pd.DataFrame, chart_type: str = "Line Chart") -> go.Figure:
        """
        Create monthly revenue trend visualization with line charts.
        
        Args:
            monthly_data: DataFrame with monthly sales summary
            chart_type: Type of chart to create (Line Chart, Bar Chart, Area Chart)
            
        Returns:
            Plotly figure with revenue trend
            
        Requirements: 4.1, 4.7
        """
        try:
            if monthly_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for revenue trends",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly Revenue Trend', 'Monthly Order Volume'),
                vertical_spacing=0.1,
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )
            
            # Determine revenue column name
            revenue_col = 'total_revenue' if 'total_revenue' in monthly_data.columns else 'revenue'
            orders_col = 'unique_orders' if 'unique_orders' in monthly_data.columns else 'orders'
            aov_col = 'avg_order_value' if 'avg_order_value' in monthly_data.columns else 'aov'
            
            # Check if required columns exist
            if revenue_col not in monthly_data.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Revenue data not available in the dataset",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Revenue trend - different chart types
            if chart_type == "Line Chart":
                fig.add_trace(
                    go.Scatter(
                        x=monthly_data['month'],
                        y=monthly_data[revenue_col],
                        mode='lines+markers',
                        name='Total Revenue',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            elif chart_type == "Bar Chart":
                fig.add_trace(
                    go.Bar(
                        x=monthly_data['month'],
                        y=monthly_data[revenue_col],
                        name='Total Revenue',
                        marker_color='#1f77b4',
                        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            elif chart_type == "Area Chart":
                fig.add_trace(
                    go.Scatter(
                        x=monthly_data['month'],
                        y=monthly_data[revenue_col],
                        mode='lines',
                        name='Total Revenue',
                        fill='tonexty',
                        line=dict(color='#1f77b4', width=2),
                        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Average order value on secondary axis (if available)
            if aov_col in monthly_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=monthly_data['month'],
                        y=monthly_data[aov_col],
                        mode='lines+markers',
                        name='Avg Order Value',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        marker=dict(size=6),
                        yaxis='y2',
                        hovertemplate='<b>%{x}</b><br>AOV: $%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Order count bars (if available)
            if orders_col in monthly_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=monthly_data['month'],
                        y=monthly_data[orders_col],
                        name='Order Count',
                        marker_color='#2ca02c',
                        opacity=0.7,
                        hovertemplate='<b>%{x}</b><br>Orders: %{y:,}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="Monthly Revenue and Order Trends",
                    x=0.5,
                    font=dict(size=20)
                ),
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Month", row=2, col=1)
            fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
            fig.update_yaxes(title_text="Average Order Value ($)", secondary_y=True, row=1, col=1)
            fig.update_yaxes(title_text="Number of Orders", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating revenue trend chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading revenue trends: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_top_products_chart(self, products_data: pd.DataFrame) -> go.Figure:
        """
        Implement top 10 products display with bar charts.
        
        Args:
            products_data: DataFrame with top products data
            
        Returns:
            Plotly figure with top products visualization
            
        Requirements: 4.2
        """
        try:
            if products_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for top products",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Determine column names
            revenue_col = 'total_revenue' if 'total_revenue' in products_data.columns else 'revenue'
            units_col = 'total_units' if 'total_units' in products_data.columns else 'units'
            
            # Check if required columns exist
            if revenue_col not in products_data.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Revenue data not available for products",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Take top 10 by revenue
            top_products = products_data.head(10).copy()
            
            # Create subplot for revenue and units
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Top 10 Products by Revenue', 'Top 10 Products by Units Sold'),
                horizontal_spacing=0.1
            )
            
            # Revenue bar chart
            fig.add_trace(
                go.Bar(
                    x=top_products[revenue_col],
                    y=top_products['product_name'],
                    orientation='h',
                    name='Revenue',
                    marker_color='#1f77b4',
                    text=[f'${x:,.0f}' for x in top_products[revenue_col]],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.2f}<br>Rank: %{customdata}<extra></extra>',
                    customdata=top_products.get('revenue_rank', range(1, len(top_products) + 1))
                ),
                row=1, col=1
            )
            
            # Units bar chart (if available)
            if units_col in top_products.columns:
                fig.add_trace(
                    go.Bar(
                        x=top_products[units_col],
                        y=top_products['product_name'],
                        orientation='h',
                        name='Units',
                        marker_color='#ff7f0e',
                        text=[f'{x:,.0f}' for x in top_products[units_col]],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Units: %{x:,}<br>Rank: %{customdata}<extra></extra>',
                    customdata=top_products['units_rank']
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="Top Performing Products",
                    x=0.5,
                    font=dict(size=20)
                ),
                height=500,
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(title_text="Revenue ($)", row=1, col=1)
            fig.update_xaxes(title_text="Units Sold", row=1, col=2)
            fig.update_yaxes(title_text="Product", row=1, col=1)
            fig.update_yaxes(title_text="Product", row=1, col=2)
            
            # Reverse y-axis to show highest values at top
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            fig.update_yaxes(autorange="reversed", row=1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating top products chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading top products: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_regional_sales_chart(self, regional_data: pd.DataFrame) -> go.Figure:
        """
        Build regional sales visualization (bar chart or map).
        
        Args:
            regional_data: DataFrame with regional performance data
            
        Returns:
            Plotly figure with regional sales visualization
            
        Requirements: 4.3
        """
        try:
            if regional_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for regional sales",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Create subplot with multiple metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Revenue by Region', 
                    'Market Share by Region',
                    'Average Order Value by Region',
                    'Orders by Region'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Determine column names
            revenue_col = 'total_revenue' if 'total_revenue' in regional_data.columns else 'revenue'
            orders_col = 'unique_orders' if 'unique_orders' in regional_data.columns else 'orders'
            aov_col = 'avg_order_value' if 'avg_order_value' in regional_data.columns else 'aov'
            
            # Check if required columns exist
            if revenue_col not in regional_data.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text="Revenue data not available for regions",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Revenue bar chart
            fig.add_trace(
                go.Bar(
                    x=regional_data['region'],
                    y=regional_data[revenue_col],
                    name='Revenue',
                    marker_color='#1f77b4',
                    text=[f'${x:,.0f}' for x in regional_data[revenue_col]],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Market share pie chart (if available)
            if 'market_share_pct' in regional_data.columns:
                fig.add_trace(
                    go.Pie(
                        labels=regional_data['region'],
                        values=regional_data['market_share_pct'],
                        name='Market Share',
                        hovertemplate='<b>%{label}</b><br>Share: %{value:.1f}%<extra></extra>',
                        textinfo='label+percent'
                    ),
                    row=1, col=2
                )
            else:
                # Create market share from revenue data
                total_revenue = regional_data[revenue_col].sum()
                market_share = (regional_data[revenue_col] / total_revenue * 100)
                fig.add_trace(
                    go.Pie(
                        labels=regional_data['region'],
                        values=market_share,
                        name='Market Share',
                        hovertemplate='<b>%{label}</b><br>Share: %{value:.1f}%<extra></extra>',
                        textinfo='label+percent'
                    ),
                    row=1, col=2
                )
            
            # Average order value (if available)
            if aov_col in regional_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=regional_data['region'],
                        y=regional_data[aov_col],
                        name='AOV',
                        marker_color='#ff7f0e',
                        text=[f'${x:,.0f}' for x in regional_data[aov_col]],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>AOV: $%{y:,.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Order count (if available)
            if orders_col in regional_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=regional_data['region'],
                        y=regional_data[orders_col],
                        name='Orders',
                        marker_color='#2ca02c',
                        text=[f'{x:,}' for x in regional_data[orders_col]],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Orders: %{y:,}<extra></extra>'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="Regional Sales Performance",
                    x=0.5,
                    font=dict(size=20)
                ),
                height=700,
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(title_text="Region", row=1, col=1)
            fig.update_xaxes(title_text="Region", row=2, col=1)
            fig.update_xaxes(title_text="Region", row=2, col=2)
            fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
            fig.update_yaxes(title_text="AOV ($)", row=2, col=1)
            fig.update_yaxes(title_text="Orders", row=2, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating regional sales chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading regional sales: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_category_heatmap(self, category_data: pd.DataFrame, color_scale: str = "RdYlBu_r") -> go.Figure:
        """
        Create category discount heatmap visualization.
        
        Args:
            category_data: DataFrame with category discount data
            color_scale: Color scale for the heatmap
            
        Returns:
            Plotly figure with category discount heatmap
            
        Requirements: 4.5, 4.7
        """
        try:
            if category_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available for category discounts",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Prepare data for heatmap
            # Create a matrix with categories and discount metrics
            metrics = ['avg_discount', 'discount_penetration_pct', 'total_revenue', 'discount_effectiveness_score']
            
            # Normalize metrics for better heatmap visualization
            heatmap_data = category_data[metrics].copy()
            for col in metrics:
                if col in heatmap_data.columns:
                    # Normalize to 0-1 scale
                    min_val = heatmap_data[col].min()
                    max_val = heatmap_data[col].max()
                    if max_val > min_val:
                        heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
                    else:
                        heatmap_data[col] = 0.5  # Default middle value if all same
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=['Avg Discount', 'Discount Penetration %', 'Total Revenue', 'Effectiveness Score'],
                y=category_data['category'].tolist(),
                colorscale=color_scale,
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>%{x}: %{customdata}<br>Normalized: %{z:.2f}<extra></extra>',
                customdata=category_data[metrics].values
            ))
            
            fig.update_layout(
                title=dict(
                    text="Category Discount Effectiveness Heatmap",
                    x=0.5,
                    font=dict(size=20)
                ),
                xaxis_title="Metrics",
                yaxis_title="Product Category",
                height=max(400, len(category_data) * 30),  # Dynamic height based on categories
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating category heatmap: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading category heatmap: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def display_data_quality_section(self) -> None:
        """
        Display comprehensive data quality section with validation results.
        
        This section shows:
        - Overall data quality score
        - Validation rule results
        - Data quality trends
        - Validation performance metrics
        - Error analysis and recommendations
        """
        try:
            st.markdown('<div class="section-header">🛡️ Data Quality Dashboard</div>', 
                       unsafe_allow_html=True)
            
            # Load validation data
            validation_data = self.data_provider.load_validation_reports()
            
            if not validation_data:
                st.warning("No validation reports found. Run the ETL pipeline to generate data quality reports.")
                return
            
            # Display overall data quality metrics
            self._display_quality_overview(validation_data)
            
            # Display validation rules results
            self._display_validation_rules_results(validation_data)
            
            # Display validation performance metrics
            self._display_validation_performance(validation_data)
            
            # Display error analysis and recommendations
            self._display_error_analysis(validation_data)
            
        except Exception as e:
            logger.error(f"Error displaying data quality section: {str(e)}")
            st.error(f"Error loading data quality information: {str(e)}")
    
    def _display_quality_overview(self, validation_data: Dict[str, Any]) -> None:
        """Display overall data quality overview metrics."""
        st.subheader("📊 Data Quality Overview")
        
        # Extract key metrics
        overall_score = validation_data.get('overall_data_quality_score', 0)
        total_rules = validation_data.get('total_rules_executed', 0)
        successful_rules = validation_data.get('successful_rules', 0)
        failed_rules = validation_data.get('failed_rules', 0)
        total_records = validation_data.get('total_records_validated', 0)
        
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Color code the quality score
            if overall_score >= 95:
                delta_color = "normal"
                score_emoji = "🟢"
            elif overall_score >= 85:
                delta_color = "normal" 
                score_emoji = "🟡"
            else:
                delta_color = "inverse"
                score_emoji = "🔴"
                
            st.metric(
                label=f"{score_emoji} Data Quality Score",
                value=f"{overall_score:.1f}%",
                delta=None
            )
        
        with col2:
            success_rate = (successful_rules / max(total_rules, 1)) * 100
            st.metric(
                label="✅ Rules Passed",
                value=f"{successful_rules}/{total_rules}",
                delta=f"{success_rate:.1f}%"
            )
        
        with col3:
            if failed_rules > 0:
                st.metric(
                    label="❌ Rules Failed",
                    value=failed_rules,
                    delta=f"{(failed_rules/max(total_rules,1)*100):.1f}%",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    label="❌ Rules Failed",
                    value=0,
                    delta="Perfect!"
                )
        
        with col4:
            st.metric(
                label="📋 Records Validated",
                value=f"{total_records:,}"
            )
        
        with col5:
            avg_execution_time = validation_data.get('average_rule_time', 0) * 1000
            st.metric(
                label="⚡ Avg Rule Time",
                value=f"{avg_execution_time:.1f}ms"
            )
        
        # Display quality status message
        if overall_score >= 95:
            st.success("🎉 Excellent data quality! Your data meets high quality standards.")
        elif overall_score >= 85:
            st.info("👍 Good data quality. Minor issues detected that may need attention.")
        elif overall_score >= 70:
            st.warning("⚠️ Moderate data quality. Several issues detected that should be addressed.")
        else:
            st.error("🚨 Poor data quality. Significant issues detected that require immediate attention.")
    
    def _display_validation_rules_results(self, validation_data: Dict[str, Any]) -> None:
        """Display detailed validation rules results."""
        st.subheader("📋 Validation Rules Results")
        
        validation_history = validation_data.get('validation_history', [])
        
        if not validation_history:
            st.warning("No validation rule results available.")
            return
        
        # Create DataFrame from validation history
        rules_df = pd.DataFrame(validation_history)
        
        # Display rules summary table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Validation Rules Summary**")
            
            # Prepare display data
            display_df = rules_df[['rule_name', 'field_name', 'severity', 'passed', 'error_rate', 'total_records', 'valid_records', 'invalid_records']].copy()
            
            # Add status emoji
            display_df['status'] = display_df.apply(lambda row: 
                "✅ PASS" if row['passed'] else 
                f"❌ FAIL ({row['error_rate']:.1f}%)", axis=1)
            
            # Add severity emoji
            severity_emoji = {
                'critical': '🔴',
                'error': '🟠', 
                'warning': '🟡',
                'info': '🔵'
            }
            display_df['severity_icon'] = display_df['severity'].map(severity_emoji)
            
            # Reorder columns for display
            display_columns = ['rule_name', 'field_name', 'severity_icon', 'status', 'valid_records', 'invalid_records']
            display_df = display_df[display_columns]
            display_df.columns = ['Rule Name', 'Field', 'Severity', 'Status', 'Valid Records', 'Invalid Records']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Rules by severity pie chart
            severity_counts = rules_df['severity'].value_counts()
            
            fig_severity = go.Figure(data=[go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                hole=0.4,
                marker_colors=['#ff4444', '#ff8800', '#ffdd00', '#4444ff']
            )])
            
            fig_severity.update_layout(
                title="Rules by Severity",
                height=300,
                showlegend=True,
                margin=dict(t=40, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Display failed rules details
        failed_rules = rules_df[rules_df['passed'] == False]
        
        if not failed_rules.empty:
            st.write("**⚠️ Failed Rules Details**")
            
            for _, rule in failed_rules.iterrows():
                with st.expander(f"❌ {rule['rule_name']} - {rule['error_rate']:.1f}% error rate"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Field:** {rule['field_name']}")
                        st.write(f"**Severity:** {rule['severity'].title()}")
                        st.write(f"**Total Records:** {rule['total_records']:,}")
                        st.write(f"**Invalid Records:** {rule['invalid_records']:,}")
                    
                    with col2:
                        st.write(f"**Error Rate:** {rule['error_rate']:.2f}%")
                        st.write(f"**Rule Type:** {rule['rule_type'].title()}")
                        
                        # Show sample invalid values if available
                        if rule.get('sample_invalid_values'):
                            st.write("**Sample Invalid Values:**")
                            for value in rule['sample_invalid_values'][:5]:
                                st.code(str(value))
    
    def _display_validation_performance(self, validation_data: Dict[str, Any]) -> None:
        """Display validation performance metrics."""
        st.subheader("⚡ Validation Performance")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Execution Performance**")
            
            total_duration = validation_data.get('total_execution_time', 0)
            total_rules = validation_data.get('total_rules_executed', 0)
            avg_rule_time = validation_data.get('average_rule_time', 0)
            memory_usage = validation_data.get('memory_usage_mb', 0)
            
            perf_metrics = {
                'Total Execution Time': f"{total_duration:.3f}s",
                'Rules Executed': f"{total_rules}",
                'Average Rule Time': f"{avg_rule_time*1000:.2f}ms",
                'Memory Usage': f"{memory_usage:.1f}MB"
            }
            
            for metric, value in perf_metrics.items():
                st.metric(metric, value)
        
        with col2:
            # Rule performance breakdown
            rule_performance = validation_data.get('rule_performance', {})
            
            if rule_performance:
                st.write("**Rule Performance (Top 5 Slowest)**")
                
                # Sort by execution time
                sorted_rules = sorted(rule_performance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                rule_names = [rule[0] for rule in sorted_rules]
                rule_times = [rule[1] * 1000 for rule in sorted_rules]  # Convert to ms
                
                fig_performance = go.Figure(data=[
                    go.Bar(
                        x=rule_times,
                        y=rule_names,
                        orientation='h',
                        marker_color='#1f77b4',
                        text=[f'{time:.2f}ms' for time in rule_times],
                        textposition='outside'
                    )
                ])
                
                fig_performance.update_layout(
                    title="Rule Execution Time",
                    xaxis_title="Execution Time (ms)",
                    height=300,
                    margin=dict(t=40, b=40, l=120, r=40)
                )
                
                st.plotly_chart(fig_performance, use_container_width=True)
    
    def _display_error_analysis(self, validation_data: Dict[str, Any]) -> None:
        """Display error analysis and recommendations."""
        st.subheader("🔍 Error Analysis & Recommendations")
        
        # Error distribution
        error_distribution = validation_data.get('error_distribution', {})
        
        if error_distribution:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Error Distribution**")
                
                # Create error distribution chart
                error_types = list(error_distribution.keys())
                error_counts = list(error_distribution.values())
                
                fig_errors = go.Figure(data=[
                    go.Bar(
                        x=error_types,
                        y=error_counts,
                        marker_color=['#ff4444', '#ff8800', '#ffdd00', '#4444ff'][:len(error_types)],
                        text=error_counts,
                        textposition='outside'
                    )
                ])
                
                fig_errors.update_layout(
                    title="Error Types Distribution",
                    xaxis_title="Error Type",
                    yaxis_title="Count",
                    height=300
                )
                
                st.plotly_chart(fig_errors, use_container_width=True)
            
            with col2:
                st.write("**Recommendations**")
                
                # Generate recommendations based on error types
                recommendations = []
                
                if 'enum_warning' in error_distribution:
                    recommendations.append("🔧 **Category Values**: Update allowed category values to include 'Home Goods'")
                
                if 'range_error' in error_distribution:
                    recommendations.append("📊 **Range Validation**: Review min/max value constraints for numeric fields")
                
                if 'format_warning' in error_distribution:
                    recommendations.append("📝 **Format Issues**: Implement data standardization for format validation")
                
                if 'not_null_critical' in error_distribution:
                    recommendations.append("⚠️ **Missing Data**: Address null values in critical fields")
                
                # Default recommendations
                if not recommendations:
                    recommendations = [
                        "✅ **Good Quality**: Data quality is within acceptable ranges",
                        "📈 **Monitor Trends**: Continue monitoring for emerging patterns",
                        "🔄 **Regular Validation**: Run validation regularly to maintain quality"
                    ]
                
                for rec in recommendations:
                    st.markdown(rec)
        
        # Data quality trends (if historical data available)
        st.write("**📈 Data Quality Insights**")
        
        validation_history = validation_data.get('validation_history', [])
        if validation_history:
            # Calculate insights
            total_records = sum(rule.get('total_records', 0) for rule in validation_history)
            total_invalid = sum(rule.get('invalid_records', 0) for rule in validation_history)
            overall_error_rate = (total_invalid / max(total_records, 1)) * 100
            
            insights = [
                f"📊 **Overall Error Rate**: {overall_error_rate:.2f}% across all validation rules",
                f"🔍 **Most Problematic Field**: {self._get_most_problematic_field(validation_history)}",
                f"⚡ **Validation Speed**: {total_records/max(validation_data.get('total_execution_time', 1), 0.001):,.0f} records/second",
                f"🎯 **Quality Status**: {'Excellent' if overall_error_rate < 5 else 'Good' if overall_error_rate < 15 else 'Needs Improvement'}"
            ]
            
            for insight in insights:
                st.markdown(insight)
    
    def _get_most_problematic_field(self, validation_history: List[Dict]) -> str:
        """Get the field with the highest error rate."""
        field_errors = {}
        
        for rule in validation_history:
            field = rule.get('field_name', 'unknown')
            error_rate = rule.get('error_rate', 0)
            
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(error_rate)
        
        # Calculate average error rate per field
        field_avg_errors = {field: sum(errors)/len(errors) for field, errors in field_errors.items()}
        
        if field_avg_errors:
            most_problematic = max(field_avg_errors, key=field_avg_errors.get)
            return f"{most_problematic} ({field_avg_errors[most_problematic]:.1f}% avg error rate)"
        
        return "No problematic fields detected"

    def display_anomaly_table(self, anomaly_data: pd.DataFrame) -> None:
        """
        Display anomaly records in a dedicated section.
        
        Args:
            anomaly_data: DataFrame with anomaly records
            
        Requirements: 4.4
        """
        try:
            st.markdown('<div class="section-header">🚨 Anomaly Detection Results</div>', 
                       unsafe_allow_html=True)
            
            if anomaly_data.empty:
                st.warning("No anomaly records found in the current dataset.")
                return
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Anomalies",
                    value=len(anomaly_data)
                )
            
            with col2:
                if 'anomaly_score' in anomaly_data.columns:
                    avg_score = anomaly_data['anomaly_score'].mean()
                    st.metric(
                        label="Avg Anomaly Score",
                        value=f"{avg_score:.2f}"
                    )
            
            with col3:
                if 'revenue' in anomaly_data.columns:
                    total_anomaly_revenue = anomaly_data['revenue'].sum()
                    st.metric(
                        label="Total Anomaly Revenue",
                        value=f"${total_anomaly_revenue:,.2f}"
                    )
            
            with col4:
                if 'anomaly_type' in anomaly_data.columns:
                    most_common_type = anomaly_data['anomaly_type'].mode().iloc[0] if len(anomaly_data) > 0 else "N/A"
                    st.metric(
                        label="Most Common Type",
                        value=most_common_type
                    )
            
            # Display detailed table
            st.subheader("Detailed Anomaly Records")
            
            # Format the dataframe for better display
            display_data = anomaly_data.copy()
            
            # Format revenue column if present
            if 'revenue' in display_data.columns:
                display_data['revenue'] = display_data['revenue'].apply(lambda x: f"${x:,.2f}")
            
            # Format anomaly score if present
            if 'anomaly_score' in display_data.columns:
                display_data['anomaly_score'] = display_data['anomaly_score'].apply(lambda x: f"{x:.3f}")
            
            # Display with pagination
            st.dataframe(
                display_data,
                use_container_width=True,
                height=400
            )
            
            # Add download button
            csv = anomaly_data.to_csv(index=False)
            st.download_button(
                label="Download Anomaly Data as CSV",
                data=csv,
                file_name=f"anomaly_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            logger.error(f"Error displaying anomaly table: {str(e)}")
            st.error(f"Error loading anomaly data: {str(e)}")
    
    def run(self):
        """
        Main method to run the Streamlit dashboard application.
        """
        try:
            # Header
            st.markdown('<h1 class="main-header">📊 E-commerce Analytics Dashboard</h1>', 
                       unsafe_allow_html=True)
            
            # Sidebar for controls
            st.sidebar.title("Dashboard Controls")
            
            # Data refresh button
            if st.sidebar.button("🔄 Refresh Data"):
                self.data_provider.refresh_data()
                st.sidebar.success("Data refreshed successfully!")
            
            # Load data
            with st.spinner("Loading dashboard data..."):
                analytical_tables = self.data_provider.load_analytical_tables()
                dashboard_metrics = self.data_provider.get_dashboard_metrics()
            
            # Interactive filtering controls
            st.sidebar.markdown("### 🔍 Filters")
            
            # Date range filter for monthly data
            date_filter_enabled = st.sidebar.checkbox("Enable Date Filtering", value=False)
            selected_months = None
            
            if date_filter_enabled and 'monthly_sales_summary' in analytical_tables:
                monthly_data = analytical_tables['monthly_sales_summary']
                if not monthly_data.empty and 'month' in monthly_data.columns:
                    available_months = sorted(monthly_data['month'].unique())
                    selected_months = st.sidebar.multiselect(
                        "Select Months",
                        options=available_months,
                        default=available_months[-6:] if len(available_months) > 6 else available_months
                    )
            
            # Region filter
            region_filter_enabled = st.sidebar.checkbox("Enable Region Filtering", value=False)
            selected_regions = None
            
            if region_filter_enabled and 'region_wise_performance' in analytical_tables:
                regional_data = analytical_tables['region_wise_performance']
                if not regional_data.empty and 'region' in regional_data.columns:
                    available_regions = regional_data['region'].unique().tolist()
                    selected_regions = st.sidebar.multiselect(
                        "Select Regions",
                        options=available_regions,
                        default=available_regions
                    )
            
            # Category filter
            category_filter_enabled = st.sidebar.checkbox("Enable Category Filtering", value=False)
            selected_categories = None
            
            if category_filter_enabled and 'category_discount_map' in analytical_tables:
                category_data = analytical_tables['category_discount_map']
                if not category_data.empty and 'category' in category_data.columns:
                    available_categories = category_data['category'].unique().tolist()
                    selected_categories = st.sidebar.multiselect(
                        "Select Categories",
                        options=available_categories,
                        default=available_categories
                    )
            
            # Apply filters to data
            filtered_tables = self._apply_filters(
                analytical_tables, 
                selected_months, 
                selected_regions, 
                selected_categories
            )
            
            # Display key metrics (recalculated based on filtered data)
            filtered_metrics = self._calculate_filtered_metrics(filtered_tables)
            
            if filtered_metrics:
                st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', 
                           unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Total Revenue",
                        value=f"${filtered_metrics.get('total_revenue', 0):,.2f}",
                        delta=filtered_metrics.get('revenue_growth', 0)
                    )
                
                with col2:
                    st.metric(
                        label="Total Orders",
                        value=f"{filtered_metrics.get('total_orders', 0):,}",
                        delta=filtered_metrics.get('order_growth', 0)
                    )
                
                with col3:
                    st.metric(
                        label="Average Order Value",
                        value=f"${filtered_metrics.get('avg_order_value', 0):,.2f}",
                        delta=filtered_metrics.get('aov_growth', 0)
                    )
                
                with col4:
                    st.metric(
                        label="Average Discount",
                        value=f"{filtered_metrics.get('avg_discount', 0)*100:.1f}%",
                        delta=filtered_metrics.get('discount_change', 0)
                    )
            
            # Revenue trends
            if 'monthly_sales_summary' in filtered_tables:
                st.markdown('<div class="section-header">📊 Revenue Trends</div>', 
                           unsafe_allow_html=True)
                
                # Add drill-down options
                col1, col2 = st.columns([3, 1])
                with col2:
                    chart_type = st.selectbox(
                        "Chart Type",
                        ["Line Chart", "Bar Chart", "Area Chart"],
                        key="revenue_chart_type"
                    )
                
                revenue_chart = self.create_revenue_trend_chart(
                    filtered_tables['monthly_sales_summary'], 
                    chart_type=chart_type
                )
                st.plotly_chart(revenue_chart, use_container_width=True)
            
            # Top products and regional sales in columns
            col1, col2 = st.columns(2)
            
            with col1:
                if 'top_products' in filtered_tables:
                    st.markdown('<div class="section-header">🏆 Top Products</div>', 
                               unsafe_allow_html=True)
                    
                    # Add product count selector
                    product_count = st.slider("Number of products to show", 5, 20, 10, key="product_count")
                    
                    products_chart = self.create_top_products_chart(
                        filtered_tables['top_products'].head(product_count)
                    )
                    st.plotly_chart(products_chart, use_container_width=True)
                    
                    # Add detailed product table
                    if st.checkbox("Show detailed product data", key="show_product_details"):
                        st.dataframe(
                            filtered_tables['top_products'].head(product_count),
                            use_container_width=True
                        )
            
            with col2:
                if 'region_wise_performance' in filtered_tables:
                    st.markdown('<div class="section-header">🌍 Regional Performance</div>', 
                               unsafe_allow_html=True)
                    regional_chart = self.create_regional_sales_chart(filtered_tables['region_wise_performance'])
                    st.plotly_chart(regional_chart, use_container_width=True)
                    
                    # Add regional comparison table
                    if st.checkbox("Show regional comparison", key="show_regional_details"):
                        st.dataframe(
                            filtered_tables['region_wise_performance'],
                            use_container_width=True
                        )
            
            # Category discount heatmap
            if 'category_discount_map' in filtered_tables:
                st.markdown('<div class="section-header">🎯 Category Discount Analysis</div>', 
                           unsafe_allow_html=True)
                
                # Add heatmap customization
                col1, col2 = st.columns([3, 1])
                with col2:
                    color_scale = st.selectbox(
                        "Color Scale",
                        ["RdYlBu_r", "Viridis", "Plasma", "Blues"],
                        key="heatmap_color"
                    )
                
                heatmap_chart = self.create_category_heatmap(
                    filtered_tables['category_discount_map'],
                    color_scale=color_scale
                )
                st.plotly_chart(heatmap_chart, use_container_width=True)
            
            # Anomaly detection results
            if 'anomaly_records' in filtered_tables:
                self.display_anomaly_table(filtered_tables['anomaly_records'])
            
            # Data Quality Dashboard Section
            self.display_data_quality_section()
            
            # Data export functionality
            st.markdown('<div class="section-header">📥 Data Export</div>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export Dashboard Data"):
                    if self.data_provider.export_dashboard_data():
                        st.success("Dashboard data exported successfully!")
                    else:
                        st.error("Failed to export dashboard data")
            
            with col2:
                # Export filtered data
                if filtered_tables:
                    export_data = {}
                    for table_name, df in filtered_tables.items():
                        if not df.empty:
                            export_data[table_name] = df.to_csv(index=False)
                    
                    if export_data:
                        st.download_button(
                            label="Download Filtered Data",
                            data=str(export_data),
                            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            
            with col3:
                # Show data freshness info
                if st.button("Check Data Freshness"):
                    freshness_info = self.data_provider.get_data_freshness()
                    st.json(freshness_info)
            
            # Performance monitoring
            st.sidebar.markdown("### 📊 Performance")
            
            # Show data freshness
            freshness_info = self.data_provider.get_data_freshness()
            if freshness_info['cache_valid']:
                st.sidebar.success("✅ Data cache is fresh")
            else:
                st.sidebar.warning("⚠️ Data cache may be stale")
            
            # Show available tables
            available_count = len(freshness_info['available_tables'])
            total_count = len(self.data_provider.table_files)
            st.sidebar.info(f"📋 {available_count}/{total_count} tables available")
            
            # Footer
            st.markdown("---")
            st.markdown(
                "**Dashboard Info:** Last updated: " + 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
                " | Data processed from e-commerce sales pipeline"
            )
            
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            st.error(f"Dashboard error: {str(e)}")
            st.info("Please check if the data pipeline has been run and analytical tables are available.")


def main():
    """Main function to run the dashboard."""
    dashboard = DashboardApp()
    dashboard.run()


if __name__ == "__main__":
    main()