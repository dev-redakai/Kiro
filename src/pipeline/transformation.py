"""
Analytical transformation engine for business intelligence.

This module provides the AnalyticalTransformer class that creates various
business intelligence tables from cleaned e-commerce data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalyticalTransformer:
    """
    Creates business intelligence tables from cleaned e-commerce data.
    
    This class generates various analytical tables including monthly sales summaries,
    top products, regional performance metrics, and category discount analysis.
    """
    
    def __init__(self):
        """Initialize the AnalyticalTransformer."""
        self.transformation_stats = {
            'tables_generated': 0,
            'total_records_processed': 0,
            'generation_errors': []
        }
    
    def create_monthly_sales_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate monthly sales summary with revenue, quantity, and average discount.
        
        Args:
            data: Cleaned DataFrame containing sales data
            
        Returns:
            DataFrame with monthly sales summary
            
        Requirements: 3.1
        """
        logger.info("Generating monthly sales summary")
        
        try:
            # Ensure required columns exist
            required_cols = ['sale_date', 'quantity', 'unit_price', 'discount_percent']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate revenue if not already present
            if 'revenue' not in data.columns:
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            # Convert sale_date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['sale_date']):
                data = data.copy()
                data['sale_date'] = pd.to_datetime(data['sale_date'], errors='coerce')
            
            # Remove rows with invalid dates
            data_clean = data.dropna(subset=['sale_date'])
            
            # Extract year-month for grouping
            data_clean = data_clean.copy()
            data_clean['year_month'] = data_clean['sale_date'].dt.to_period('M')
            
            # Group by month and calculate metrics
            monthly_summary = data_clean.groupby('year_month').agg({
                'revenue': ['sum', 'count'],
                'quantity': 'sum',
                'discount_percent': 'mean',
                'order_id': 'nunique'  # Unique orders per month
            }).round(2)
            
            # Flatten column names
            monthly_summary.columns = [
                'total_revenue', 'transaction_count', 'total_quantity', 
                'avg_discount', 'unique_orders'
            ]
            
            # Reset index to make year_month a column
            monthly_summary = monthly_summary.reset_index()
            monthly_summary['month'] = monthly_summary['year_month'].astype(str)
            monthly_summary = monthly_summary.drop('year_month', axis=1)
            
            # Calculate average order value
            monthly_summary['avg_order_value'] = (
                monthly_summary['total_revenue'] / monthly_summary['unique_orders']
            ).round(2)
            
            # Sort by month
            monthly_summary = monthly_summary.sort_values('month')
            
            logger.info(f"Generated monthly sales summary with {len(monthly_summary)} months")
            self.transformation_stats['tables_generated'] += 1
            
            return monthly_summary
            
        except Exception as e:
            error_msg = f"Error generating monthly sales summary: {str(e)}"
            logger.error(error_msg)
            self.transformation_stats['generation_errors'].append(error_msg)
            raise
    
    def create_top_products_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create top products table with top 10 by revenue and units.
        
        Args:
            data: Cleaned DataFrame containing sales data
            
        Returns:
            DataFrame with top products analysis
            
        Requirements: 3.2
        """
        logger.info("Generating top products table")
        
        try:
            # Ensure required columns exist
            required_cols = ['product_name', 'quantity', 'unit_price', 'discount_percent']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate revenue if not already present
            if 'revenue' not in data.columns:
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            # Group by product and calculate metrics
            product_metrics = data.groupby('product_name', observed=True).agg({
                'revenue': 'sum',
                'quantity': 'sum',
                'unit_price': 'mean',
                'discount_percent': 'mean',
                'order_id': 'nunique'  # Number of unique orders
            }).round(2)
            
            # Rename columns for clarity
            product_metrics.columns = [
                'total_revenue', 'total_units', 'avg_unit_price', 
                'avg_discount', 'order_count'
            ]
            
            # Add category information if available
            if 'category' in data.columns:
                category_info = data.groupby('product_name', observed=True)['category'].first()
                product_metrics['category'] = category_info
            
            # Reset index to make product_name a column
            product_metrics = product_metrics.reset_index()
            
            # Create rankings
            product_metrics['revenue_rank'] = product_metrics['total_revenue'].rank(
                method='dense', ascending=False
            ).astype(int)
            
            product_metrics['units_rank'] = product_metrics['total_units'].rank(
                method='dense', ascending=False
            ).astype(int)
            
            # Get top 10 by revenue
            top_by_revenue = product_metrics.nsmallest(10, 'revenue_rank')
            
            # Get top 10 by units (may overlap with revenue)
            top_by_units = product_metrics.nsmallest(10, 'units_rank')
            
            # Combine and remove duplicates, keeping the best rank
            top_products = pd.concat([top_by_revenue, top_by_units]).drop_duplicates(
                subset=['product_name']
            )
            
            # Add a flag to indicate why the product is in top 10
            top_products['top_reason'] = top_products.apply(
                lambda row: 'Revenue' if row['revenue_rank'] <= 10 
                else 'Units' if row['units_rank'] <= 10 
                else 'Both', axis=1
            )
            
            # Sort by revenue rank
            top_products = top_products.sort_values('revenue_rank')
            
            logger.info(f"Generated top products table with {len(top_products)} products")
            self.transformation_stats['tables_generated'] += 1
            
            return top_products
            
        except Exception as e:
            error_msg = f"Error generating top products table: {str(e)}"
            logger.error(error_msg)
            self.transformation_stats['generation_errors'].append(error_msg)
            raise
    
    def create_region_wise_performance_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build region-wise performance table with sales metrics by region.
        
        Args:
            data: Cleaned DataFrame containing sales data
            
        Returns:
            DataFrame with regional performance metrics
            
        Requirements: 3.3
        """
        logger.info("Generating region-wise performance table")
        
        try:
            # Ensure required columns exist
            required_cols = ['region', 'quantity', 'unit_price', 'discount_percent']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate revenue if not already present
            if 'revenue' not in data.columns:
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            # Group by region and calculate metrics
            regional_metrics = data.groupby('region', observed=True).agg({
                'revenue': ['sum', 'mean'],
                'quantity': 'sum',
                'unit_price': 'mean',
                'discount_percent': 'mean',
                'order_id': 'nunique'  # Unique orders per region
            }).round(2)
            
            # Flatten column names
            regional_metrics.columns = [
                'total_revenue', 'avg_revenue_per_transaction', 'total_quantity',
                'avg_unit_price', 'avg_discount', 'unique_orders'
            ]
            
            # Reset index to make region a column
            regional_metrics = regional_metrics.reset_index()
            
            # Calculate additional metrics
            regional_metrics['avg_order_value'] = (
                regional_metrics['total_revenue'] / regional_metrics['unique_orders']
            ).round(2)
            
            regional_metrics['avg_items_per_order'] = (
                regional_metrics['total_quantity'] / regional_metrics['unique_orders']
            ).round(2)
            
            # Calculate market share
            total_revenue = regional_metrics['total_revenue'].sum()
            regional_metrics['market_share_pct'] = (
                (regional_metrics['total_revenue'] / total_revenue) * 100
            ).round(2)
            
            # Find top category per region if category data is available
            if 'category' in data.columns:
                top_categories = []
                for region in regional_metrics['region']:
                    region_data = data[data['region'] == region]
                    top_category = region_data.groupby('category', observed=True)['revenue'].sum().idxmax()
                    top_categories.append(top_category)
                
                regional_metrics['top_category'] = top_categories
            
            # Sort by total revenue descending
            regional_metrics = regional_metrics.sort_values('total_revenue', ascending=False)
            
            # Add performance ranking
            regional_metrics['performance_rank'] = range(1, len(regional_metrics) + 1)
            
            logger.info(f"Generated regional performance table with {len(regional_metrics)} regions")
            self.transformation_stats['tables_generated'] += 1
            
            return regional_metrics
            
        except Exception as e:
            error_msg = f"Error generating regional performance table: {str(e)}"
            logger.error(error_msg)
            self.transformation_stats['generation_errors'].append(error_msg)
            raise
    
    def create_category_discount_map(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate category discount map with average discount by category.
        
        Args:
            data: Cleaned DataFrame containing sales data
            
        Returns:
            DataFrame with category discount analysis
            
        Requirements: 3.4
        """
        logger.info("Generating category discount map")
        
        try:
            # Ensure required columns exist
            required_cols = ['category', 'discount_percent', 'quantity', 'unit_price']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate revenue if not already present
            if 'revenue' not in data.columns:
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            # Group by category and calculate discount metrics
            category_discounts = data.groupby('category', observed=True).agg({
                'discount_percent': ['mean', 'median', 'std', 'min', 'max'],
                'revenue': 'sum',
                'quantity': 'sum',
                'order_id': 'nunique'
            }).round(4)
            
            # Flatten column names
            category_discounts.columns = [
                'avg_discount', 'median_discount', 'std_discount', 
                'min_discount', 'max_discount', 'total_revenue', 
                'total_quantity', 'order_count'
            ]
            
            # Reset index to make category a column
            category_discounts = category_discounts.reset_index()
            
            # Calculate discount effectiveness metrics
            # Higher revenue with higher discounts might indicate effective discounting
            category_discounts['discount_effectiveness_score'] = (
                category_discounts['total_revenue'] * category_discounts['avg_discount']
            ).round(2)
            
            # Calculate percentage of orders with discounts
            discount_orders = data[data['discount_percent'] > 0].groupby('category', observed=True)['order_id'].nunique()
            total_orders = data.groupby('category', observed=True)['order_id'].nunique()
            
            category_discounts['discount_penetration_pct'] = (
                (discount_orders / total_orders) * 100
            ).fillna(0).round(2)
            
            # Calculate average discount amount in currency
            category_discounts['avg_discount_amount'] = (
                category_discounts['total_revenue'] * category_discounts['avg_discount'] / 
                category_discounts['order_count']
            ).round(2)
            
            # Categorize discount strategy
            def categorize_discount_strategy(row):
                if row['avg_discount'] < 0.05:
                    return 'Low Discount'
                elif row['avg_discount'] < 0.15:
                    return 'Moderate Discount'
                elif row['avg_discount'] < 0.30:
                    return 'High Discount'
                else:
                    return 'Very High Discount'
            
            category_discounts['discount_strategy'] = category_discounts.apply(
                categorize_discount_strategy, axis=1
            )
            
            # Sort by average discount descending
            category_discounts = category_discounts.sort_values('avg_discount', ascending=False)
            
            logger.info(f"Generated category discount map with {len(category_discounts)} categories")
            self.transformation_stats['tables_generated'] += 1
            
            return category_discounts
            
        except Exception as e:
            error_msg = f"Error generating category discount map: {str(e)}"
            logger.error(error_msg)
            self.transformation_stats['generation_errors'].append(error_msg)
            raise
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """
        Get transformation statistics.
        
        Returns:
            Dictionary containing transformation statistics
        """
        return {
            'tables_generated': self.transformation_stats['tables_generated'],
            'total_records_processed': self.transformation_stats['total_records_processed'],
            'generation_errors': self.transformation_stats['generation_errors'],
            'success_rate': (
                (self.transformation_stats['tables_generated'] - len(self.transformation_stats['generation_errors'])) /
                max(self.transformation_stats['tables_generated'], 1)
            )
        }
    
    def reset_stats(self) -> None:
        """Reset transformation statistics."""
        self.transformation_stats = {
            'tables_generated': 0,
            'total_records_processed': 0,
            'generation_errors': []
        }


class MetricsCalculator:
    """
    Calculates business metrics and advanced analytics.
    
    This class provides methods to calculate revenue metrics, discount effectiveness,
    regional performance, and handles edge cases like division by zero and null values.
    """
    
    def __init__(self):
        """Initialize the MetricsCalculator."""
        self.calculation_stats = {
            'metrics_calculated': 0,
            'calculation_errors': [],
            'null_value_handling': 0,
            'division_by_zero_handling': 0
        }
    
    def calculate_revenue_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Create revenue metrics calculation functions.
        
        Args:
            data: DataFrame containing sales data
            
        Returns:
            Dictionary containing various revenue metrics
            
        Requirements: 3.6
        """
        logger.info("Calculating revenue metrics")
        
        try:
            # Ensure required columns exist
            required_cols = ['quantity', 'unit_price', 'discount_percent']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate revenue if not already present
            if 'revenue' not in data.columns:
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            # Handle null values
            clean_data = data.dropna(subset=['revenue', 'quantity', 'unit_price'])
            if len(clean_data) < len(data):
                null_count = len(data) - len(clean_data)
                self.calculation_stats['null_value_handling'] += null_count
                logger.warning(f"Handled {null_count} null values in revenue calculations")
            
            # Basic revenue metrics
            total_revenue = float(clean_data['revenue'].sum())
            avg_revenue_per_transaction = float(clean_data['revenue'].mean())
            median_revenue_per_transaction = float(clean_data['revenue'].median())
            revenue_std = float(clean_data['revenue'].std())
            
            # Revenue distribution metrics
            revenue_percentiles = clean_data['revenue'].quantile([0.25, 0.75, 0.90, 0.95, 0.99])
            
            # Growth metrics (if date information is available)
            growth_metrics = {}
            if 'sale_date' in clean_data.columns:
                try:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(clean_data['sale_date']):
                        clean_data = clean_data.copy()
                        clean_data['sale_date'] = pd.to_datetime(clean_data['sale_date'], errors='coerce')
                    
                    # Calculate monthly revenue for growth analysis
                    clean_data = clean_data.dropna(subset=['sale_date'])
                    monthly_revenue = clean_data.groupby(
                        clean_data['sale_date'].dt.to_period('M')
                    )['revenue'].sum()
                    
                    if len(monthly_revenue) > 1:
                        # Calculate month-over-month growth
                        monthly_growth = monthly_revenue.pct_change().dropna()
                        growth_metrics = {
                            'avg_monthly_growth_rate': float(monthly_growth.mean()),
                            'monthly_growth_volatility': float(monthly_growth.std()),
                            'total_growth_rate': float(
                                (monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / 
                                monthly_revenue.iloc[0] if monthly_revenue.iloc[0] != 0 else 0
                            )
                        }
                except Exception as e:
                    logger.warning(f"Could not calculate growth metrics: {str(e)}")
                    growth_metrics = {}
            
            # Customer value metrics (if customer data is available)
            customer_metrics = {}
            if 'customer_email' in clean_data.columns:
                try:
                    customer_revenue = clean_data.groupby('customer_email')['revenue'].sum()
                    customer_metrics = {
                        'avg_customer_lifetime_value': float(customer_revenue.mean()),
                        'median_customer_lifetime_value': float(customer_revenue.median()),
                        'top_10_percent_customer_contribution': float(
                            customer_revenue.nlargest(int(len(customer_revenue) * 0.1)).sum() / 
                            total_revenue if total_revenue > 0 else 0
                        )
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate customer metrics: {str(e)}")
                    customer_metrics = {}
            
            metrics = {
                # Basic metrics
                'total_revenue': round(total_revenue, 2),
                'avg_revenue_per_transaction': round(avg_revenue_per_transaction, 2),
                'median_revenue_per_transaction': round(median_revenue_per_transaction, 2),
                'revenue_standard_deviation': round(revenue_std, 2),
                
                # Distribution metrics
                'revenue_25th_percentile': round(float(revenue_percentiles[0.25]), 2),
                'revenue_75th_percentile': round(float(revenue_percentiles[0.75]), 2),
                'revenue_90th_percentile': round(float(revenue_percentiles[0.90]), 2),
                'revenue_95th_percentile': round(float(revenue_percentiles[0.95]), 2),
                'revenue_99th_percentile': round(float(revenue_percentiles[0.99]), 2),
                
                # Concentration metrics
                'revenue_concentration_ratio': round(
                    clean_data['revenue'].nlargest(int(len(clean_data) * 0.05)).sum() / 
                    total_revenue if total_revenue > 0 else 0, 4
                ),
                
                # Transaction metrics
                'total_transactions': len(clean_data),
                'avg_transaction_size': round(float(clean_data['quantity'].mean()), 2),
                'avg_unit_price': round(float(clean_data['unit_price'].mean()), 2)
            }
            
            # Add growth metrics if available
            metrics.update(growth_metrics)
            
            # Add customer metrics if available
            metrics.update(customer_metrics)
            
            self.calculation_stats['metrics_calculated'] += 1
            logger.info("Successfully calculated revenue metrics")
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error calculating revenue metrics: {str(e)}"
            logger.error(error_msg)
            self.calculation_stats['calculation_errors'].append(error_msg)
            raise  
  
    def calculate_discount_effectiveness_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Implement discount effectiveness analysis.
        
        Args:
            data: DataFrame containing sales data
            
        Returns:
            Dictionary containing discount effectiveness metrics
            
        Requirements: 3.6
        """
        logger.info("Calculating discount effectiveness analysis")
        
        try:
            # Ensure required columns exist
            required_cols = ['discount_percent', 'quantity', 'unit_price']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate revenue if not already present
            if 'revenue' not in data.columns:
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            # Handle null values
            clean_data = data.dropna(subset=['discount_percent', 'revenue', 'quantity'])
            if len(clean_data) < len(data):
                null_count = len(data) - len(clean_data)
                self.calculation_stats['null_value_handling'] += null_count
                logger.warning(f"Handled {null_count} null values in discount analysis")
            
            # Separate discounted and non-discounted transactions
            discounted_data = clean_data[clean_data['discount_percent'] > 0]
            non_discounted_data = clean_data[clean_data['discount_percent'] == 0]
            
            # Basic discount metrics
            total_discount_amount = float(
                (clean_data['quantity'] * clean_data['unit_price'] * clean_data['discount_percent']).sum()
            )
            avg_discount_rate = float(clean_data['discount_percent'].mean())
            discount_penetration = float(len(discounted_data) / len(clean_data) if len(clean_data) > 0 else 0)
            
            # Effectiveness metrics
            effectiveness_metrics = {}
            
            if len(discounted_data) > 0 and len(non_discounted_data) > 0:
                # Compare average transaction values
                avg_discounted_revenue = float(discounted_data['revenue'].mean())
                avg_non_discounted_revenue = float(non_discounted_data['revenue'].mean())
                
                # Compare quantities
                avg_discounted_quantity = float(discounted_data['quantity'].mean())
                avg_non_discounted_quantity = float(non_discounted_data['quantity'].mean())
                
                effectiveness_metrics = {
                    'avg_discounted_transaction_value': round(avg_discounted_revenue, 2),
                    'avg_non_discounted_transaction_value': round(avg_non_discounted_revenue, 2),
                    'revenue_impact_ratio': round(
                        avg_discounted_revenue / avg_non_discounted_revenue 
                        if avg_non_discounted_revenue > 0 else 0, 4
                    ),
                    'avg_discounted_quantity': round(avg_discounted_quantity, 2),
                    'avg_non_discounted_quantity': round(avg_non_discounted_quantity, 2),
                    'quantity_impact_ratio': round(
                        avg_discounted_quantity / avg_non_discounted_quantity 
                        if avg_non_discounted_quantity > 0 else 0, 4
                    )
                }
            else:
                # Handle edge case where all or no transactions have discounts
                if len(discounted_data) == 0:
                    logger.warning("No discounted transactions found")
                    self.calculation_stats['division_by_zero_handling'] += 1
                else:
                    logger.warning("No non-discounted transactions found")
                    self.calculation_stats['division_by_zero_handling'] += 1
                
                effectiveness_metrics = {
                    'avg_discounted_transaction_value': round(float(discounted_data['revenue'].mean()), 2) if len(discounted_data) > 0 else 0,
                    'avg_non_discounted_transaction_value': round(float(non_discounted_data['revenue'].mean()), 2) if len(non_discounted_data) > 0 else 0,
                    'revenue_impact_ratio': 0,
                    'avg_discounted_quantity': round(float(discounted_data['quantity'].mean()), 2) if len(discounted_data) > 0 else 0,
                    'avg_non_discounted_quantity': round(float(non_discounted_data['quantity'].mean()), 2) if len(non_discounted_data) > 0 else 0,
                    'quantity_impact_ratio': 0
                }
            
            # Discount tier analysis
            discount_tiers = {
                'low_discount': clean_data[(clean_data['discount_percent'] > 0) & (clean_data['discount_percent'] <= 0.1)],
                'medium_discount': clean_data[(clean_data['discount_percent'] > 0.1) & (clean_data['discount_percent'] <= 0.3)],
                'high_discount': clean_data[clean_data['discount_percent'] > 0.3]
            }
            
            tier_analysis = {}
            for tier_name, tier_data in discount_tiers.items():
                if len(tier_data) > 0:
                    tier_analysis[f'{tier_name}_avg_revenue'] = round(float(tier_data['revenue'].mean()), 2)
                    tier_analysis[f'{tier_name}_avg_quantity'] = round(float(tier_data['quantity'].mean()), 2)
                    tier_analysis[f'{tier_name}_transaction_count'] = len(tier_data)
                else:
                    tier_analysis[f'{tier_name}_avg_revenue'] = 0
                    tier_analysis[f'{tier_name}_avg_quantity'] = 0
                    tier_analysis[f'{tier_name}_transaction_count'] = 0
            
            # Category-wise discount effectiveness (if category data is available)
            category_effectiveness = {}
            if 'category' in clean_data.columns:
                try:
                    for category in clean_data['category'].unique():
                        cat_data = clean_data[clean_data['category'] == category]
                        cat_discounted = cat_data[cat_data['discount_percent'] > 0]
                        cat_non_discounted = cat_data[cat_data['discount_percent'] == 0]
                        
                        if len(cat_discounted) > 0 and len(cat_non_discounted) > 0:
                            category_effectiveness[f'{category}_effectiveness_ratio'] = round(
                                float(cat_discounted['revenue'].mean()) / 
                                float(cat_non_discounted['revenue'].mean())
                                if cat_non_discounted['revenue'].mean() > 0 else 0, 4
                            )
                        else:
                            category_effectiveness[f'{category}_effectiveness_ratio'] = 0
                            self.calculation_stats['division_by_zero_handling'] += 1
                            
                except Exception as e:
                    logger.warning(f"Could not calculate category effectiveness: {str(e)}")
            
            analysis = {
                # Basic discount metrics
                'total_discount_amount': round(total_discount_amount, 2),
                'avg_discount_rate': round(avg_discount_rate, 4),
                'discount_penetration_rate': round(discount_penetration, 4),
                'total_discounted_transactions': len(discounted_data),
                'total_non_discounted_transactions': len(non_discounted_data),
                
                # Effectiveness metrics
                **effectiveness_metrics,
                
                # Tier analysis
                **tier_analysis,
                
                # Category effectiveness
                **category_effectiveness,
                
                # ROI metrics
                'discount_roi': round(
                    (effectiveness_metrics.get('revenue_impact_ratio', 0) - 1) * 100, 2
                ) if effectiveness_metrics.get('revenue_impact_ratio', 0) > 0 else 0
            }
            
            self.calculation_stats['metrics_calculated'] += 1
            logger.info("Successfully calculated discount effectiveness analysis")
            
            return analysis
            
        except Exception as e:
            error_msg = f"Error calculating discount effectiveness: {str(e)}"
            logger.error(error_msg)
            self.calculation_stats['calculation_errors'].append(error_msg)
            raise
    
    def calculate_regional_performance_calculation_logic(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Build regional performance calculation logic.
        
        Args:
            data: DataFrame containing sales data
            
        Returns:
            Dictionary containing regional performance metrics by region
            
        Requirements: 3.7
        """
        logger.info("Calculating regional performance metrics")
        
        try:
            # Ensure required columns exist
            required_cols = ['region', 'quantity', 'unit_price', 'discount_percent']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate revenue if not already present
            if 'revenue' not in data.columns:
                data = data.copy()
                data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            
            # Handle null values
            clean_data = data.dropna(subset=['region', 'revenue'])
            if len(clean_data) < len(data):
                null_count = len(data) - len(clean_data)
                self.calculation_stats['null_value_handling'] += null_count
                logger.warning(f"Handled {null_count} null values in regional analysis")
            
            regional_performance = {}
            
            for region in clean_data['region'].unique():
                region_data = clean_data[clean_data['region'] == region]
                
                # Basic metrics
                total_revenue = float(region_data['revenue'].sum())
                total_quantity = float(region_data['quantity'].sum())
                total_orders = len(region_data)
                unique_customers = region_data['customer_email'].nunique() if 'customer_email' in region_data.columns else 0
                
                # Performance metrics
                avg_order_value = float(total_revenue / total_orders if total_orders > 0 else 0)
                avg_items_per_order = float(total_quantity / total_orders if total_orders > 0 else 0)
                avg_unit_price = float(region_data['unit_price'].mean())
                avg_discount = float(region_data['discount_percent'].mean())
                
                # Handle division by zero
                if total_orders == 0:
                    self.calculation_stats['division_by_zero_handling'] += 1
                    logger.warning(f"No orders found for region {region}")
                
                # Customer metrics (if available)
                customer_metrics = {}
                if unique_customers > 0:
                    avg_customer_value = float(total_revenue / unique_customers)
                    avg_orders_per_customer = float(total_orders / unique_customers)
                    customer_metrics = {
                        'avg_customer_lifetime_value': round(avg_customer_value, 2),
                        'avg_orders_per_customer': round(avg_orders_per_customer, 2),
                        'unique_customers': unique_customers
                    }
                
                # Time-based metrics (if date data is available)
                time_metrics = {}
                if 'sale_date' in region_data.columns:
                    try:
                        # Convert to datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(region_data['sale_date']):
                            region_data = region_data.copy()
                            region_data['sale_date'] = pd.to_datetime(region_data['sale_date'], errors='coerce')
                        
                        region_data_clean = region_data.dropna(subset=['sale_date'])
                        if len(region_data_clean) > 0:
                            # Calculate monthly performance
                            monthly_revenue = region_data_clean.groupby(
                                region_data_clean['sale_date'].dt.to_period('M')
                            )['revenue'].sum()
                            
                            if len(monthly_revenue) > 1:
                                monthly_growth = monthly_revenue.pct_change().dropna()
                                time_metrics = {
                                    'avg_monthly_revenue': round(float(monthly_revenue.mean()), 2),
                                    'monthly_revenue_volatility': round(float(monthly_revenue.std()), 2),
                                    'avg_monthly_growth_rate': round(float(monthly_growth.mean()), 4)
                                }
                    except Exception as e:
                        logger.warning(f"Could not calculate time metrics for region {region}: {str(e)}")
                
                # Category performance (if available)
                category_metrics = {}
                if 'category' in region_data.columns:
                    try:
                        category_revenue = region_data.groupby('category')['revenue'].sum()
                        top_category = category_revenue.idxmax()
                        top_category_revenue = float(category_revenue.max())
                        category_diversity = len(category_revenue)
                        
                        category_metrics = {
                            'top_category': top_category,
                            'top_category_revenue': round(top_category_revenue, 2),
                            'category_diversity_count': category_diversity,
                            'top_category_share': round(
                                top_category_revenue / total_revenue if total_revenue > 0 else 0, 4
                            )
                        }
                    except Exception as e:
                        logger.warning(f"Could not calculate category metrics for region {region}: {str(e)}")
                
                # Compile all metrics for this region
                regional_performance[region] = {
                    # Basic metrics
                    'total_revenue': round(total_revenue, 2),
                    'total_quantity': round(total_quantity, 2),
                    'total_orders': total_orders,
                    'avg_order_value': round(avg_order_value, 2),
                    'avg_items_per_order': round(avg_items_per_order, 2),
                    'avg_unit_price': round(avg_unit_price, 2),
                    'avg_discount_rate': round(avg_discount, 4),
                    
                    # Performance indicators
                    'revenue_per_order': round(avg_order_value, 2),
                    'efficiency_score': round(
                        (total_revenue / total_orders) * (1 - avg_discount) 
                        if total_orders > 0 else 0, 2
                    ),
                    
                    # Add customer metrics
                    **customer_metrics,
                    
                    # Add time metrics
                    **time_metrics,
                    
                    # Add category metrics
                    **category_metrics
                }
            
            # Calculate relative performance metrics
            total_market_revenue = sum(metrics['total_revenue'] for metrics in regional_performance.values())
            
            for region, metrics in regional_performance.items():
                if total_market_revenue > 0:
                    metrics['market_share'] = round(
                        metrics['total_revenue'] / total_market_revenue, 4
                    )
                    metrics['performance_index'] = round(
                        (metrics['avg_order_value'] / 
                         sum(m['avg_order_value'] for m in regional_performance.values()) * 
                         len(regional_performance)) if len(regional_performance) > 0 else 0, 4
                    )
                else:
                    metrics['market_share'] = 0
                    metrics['performance_index'] = 0
                    self.calculation_stats['division_by_zero_handling'] += 1
            
            self.calculation_stats['metrics_calculated'] += 1
            logger.info(f"Successfully calculated regional performance for {len(regional_performance)} regions")
            
            return regional_performance
            
        except Exception as e:
            error_msg = f"Error calculating regional performance: {str(e)}"
            logger.error(error_msg)
            self.calculation_stats['calculation_errors'].append(error_msg)
            raise
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """
        Get calculation statistics including edge case handling.
        
        Returns:
            Dictionary containing calculation statistics
        """
        return {
            'metrics_calculated': self.calculation_stats['metrics_calculated'],
            'calculation_errors': self.calculation_stats['calculation_errors'],
            'null_values_handled': self.calculation_stats['null_value_handling'],
            'division_by_zero_cases_handled': self.calculation_stats['division_by_zero_handling'],
            'success_rate': (
                (self.calculation_stats['metrics_calculated'] - len(self.calculation_stats['calculation_errors'])) /
                max(self.calculation_stats['metrics_calculated'], 1)
            )
        }
    
    def reset_stats(self) -> None:
        """Reset calculation statistics."""
        self.calculation_stats = {
            'metrics_calculated': 0,
            'calculation_errors': [],
            'null_value_handling': 0,
            'division_by_zero_handling': 0
        }