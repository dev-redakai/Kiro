#!/usr/bin/env python3
"""
Generate Dashboard Data from ETL Pipeline Output

This script creates analytical tables for the dashboard using the actual
processed data from the ETL pipeline, ensuring the dashboard shows real
insights from the data processing results.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_processed_data(data_dir: str = "data/output") -> str:
    """
    Find the most recent processed data file from ETL pipeline.
    
    Args:
        data_dir: Directory containing ETL output
        
    Returns:
        Path to the latest processed data file
    """
    csv_dir = os.path.join(data_dir, 'csv')
    parquet_dir = os.path.join(data_dir, 'parquet')
    
    # Look for processed data files
    csv_pattern = os.path.join(csv_dir, f'*_20250811_*.csv')
    parquet_pattern = os.path.join(parquet_dir, '*_20250811_*.parquet')
    
    # Find all processed files
    csv_files = glob.glob(csv_pattern)
    parquet_files = glob.glob(parquet_pattern)
    
    all_files = csv_files # + parquet_files #+ csv_files
    
    if not all_files:
        raise FileNotFoundError("No processed data files found from ETL pipeline")
    
    # Sort by modification time, most recent first
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = all_files[0]
    
    logger.info(f"Found latest processed data: {latest_file}")
    return latest_file


def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    Load processed data from file.
    
    Args:
        file_path: Path to the processed data file
        
    Returns:
        DataFrame with processed data
    """
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def ensure_revenue_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure revenue column exists in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with revenue column
    """
    if 'revenue' not in df.columns:
        if all(col in df.columns for col in ['quantity', 'unit_price', 'discount_percent']):
            df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
            logger.info("Calculated revenue column from quantity, unit_price, and discount_percent")
        else:
            logger.warning("Cannot calculate revenue - missing required columns")
            # Create a dummy revenue column
            df['revenue'] = df.get('unit_price', 100) * df.get('quantity', 1)
    
    return df


def generate_monthly_sales_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate monthly sales summary from processed data.
    
    Args:
        df: Processed data DataFrame
        
    Returns:
        Monthly sales summary DataFrame
    """
    logger.info("Generating monthly sales summary...")
    
    try:
        # Ensure sale_date is datetime
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        
        # Group by month
        monthly_sales = df.groupby(
            df['sale_date'].dt.to_period('M')
        ).agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'discount_percent': 'mean',
            'order_id': 'nunique'
        }).reset_index()
        
        # Rename columns
        monthly_sales.columns = ['month', 'total_revenue', 'total_quantity', 'avg_discount', 'unique_orders']
        monthly_sales['month'] = monthly_sales['month'].astype(str)
        monthly_sales['avg_order_value'] = monthly_sales['total_revenue'] / monthly_sales['unique_orders']
        
        logger.info(f"Generated monthly sales summary: {len(monthly_sales)} months")
        return monthly_sales
        
    except Exception as e:
        logger.error(f"Error generating monthly sales summary: {e}")
        return pd.DataFrame()


def generate_top_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate top products analysis from processed data.
    
    Args:
        df: Processed data DataFrame
        
    Returns:
        Top products DataFrame
    """
    logger.info("Generating top products analysis...")
    
    try:
        # Group by product
        top_products = df.groupby('product_name').agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'unit_price': 'mean',
            'discount_percent': 'mean',
            'order_id': 'nunique'
        }).reset_index()
        
        # Rename columns
        top_products.columns = ['product_name', 'total_revenue', 'total_units', 'avg_unit_price', 'avg_discount', 'order_count']
        
        # Sort by revenue and add ranks
        top_products = top_products.sort_values('total_revenue', ascending=False)
        top_products['revenue_rank'] = range(1, len(top_products) + 1)
        
        # Add units rank
        top_products_by_units = top_products.sort_values('total_units', ascending=False)
        units_rank_map = {name: rank for rank, name in enumerate(top_products_by_units['product_name'], 1)}
        top_products['units_rank'] = top_products['product_name'].map(units_rank_map)
        
        # Take top 50 products
        top_products = top_products.head(50)
        
        logger.info(f"Generated top products: {len(top_products)} products")
        return top_products
        
    except Exception as e:
        logger.error(f"Error generating top products: {e}")
        return pd.DataFrame()


def generate_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate regional performance analysis from processed data.
    
    Args:
        df: Processed data DataFrame
        
    Returns:
        Regional performance DataFrame
    """
    logger.info("Generating regional performance analysis...")
    
    try:
        # Group by region
        regional_perf = df.groupby('region').agg({
            'revenue': 'sum',
            'order_id': 'nunique',
            'quantity': 'sum',
            'discount_percent': 'mean'
        }).reset_index()
        
        # Rename columns
        regional_perf.columns = ['region', 'total_revenue', 'unique_orders', 'total_quantity', 'avg_discount']
        regional_perf['avg_order_value'] = regional_perf['total_revenue'] / regional_perf['unique_orders']
        
        # Calculate market share
        total_revenue = regional_perf['total_revenue'].sum()
        regional_perf['market_share_pct'] = (regional_perf['total_revenue'] / total_revenue * 100)
        
        # Add top category per region
        if 'category' in df.columns:
            region_category = df.groupby(['region', 'category'])['revenue'].sum().reset_index()
            top_category_per_region = region_category.loc[region_category.groupby('region')['revenue'].idxmax()]
            top_category_map = dict(zip(top_category_per_region['region'], top_category_per_region['category']))
            regional_perf['top_category'] = regional_perf['region'].map(top_category_map)
        
        logger.info(f"Generated regional performance: {len(regional_perf)} regions")
        return regional_perf
        
    except Exception as e:
        logger.error(f"Error generating regional performance: {e}")
        return pd.DataFrame()


def generate_category_discount_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate category discount analysis from processed data.
    
    Args:
        df: Processed data DataFrame
        
    Returns:
        Category discount map DataFrame
    """
    logger.info("Generating category discount analysis...")
    
    try:
        # Group by category
        category_discounts = df.groupby('category').agg({
            'discount_percent': ['mean', 'count'],
            'revenue': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        category_discounts.columns = ['category', 'avg_discount', 'discount_count', 'total_revenue', 'total_orders']
        
        # Calculate discount penetration (percentage of orders with discount > 0)
        discount_penetration = df[df['discount_percent'] > 0].groupby('category').size()
        total_orders_per_category = df.groupby('category').size()
        penetration_pct = (discount_penetration / total_orders_per_category * 100).fillna(0)
        
        category_discounts['discount_penetration_pct'] = category_discounts['category'].map(penetration_pct)
        
        # Calculate discount effectiveness score (revenue per discount point)
        category_discounts['discount_effectiveness_score'] = (
            category_discounts['total_revenue'] / (category_discounts['avg_discount'] * 100 + 1)
        )
        
        logger.info(f"Generated category discount map: {len(category_discounts)} categories")
        return category_discounts
        
    except Exception as e:
        logger.error(f"Error generating category discount map: {e}")
        return pd.DataFrame()


def generate_anomaly_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate anomaly records from processed data.
    
    Args:
        df: Processed data DataFrame
        
    Returns:
        Anomaly records DataFrame
    """
    logger.info("Generating anomaly records...")
    
    try:
        # Check if anomaly columns exist from ETL pipeline
        if 'is_anomaly' in df.columns:
            anomaly_records = df[df['is_anomaly'] == True].copy()
        elif 'anomaly_score' in df.columns:
            # Use anomaly score threshold
            anomaly_threshold = 3.0
            anomaly_records = df[df['anomaly_score'] > anomaly_threshold].copy()
        else:
            # Generate anomalies using statistical methods
            logger.info("No anomaly columns found, generating using statistical methods...")
            
            # Revenue-based anomalies (Z-score > 3)
            revenue_mean = df['revenue'].mean()
            revenue_std = df['revenue'].std()
            revenue_z_scores = np.abs((df['revenue'] - revenue_mean) / revenue_std)
            
            # Quantity-based anomalies
            quantity_mean = df['quantity'].mean()
            quantity_std = df['quantity'].std()
            quantity_z_scores = np.abs((df['quantity'] - quantity_mean) / quantity_std)
            
            # Combine anomaly conditions
            anomaly_mask = (revenue_z_scores > 3) | (quantity_z_scores > 3)
            anomaly_records = df[anomaly_mask].copy()
            
            # Add anomaly score and type
            anomaly_records['anomaly_score'] = np.maximum(revenue_z_scores[anomaly_mask], quantity_z_scores[anomaly_mask])
            anomaly_records['anomaly_type'] = np.where(
                revenue_z_scores[anomaly_mask] > quantity_z_scores[anomaly_mask],
                'High Revenue',
                'High Quantity'
            )
        
        if len(anomaly_records) > 0:
            # Select relevant columns for display
            display_columns = ['order_id', 'product_name', 'revenue', 'quantity', 'unit_price']
            if 'anomaly_score' in anomaly_records.columns:
                display_columns.append('anomaly_score')
            if 'anomaly_type' in anomaly_records.columns:
                display_columns.append('anomaly_type')
            
            # Filter to existing columns
            available_columns = [col for col in display_columns if col in anomaly_records.columns]
            anomaly_records = anomaly_records[available_columns]
            
            # Sort by anomaly score if available, otherwise by revenue
            if 'anomaly_score' in anomaly_records.columns:
                anomaly_records = anomaly_records.sort_values('anomaly_score', ascending=False)
            else:
                anomaly_records = anomaly_records.sort_values('revenue', ascending=False)
            
            # Take top 100 anomalies
            anomaly_records = anomaly_records.head(100)
            
            # Add reason if not present
            if 'reason' not in anomaly_records.columns:
                anomaly_records['reason'] = 'Statistical outlier detected from processed data'
            
            logger.info(f"Generated anomaly records: {len(anomaly_records)} anomalies")
        else:
            logger.info("No anomalies found in processed data")
            anomaly_records = pd.DataFrame()
        
        return anomaly_records
        
    except Exception as e:
        logger.error(f"Error generating anomaly records: {e}")
        return pd.DataFrame()


def save_analytical_tables(tables: dict, output_dir: str = "data/output"):
    """
    Save analytical tables to output directory.
    
    Args:
        tables: Dictionary of analytical tables
        output_dir: Output directory
    """
    logger.info("Saving analytical tables...")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for table_name, df in tables.items():
        if len(df) > 0:
            try:
                # Save as CSV
                csv_path = os.path.join(output_dir, f"{table_name}.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {table_name}: {len(df)} records to {csv_path}")
                saved_count += 1
                
                # Also save as Parquet if possible
                try:
                    parquet_path = os.path.join(output_dir, f"{table_name}.parquet")
                    df.to_parquet(parquet_path, index=False)
                    logger.info(f"Saved {table_name} to Parquet: {parquet_path}")
                except ImportError:
                    logger.warning("Parquet format not available (pyarrow not installed)")
                except Exception as e:
                    logger.warning(f"Could not save {table_name} as Parquet: {e}")
                    
            except Exception as e:
                logger.error(f"Error saving {table_name}: {e}")
        else:
            logger.warning(f"Skipping empty table: {table_name}")
    
    logger.info(f"Successfully saved {saved_count} analytical tables")


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("GENERATING DASHBOARD DATA FROM ETL PIPELINE OUTPUT")
    logger.info("=" * 60)
    
    try:
        # Step 1: Find latest processed data
        logger.info("Step 1: Finding latest processed data from ETL pipeline...")
        latest_file = find_latest_processed_data()
        
        # Step 2: Load processed data
        logger.info("Step 2: Loading processed data...")
        processed_data = load_processed_data(latest_file)
        
        # Step 3: Ensure revenue column exists
        logger.info("Step 3: Ensuring revenue column exists...")
        processed_data = ensure_revenue_column(processed_data)
        
        # Step 4: Generate analytical tables
        logger.info("Step 4: Generating analytical tables...")
        
        tables = {}
        
        # Generate each analytical table
        tables['monthly_sales_summary'] = generate_monthly_sales_summary(processed_data)
        tables['top_products'] = generate_top_products(processed_data)
        tables['region_wise_performance'] = generate_regional_performance(processed_data)
        tables['category_discount_map'] = generate_category_discount_map(processed_data)
        tables['anomaly_records'] = generate_anomaly_records(processed_data)
        
        # Step 5: Save analytical tables
        logger.info("Step 5: Saving analytical tables...")
        save_analytical_tables(tables)
        
        # Step 6: Generate summary
        logger.info("Step 6: Generating summary...")
        
        total_tables = len(tables)
        non_empty_tables = sum(1 for df in tables.values() if len(df) > 0)
        total_records = sum(len(df) for df in tables.values())
        
        logger.info("=" * 60)
        logger.info("DASHBOARD DATA GENERATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Source Data: {latest_file}")
        logger.info(f"Source Records: {len(processed_data):,}")
        logger.info(f"Generated Tables: {non_empty_tables}/{total_tables}")
        logger.info(f"Total Analytical Records: {total_records:,}")
        logger.info("")
        
        # Table details
        for table_name, df in tables.items():
            if len(df) > 0:
                logger.info(f"✓ {table_name}: {len(df)} records")
            else:
                logger.info(f"✗ {table_name}: EMPTY")
        
        logger.info("")
        logger.info("Dashboard data is ready! You can now start the dashboard:")
        logger.info("streamlit run src/dashboard/dashboard_app.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating dashboard data: {e}")
        logger.error("Falling back to sample data generation...")
        
        # Fallback to sample data
        try:
            import generate_dashboard_data
            generate_dashboard_data.generate_sample_analytical_data()
            logger.info("Generated sample data as fallback")
            return True
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)