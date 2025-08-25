"""
Data provider for dashboard application.

This module provides the DataProvider class that loads analytical tables
and supplies data to dashboard components with caching and refresh capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import os
import glob
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class DataProvider:
    """
    DataProvider for loading analytical tables and dashboard data.
    
    This class handles loading processed analytical tables from the ETL pipeline
    output and provides cached data access for the dashboard components.
    Enhanced to prioritize actual ETL pipeline output over sample data.
    """
    
    def __init__(self, data_dir: str = "data/output"):
        """
        Initialize the DataProvider.
        
        Args:
            data_dir: Directory containing processed analytical tables
        """
        self.data_dir = data_dir
        self.cache = {}
        self.cache_timestamp = None
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
        # Expected analytical table files (prioritize ETL output)
        self.table_files = {
            'monthly_sales_summary': 'monthly_sales_summary.csv',
            'top_products': 'top_products.csv',
            'region_wise_performance': 'region_wise_performance.csv',
            'category_discount_map': 'category_discount_map.csv',
            'anomaly_records': 'anomaly_records.csv'
        }
        
        # ETL pipeline output directories
        self.csv_dir = os.path.join(data_dir, 'csv')
        self.parquet_dir = os.path.join(data_dir, 'parquet')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
    
    def _is_cache_valid(self) -> bool:
        """
        Check if the current cache is still valid.
        
        Returns:
            True if cache is valid, False otherwise
        """
        if self.cache_timestamp is None:
            return False
        
        return datetime.now() - self.cache_timestamp < self.cache_duration
    
    def _find_latest_etl_file(self, pattern: str, directory: str) -> Optional[str]:
        """
        Find the most recent ETL pipeline output file matching the pattern.
        
        Args:
            pattern: File pattern to search for
            directory: Directory to search in
            
        Returns:
            Path to the most recent file, or None if not found
        """
        try:
            search_pattern = os.path.join(directory, pattern)
            files = glob.glob(search_pattern)
            
            if not files:
                return None
            
            # Sort by modification time, most recent first
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = files[0]
            
            logger.info(f"Found latest ETL file: {latest_file}")
            return latest_file
            
        except Exception as e:
            logger.error(f"Error finding latest ETL file with pattern {pattern}: {str(e)}")
            return None
    
    def _load_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} records from {file_path}")
                return df
            else:
                logger.warning(f"File not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def _load_parquet_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a Parquet file with error handling.
        
        Args:
            file_path: Path to the Parquet file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(df)} records from {file_path}")
                return df
            else:
                logger.warning(f"File not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def _get_latest_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Get the most recent processed dataset from ETL pipeline output.
        
        Returns:
            DataFrame with the latest processed data, or None if not found
        """
        # Try to find the latest processed data file
        csv_pattern = "*_processed_*.csv"
        parquet_pattern = "*_processed_*.parquet"
        
        # First try CSV format
        latest_csv = self._find_latest_etl_file(csv_pattern, self.csv_dir)
        if latest_csv:
            df = self._load_csv_file(latest_csv)
            if df is not None:
                logger.info(f"Using latest processed CSV data: {latest_csv}")
                return df
        
        # Then try Parquet format
        latest_parquet = self._find_latest_etl_file(parquet_pattern, self.parquet_dir)
        if latest_parquet:
            df = self._load_parquet_file(latest_parquet)
            if df is not None:
                logger.info(f"Using latest processed Parquet data: {latest_parquet}")
                return df
        
        logger.warning("No processed data files found from ETL pipeline")
        return None
    
    def _generate_analytical_tables_from_processed_data(self, processed_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate analytical tables from processed data if individual tables aren't available.
        
        Args:
            processed_data: The main processed dataset from ETL pipeline
            
        Returns:
            Dictionary of analytical tables
        """
        logger.info("Generating analytical tables from processed data")
        tables = {}
        
        try:
            # Ensure required columns exist
            if 'revenue' not in processed_data.columns:
                if all(col in processed_data.columns for col in ['quantity', 'unit_price', 'discount_percent']):
                    processed_data['revenue'] = (processed_data['quantity'] * 
                                               processed_data['unit_price'] * 
                                               (1 - processed_data['discount_percent']))
                else:
                    logger.warning("Cannot calculate revenue - missing required columns")
                    return {}
            
            # 1. Monthly Sales Summary
            if 'sale_date' in processed_data.columns:
                try:
                    processed_data['sale_date'] = pd.to_datetime(processed_data['sale_date'], errors='coerce')
                    monthly_sales = processed_data.groupby(
                        processed_data['sale_date'].dt.to_period('M')
                    ).agg({
                        'revenue': 'sum',
                        'quantity': 'sum',
                        'discount_percent': 'mean',
                        'order_id': 'nunique'
                    }).reset_index()
                    
                    monthly_sales.columns = ['month', 'total_revenue', 'total_quantity', 'avg_discount', 'unique_orders']
                    monthly_sales['month'] = monthly_sales['month'].astype(str)
                    monthly_sales['avg_order_value'] = monthly_sales['total_revenue'] / monthly_sales['unique_orders']
                    
                    tables['monthly_sales_summary'] = monthly_sales
                    logger.info(f"Generated monthly sales summary: {len(monthly_sales)} records")
                except Exception as e:
                    logger.error(f"Error generating monthly sales summary: {e}")
            
            # 2. Top Products
            if 'product_name' in processed_data.columns:
                try:
                    top_products = processed_data.groupby('product_name').agg({
                        'revenue': 'sum',
                        'quantity': 'sum',
                        'unit_price': 'mean',
                        'discount_percent': 'mean'
                    }).reset_index()
                    
                    top_products.columns = ['product_name', 'total_revenue', 'total_units', 'avg_unit_price', 'avg_discount']
                    top_products = top_products.sort_values('total_revenue', ascending=False).head(20)
                    top_products['revenue_rank'] = range(1, len(top_products) + 1)
                    
                    # Add units rank
                    top_products_by_units = top_products.sort_values('total_units', ascending=False)
                    units_rank_map = {name: rank for rank, name in enumerate(top_products_by_units['product_name'], 1)}
                    top_products['units_rank'] = top_products['product_name'].map(units_rank_map)
                    
                    tables['top_products'] = top_products
                    logger.info(f"Generated top products: {len(top_products)} records")
                except Exception as e:
                    logger.error(f"Error generating top products: {e}")
            
            # 3. Regional Performance
            if 'region' in processed_data.columns:
                try:
                    regional_perf = processed_data.groupby('region').agg({
                        'revenue': 'sum',
                        'order_id': 'nunique',
                        'quantity': 'sum'
                    }).reset_index()
                    
                    regional_perf.columns = ['region', 'total_revenue', 'unique_orders', 'total_quantity']
                    regional_perf['avg_order_value'] = regional_perf['total_revenue'] / regional_perf['unique_orders']
                    
                    # Calculate market share
                    total_revenue = regional_perf['total_revenue'].sum()
                    regional_perf['market_share_pct'] = (regional_perf['total_revenue'] / total_revenue * 100)
                    
                    # Add top category per region
                    if 'category' in processed_data.columns:
                        region_category = processed_data.groupby(['region', 'category'])['revenue'].sum().reset_index()
                        top_category_per_region = region_category.loc[region_category.groupby('region')['revenue'].idxmax()]
                        top_category_map = dict(zip(top_category_per_region['region'], top_category_per_region['category']))
                        regional_perf['top_category'] = regional_perf['region'].map(top_category_map)
                    
                    tables['region_wise_performance'] = regional_perf
                    logger.info(f"Generated regional performance: {len(regional_perf)} records")
                except Exception as e:
                    logger.error(f"Error generating regional performance: {e}")
            
            # 4. Category Discount Map
            if 'category' in processed_data.columns:
                try:
                    category_discounts = processed_data.groupby('category').agg({
                        'discount_percent': ['mean', 'count'],
                        'revenue': 'sum'
                    }).reset_index()
                    
                    # Flatten column names
                    category_discounts.columns = ['category', 'avg_discount', 'discount_count', 'total_revenue']
                    
                    # Calculate discount penetration (percentage of orders with discount > 0)
                    discount_penetration = processed_data[processed_data['discount_percent'] > 0].groupby('category').size()
                    total_orders_per_category = processed_data.groupby('category').size()
                    penetration_pct = (discount_penetration / total_orders_per_category * 100).fillna(0)
                    
                    category_discounts['discount_penetration_pct'] = category_discounts['category'].map(penetration_pct)
                    
                    # Calculate discount effectiveness score (revenue per discount point)
                    category_discounts['discount_effectiveness_score'] = (
                        category_discounts['total_revenue'] / (category_discounts['avg_discount'] * 100 + 1)
                    )
                    
                    tables['category_discount_map'] = category_discounts
                    logger.info(f"Generated category discount map: {len(category_discounts)} records")
                except Exception as e:
                    logger.error(f"Error generating category discount map: {e}")
            
            # 5. Anomaly Records (if available from processed data)
            if 'is_anomaly' in processed_data.columns or 'anomaly_score' in processed_data.columns:
                try:
                    if 'is_anomaly' in processed_data.columns:
                        anomaly_records = processed_data[processed_data['is_anomaly'] == True].copy()
                    else:
                        # Use anomaly score threshold
                        anomaly_threshold = 3.0
                        anomaly_records = processed_data[processed_data['anomaly_score'] > anomaly_threshold].copy()
                    
                    if len(anomaly_records) > 0:
                        # Select relevant columns for anomaly display
                        anomaly_columns = ['order_id', 'product_name', 'revenue', 'quantity', 'unit_price']
                        if 'anomaly_score' in anomaly_records.columns:
                            anomaly_columns.append('anomaly_score')
                        if 'anomaly_type' in anomaly_records.columns:
                            anomaly_columns.append('anomaly_type')
                        
                        # Filter to existing columns
                        available_columns = [col for col in anomaly_columns if col in anomaly_records.columns]
                        anomaly_records = anomaly_records[available_columns]
                        
                        # Sort by anomaly score if available, otherwise by revenue
                        if 'anomaly_score' in anomaly_records.columns:
                            anomaly_records = anomaly_records.sort_values('anomaly_score', ascending=False)
                        else:
                            anomaly_records = anomaly_records.sort_values('revenue', ascending=False)
                        
                        # Take top 50 anomalies
                        anomaly_records = anomaly_records.head(50)
                        
                        # Add reason if not present
                        if 'reason' not in anomaly_records.columns:
                            anomaly_records['reason'] = 'Statistical outlier detected by ETL pipeline'
                        
                        tables['anomaly_records'] = anomaly_records
                        logger.info(f"Generated anomaly records: {len(anomaly_records)} records")
                    else:
                        logger.info("No anomaly records found in processed data")
                        tables['anomaly_records'] = pd.DataFrame()
                except Exception as e:
                    logger.error(f"Error generating anomaly records: {e}")
                    tables['anomaly_records'] = pd.DataFrame()
            
            logger.info(f"Successfully generated {len(tables)} analytical tables from processed data")
            return tables
            
        except Exception as e:
            logger.error(f"Error generating analytical tables from processed data: {str(e)}")
            return {}
    
    def _generate_sample_data(self, table_name: str) -> pd.DataFrame:
        """
        Generate sample data for testing when actual data is not available.
        
        Args:
            table_name: Name of the analytical table
            
        Returns:
            Sample DataFrame
        """
        logger.warning(f"Generating sample data for {table_name}")
        
        if table_name == 'monthly_sales_summary':
            dates = pd.date_range('2023-01', '2024-12', freq='ME')
            return pd.DataFrame({
                'month': [d.strftime('%Y-%m') for d in dates],
                'total_revenue': np.random.uniform(50000, 200000, len(dates)),
                'total_quantity': np.random.randint(1000, 5000, len(dates)),
                'avg_discount': np.random.uniform(0.05, 0.25, len(dates)),
                'unique_orders': np.random.randint(500, 2000, len(dates)),
                'avg_order_value': np.random.uniform(80, 150, len(dates))
            })
        
        elif table_name == 'top_products':
            products = [f'Product {i}' for i in range(1, 21)]
            categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Books']
            return pd.DataFrame({
                'product_name': products,
                'category': np.random.choice(categories, len(products)),
                'total_revenue': np.random.uniform(10000, 100000, len(products)),
                'total_units': np.random.randint(100, 2000, len(products)),
                'avg_unit_price': np.random.uniform(20, 500, len(products)),
                'avg_discount': np.random.uniform(0.0, 0.3, len(products)),
                'revenue_rank': range(1, len(products) + 1),
                'units_rank': np.random.permutation(range(1, len(products) + 1))
            })
        
        elif table_name == 'region_wise_performance':
            regions = ['North', 'South', 'East', 'West', 'Central']
            return pd.DataFrame({
                'region': regions,
                'total_revenue': np.random.uniform(100000, 500000, len(regions)),
                'unique_orders': np.random.randint(2000, 10000, len(regions)),
                'avg_order_value': np.random.uniform(80, 200, len(regions)),
                'market_share_pct': np.random.uniform(15, 25, len(regions)),
                'top_category': np.random.choice(['Electronics', 'Fashion', 'Home'], len(regions))
            })
        
        elif table_name == 'category_discount_map':
            categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Books', 'Beauty']
            return pd.DataFrame({
                'category': categories,
                'avg_discount': np.random.uniform(0.05, 0.35, len(categories)),
                'discount_penetration_pct': np.random.uniform(30, 80, len(categories)),
                'total_revenue': np.random.uniform(50000, 300000, len(categories)),
                'discount_effectiveness_score': np.random.uniform(1000, 50000, len(categories))
            })
        
        elif table_name == 'anomaly_records':
            return pd.DataFrame({
                'order_id': [f'ORD{i:06d}' for i in range(1, 11)],
                'product_name': [f'Anomaly Product {i}' for i in range(1, 11)],
                'revenue': np.random.uniform(5000, 50000, 10),
                'anomaly_type': np.random.choice(['High Revenue', 'High Quantity', 'High Discount'], 10),
                'anomaly_score': np.random.uniform(3.0, 10.0, 10),
                'reason': ['Statistical outlier'] * 10
            })
        
        return pd.DataFrame()
    
    def load_analytical_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load all analytical tables, prioritizing ETL pipeline output.
        
        Loading priority:
        1. Individual analytical tables from ETL pipeline output (timestamped files)
        2. Direct analytical table files in data/output
        3. Generate from latest processed data
        4. Generate sample data as fallback
        
        Returns:
            Dictionary containing all analytical tables
            
        Requirements: 4.4, 4.6
        """
        # Check cache first
        if self._is_cache_valid() and self.cache:
            logger.info("Returning cached analytical tables")
            return self.cache
        
        logger.info("Loading analytical tables from ETL pipeline output")
        tables = {}
        
        # Step 1: Try to load individual analytical tables from ETL pipeline output
        etl_tables_loaded = 0
        for table_name, file_name in self.table_files.items():
            df = None
            
            # Try to find timestamped analytical files from ETL pipeline
            csv_pattern = f"analytical_{table_name}_*.csv"
            parquet_pattern = f"analytical_{table_name}_*.parquet"
            
            # First try CSV format from ETL output
            latest_csv = self._find_latest_etl_file(csv_pattern, self.csv_dir)
            if latest_csv:
                df = self._load_csv_file(latest_csv)
                if df is not None:
                    logger.info(f"Loaded {table_name} from ETL CSV output: {latest_csv}")
                    etl_tables_loaded += 1
            
            # If not found, try Parquet format from ETL output
            if df is None:
                latest_parquet = self._find_latest_etl_file(parquet_pattern, self.parquet_dir)
                if latest_parquet:
                    df = self._load_parquet_file(latest_parquet)
                    if df is not None:
                        logger.info(f"Loaded {table_name} from ETL Parquet output: {latest_parquet}")
                        etl_tables_loaded += 1
            
            # Step 2: Try direct files in data/output (fallback)
            if df is None:
                file_path = os.path.join(self.data_dir, file_name)
                df = self._load_csv_file(file_path)
                
                if df is None:
                    # Try Parquet format
                    parquet_path = file_path.replace('.csv', '.parquet')
                    df = self._load_parquet_file(parquet_path)
            
            tables[table_name] = df
        
        logger.info(f"Loaded {etl_tables_loaded} tables directly from ETL pipeline output")
        
        # Step 3: For missing tables, try to generate from latest processed data
        missing_tables = [name for name, df in tables.items() if df is None]
        
        if missing_tables:
            logger.info(f"Attempting to generate {len(missing_tables)} missing tables from processed data")
            processed_data = self._get_latest_processed_data()
            
            if processed_data is not None:
                generated_tables = self._generate_analytical_tables_from_processed_data(processed_data)
                
                # Update missing tables with generated ones
                for table_name in missing_tables:
                    if table_name in generated_tables:
                        tables[table_name] = generated_tables[table_name]
                        logger.info(f"Generated {table_name} from processed data")
                    else:
                        logger.warning(f"Could not generate {table_name} from processed data")
        
        # Step 4: Generate sample data for any remaining missing tables (fallback)
        final_missing_tables = [name for name, df in tables.items() if df is None]
        
        if final_missing_tables:
            logger.warning(f"Generating sample data for {len(final_missing_tables)} missing tables")
            for table_name in final_missing_tables:
                tables[table_name] = self._generate_sample_data(table_name)
        
        # Ensure all tables are DataFrames (not None)
        for table_name in self.table_files.keys():
            if tables[table_name] is None:
                tables[table_name] = pd.DataFrame()
        
        # Update cache
        self.cache = tables
        self.cache_timestamp = datetime.now()
        
        # Log summary
        total_tables = len(tables)
        non_empty_tables = sum(1 for df in tables.values() if len(df) > 0)
        logger.info(f"Loaded {total_tables} analytical tables ({non_empty_tables} non-empty)")
        
        # Log table details
        for table_name, df in tables.items():
            if len(df) > 0:
                logger.info(f"  - {table_name}: {len(df)} records, {len(df.columns)} columns")
            else:
                logger.warning(f"  - {table_name}: EMPTY")
        
        return tables
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get dashboard summary metrics.
        
        Returns:
            Dictionary containing key dashboard metrics
            
        Requirements: 4.6
        """
        try:
            tables = self.load_analytical_tables()
            metrics = {}
            
            # Calculate metrics from monthly sales summary
            if 'monthly_sales_summary' in tables and not tables['monthly_sales_summary'].empty:
                monthly_data = tables['monthly_sales_summary']
                
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
                    # Sort by month to ensure proper order
                    monthly_data_sorted = monthly_data.sort_values('month')
                    
                    # Revenue growth (last month vs previous)
                    if len(monthly_data_sorted) >= 2 and revenue_col in monthly_data_sorted.columns:
                        last_revenue = monthly_data_sorted[revenue_col].iloc[-1]
                        prev_revenue = monthly_data_sorted[revenue_col].iloc[-2]
                        if prev_revenue > 0:
                            metrics['revenue_growth'] = f"{((last_revenue - prev_revenue) / prev_revenue * 100):+.1f}%"
                        else:
                            metrics['revenue_growth'] = "N/A"
                    else:
                        metrics['revenue_growth'] = "N/A"
                        
                    # Order growth
                    if orders_col in monthly_data_sorted.columns:
                        last_orders = monthly_data_sorted[orders_col].iloc[-1]
                        prev_orders = monthly_data_sorted[orders_col].iloc[-2]
                        if prev_orders > 0:
                            metrics['order_growth'] = f"{((last_orders - prev_orders) / prev_orders * 100):+.1f}%"
                        else:
                            metrics['order_growth'] = "N/A"
                    else:
                        metrics['order_growth'] = "N/A"
                        
                    # AOV growth
                    if aov_col in monthly_data_sorted.columns:
                        last_aov = monthly_data_sorted[aov_col].iloc[-1]
                        prev_aov = monthly_data_sorted[aov_col].iloc[-2]
                        if prev_aov > 0:
                            metrics['aov_growth'] = f"{((last_aov - prev_aov) / prev_aov * 100):+.1f}%"
                        else:
                            metrics['aov_growth'] = "N/A"
                    else:
                        metrics['aov_growth'] = "N/A"
                else:
                    metrics['revenue_growth'] = "N/A"
                    metrics['order_growth'] = "N/A"
                    metrics['aov_growth'] = "N/A"
            else:
                # Set default values if no monthly data
                metrics['total_revenue'] = 0
                metrics['total_orders'] = 0
                metrics['avg_order_value'] = 0
                metrics['avg_discount'] = 0
                metrics['revenue_growth'] = "N/A"
                metrics['order_growth'] = "N/A"
                metrics['aov_growth'] = "N/A"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating dashboard metrics: {str(e)}")
            return {
                'total_revenue': 0,
                'total_orders': 0,
                'avg_order_value': 0,
                'avg_discount': 0,
                'revenue_growth': "N/A",
                'order_growth': "N/A",
                'aov_growth': "N/A"
            }
    
    def load_validation_reports(self) -> Dict[str, Any]:
        """
        Load validation reports from the ETL pipeline for data quality dashboard.
        
        Returns:
            Dictionary containing validation data and metrics
        """
        try:
            logger.info("Loading validation reports for data quality dashboard")
            
            # Look for validation reports in logs directory
            logs_dir = "logs"
            validation_data = {}
            
            # Find the most recent validation debug report
            debug_report_pattern = os.path.join(logs_dir, "validation_debug_report_*.json")
            debug_files = glob.glob(debug_report_pattern)
            
            if debug_files:
                # Get the most recent debug report
                latest_debug_file = max(debug_files, key=os.path.getmtime)
                logger.info(f"Loading validation debug report: {latest_debug_file}")
                
                try:
                    with open(latest_debug_file, 'r') as f:
                        debug_data = json.load(f)
                    
                    # Extract validation metrics
                    validation_metrics = debug_data.get('validation_metrics', {})
                    validation_data.update({
                        'total_rules_executed': validation_metrics.get('total_rules_executed', 0),
                        'successful_rules': validation_metrics.get('successful_rules', 0),
                        'failed_rules': validation_metrics.get('failed_rules', 0),
                        'overall_data_quality_score': validation_metrics.get('overall_data_quality_score', 0),
                        'rule_performance': validation_metrics.get('rule_performance', {}),
                        'error_distribution': validation_metrics.get('error_distribution', {}),
                        'memory_usage_mb': validation_metrics.get('memory_usage_mb', 0),
                        'total_execution_time': sum(validation_metrics.get('rule_performance', {}).values()),
                        'average_rule_time': (sum(validation_metrics.get('rule_performance', {}).values()) / 
                                            max(validation_metrics.get('total_rules_executed', 1), 1)),
                        'validation_history': debug_data.get('validation_history', []),
                        'registered_rules': debug_data.get('registered_rules', []),
                        'report_metadata': debug_data.get('report_metadata', {})
                    })
                    
                    # Calculate total records validated
                    validation_history = debug_data.get('validation_history', [])
                    if validation_history:
                        # Sum unique records from all rules (avoid double counting)
                        unique_records = set()
                        for rule in validation_history:
                            total_records = rule.get('total_records', 0)
                            if total_records > 0:
                                unique_records.add(total_records)
                        
                        validation_data['total_records_validated'] = max(unique_records) if unique_records else 0
                    else:
                        validation_data['total_records_validated'] = 0
                    
                    logger.info(f"Loaded validation data with {len(validation_history)} rule results")
                    
                except Exception as e:
                    logger.error(f"Error parsing validation debug report: {str(e)}")
            
            # Look for validation error reports
            error_report_file = os.path.join(logs_dir, "validation_error_report.json")
            if os.path.exists(error_report_file):
                try:
                    with open(error_report_file, 'r') as f:
                        error_data = json.load(f)
                    
                    validation_data['error_reports'] = error_data
                    logger.info("Loaded validation error reports")
                    
                except Exception as e:
                    logger.error(f"Error loading validation error report: {str(e)}")
            
            # Look for general validation reports
            validation_report_pattern = os.path.join(logs_dir, "validation_report_*.json")
            validation_files = glob.glob(validation_report_pattern)
            
            if validation_files:
                latest_validation_file = max(validation_files, key=os.path.getmtime)
                logger.info(f"Loading validation report: {latest_validation_file}")
                
                try:
                    with open(latest_validation_file, 'r') as f:
                        report_data = json.load(f)
                    
                    # Merge additional validation data
                    validation_data.update(report_data)
                    
                except Exception as e:
                    logger.error(f"Error loading validation report: {str(e)}")
            
            # If no validation data found, return empty dict
            if not validation_data:
                logger.warning("No validation reports found")
                return {}
            
            logger.info(f"Successfully loaded validation data with {validation_data.get('total_rules_executed', 0)} rules")
            return validation_data
            
        except Exception as e:
            logger.error(f"Error loading validation reports: {str(e)}")
            return {}
    
    def refresh_data(self) -> None:
        """
        Refresh cached data by clearing the cache.
        """
        self.cache = {}
        self.cache_timestamp = None
        logger.info("Data cache cleared - will reload on next access")
    
    def export_dashboard_data(self) -> bool:
        """
        Export dashboard data to files.
        
        Returns:
            True if export successful, False otherwise
        """
        try:
            tables = self.load_analytical_tables()
            export_dir = os.path.join(self.data_dir, 'dashboard_export')
            
            # Create export directory
            os.makedirs(export_dir, exist_ok=True)
            
            # Export each table
            for table_name, df in tables.items():
                if not df.empty:
                    export_path = os.path.join(export_dir, f"{table_name}_export.csv")
                    df.to_csv(export_path, index=False)
                    logger.info(f"Exported {table_name} to {export_path}")
            
            logger.info(f"Dashboard data exported to {export_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {str(e)}")
            return False
    
    def get_data_freshness(self) -> Dict[str, Any]:
        """
        Get information about data freshness and availability.
        
        Returns:
            Dictionary with freshness information
        """
        try:
            freshness_info = {
                'cache_valid': self._is_cache_valid(),
                'cache_timestamp': self.cache_timestamp.isoformat() if self.cache_timestamp else None,
                'available_tables': [],
                'missing_tables': [],
                'data_dir_exists': os.path.exists(self.data_dir)
            }
            
            # Check which tables are available
            for table_name, file_name in self.table_files.items():
                file_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(file_path):
                    freshness_info['available_tables'].append(table_name)
                else:
                    freshness_info['missing_tables'].append(table_name)
            
            return freshness_info
            
        except Exception as e:
            logger.error(f"Error getting data freshness info: {str(e)}")
            return {
                'cache_valid': False,
                'cache_timestamp': None,
                'available_tables': [],
                'missing_tables': list(self.table_files.keys()),
                'data_dir_exists': False
            }
            
            # Add regional metrics
            if 'region_wise_performance' in tables and not tables['region_wise_performance'].empty:
                regional_data = tables['region_wise_performance']
                metrics['total_regions'] = len(regional_data)
                if 'total_revenue' in regional_data.columns:
                    metrics['top_region'] = regional_data.loc[regional_data['total_revenue'].idxmax(), 'region']
                else:
                    metrics['top_region'] = "N/A"
            else:
                metrics['total_regions'] = 0
                metrics['top_region'] = "N/A"
            
            # Add product metrics
            if 'top_products' in tables and not tables['top_products'].empty:
                products_data = tables['top_products']
                metrics['total_products'] = len(products_data)
                if 'total_revenue' in products_data.columns:
                    metrics['top_product'] = products_data.loc[products_data['total_revenue'].idxmax(), 'product_name']
                else:
                    metrics['top_product'] = "N/A"
            else:
                metrics['total_products'] = 0
                metrics['top_product'] = "N/A"
            
            # Add anomaly metrics
            if 'anomaly_records' in tables and not tables['anomaly_records'].empty:
                anomaly_data = tables['anomaly_records']
                metrics['total_anomalies'] = len(anomaly_data)
                if 'anomaly_score' in anomaly_data.columns:
                    metrics['avg_anomaly_score'] = anomaly_data['anomaly_score'].mean()
                else:
                    metrics['avg_anomaly_score'] = 0
            else:
                metrics['total_anomalies'] = 0
                metrics['avg_anomaly_score'] = 0
            
            logger.info("Generated dashboard metrics successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating dashboard metrics: {str(e)}")
            return {
                'total_revenue': 0,
                'total_orders': 0,
                'avg_order_value': 0,
                'avg_discount': 0,
                'revenue_growth': "N/A",
                'order_growth': "N/A",
                'aov_growth': "N/A",
                'total_regions': 0,
                'top_region': "N/A",
                'total_products': 0,
                'top_product': "N/A",
                'total_anomalies': 0,
                'avg_anomaly_score': 0
            }
    
    def refresh_data(self) -> bool:
        """
        Refresh cached data by clearing cache and reloading.
        
        Returns:
            True if refresh was successful, False otherwise
            
        Requirements: 4.6
        """
        try:
            logger.info("Refreshing dashboard data")
            
            # Clear cache
            self.cache = {}
            self.cache_timestamp = None
            
            # Reload data
            tables = self.load_analytical_tables()
            
            if tables:
                logger.info("Data refresh completed successfully")
                return True
            else:
                logger.warning("Data refresh completed but no tables loaded")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing data: {str(e)}")
            return False
    
    def get_data_freshness(self) -> Dict[str, Any]:
        """
        Get information about data freshness and availability.
        
        Returns:
            Dictionary with data freshness information
        """
        freshness_info = {
            'cache_valid': self._is_cache_valid(),
            'cache_timestamp': self.cache_timestamp,
            'available_tables': [],
            'missing_tables': [],
            'file_timestamps': {}
        }
        
        for table_name, file_name in self.table_files.items():
            file_path = os.path.join(self.data_dir, file_name)
            
            if os.path.exists(file_path):
                freshness_info['available_tables'].append(table_name)
                # Get file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                freshness_info['file_timestamps'][table_name] = mod_time
            else:
                freshness_info['missing_tables'].append(table_name)
        
        return freshness_info
    
    def export_dashboard_data(self, output_path: str = "dashboard_export.json") -> bool:
        """
        Export current dashboard data to JSON for backup or analysis.
        
        Args:
            output_path: Path for the export file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            tables = self.load_analytical_tables()
            metrics = self.get_dashboard_metrics()
            freshness = self.get_data_freshness()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'freshness_info': {
                    'cache_valid': freshness['cache_valid'],
                    'available_tables': freshness['available_tables'],
                    'missing_tables': freshness['missing_tables']
                },
                'table_summaries': {}
            }
            
            # Add table summaries (not full data to keep file size reasonable)
            for table_name, df in tables.items():
                if not df.empty:
                    export_data['table_summaries'][table_name] = {
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'columns': df.columns.tolist(),
                        'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
                    }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {str(e)}")
            return False