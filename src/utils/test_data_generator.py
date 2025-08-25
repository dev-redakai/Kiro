"""
Test Data Generator for creating realistic e-commerce datasets.

This module provides the TestDataGenerator class that creates clean, dirty,
and anomalous test datasets for testing the data processing pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import random
import string
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """
    Generates realistic test datasets for e-commerce data processing pipeline.
    
    This class creates three types of datasets:
    1. Clean data with realistic e-commerce patterns
    2. Dirty data with common quality issues
    3. Anomaly data with known suspicious patterns
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the TestDataGenerator.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self._setup_data_patterns()
    
    def _setup_data_patterns(self) -> None:
        """Set up realistic data patterns for generation."""
        
        # Product catalog with realistic pricing
        self.products = {
            'Electronics': {
                'Laptop': {'price_range': (300, 2000), 'weight': 0.15},
                'Desktop Computer': {'price_range': (400, 3000), 'weight': 0.08},
                'Smartphone': {'price_range': (100, 1200), 'weight': 0.20},
                'Tablet': {'price_range': (150, 800), 'weight': 0.12},
                'Headphones': {'price_range': (20, 400), 'weight': 0.15},
                'Smart Watch': {'price_range': (100, 600), 'weight': 0.10},
                'Camera': {'price_range': (200, 2500), 'weight': 0.08},
                'Gaming Console': {'price_range': (300, 600), 'weight': 0.05},
                'Monitor': {'price_range': (150, 1000), 'weight': 0.07}
            },
            'Fashion': {
                'T-Shirt': {'price_range': (10, 80), 'weight': 0.20},
                'Jeans': {'price_range': (30, 150), 'weight': 0.15},
                'Dress': {'price_range': (25, 200), 'weight': 0.12},
                'Shoes': {'price_range': (40, 300), 'weight': 0.18},
                'Jacket': {'price_range': (50, 250), 'weight': 0.10},
                'Handbag': {'price_range': (30, 400), 'weight': 0.08},
                'Watch': {'price_range': (25, 500), 'weight': 0.07},
                'Sunglasses': {'price_range': (15, 200), 'weight': 0.05},
                'Belt': {'price_range': (15, 100), 'weight': 0.05}
            },
            'Home Goods': {
                'Coffee Maker': {'price_range': (30, 300), 'weight': 0.12},
                'Vacuum Cleaner': {'price_range': (80, 500), 'weight': 0.10},
                'Bed Sheets': {'price_range': (20, 150), 'weight': 0.15},
                'Kitchen Knife Set': {'price_range': (25, 200), 'weight': 0.08},
                'Lamp': {'price_range': (20, 150), 'weight': 0.12},
                'Pillow': {'price_range': (15, 80), 'weight': 0.15},
                'Cookware Set': {'price_range': (50, 300), 'weight': 0.08},
                'Towel Set': {'price_range': (20, 100), 'weight': 0.10},
                'Storage Box': {'price_range': (10, 60), 'weight': 0.10}
            }
        }
        
        # Regional patterns
        self.regions = {
            'North': {'weight': 0.25, 'avg_discount': 0.12},
            'South': {'weight': 0.30, 'avg_discount': 0.15},
            'East': {'weight': 0.20, 'avg_discount': 0.10},
            'West': {'weight': 0.25, 'avg_discount': 0.13}
        }
        
        # Seasonal patterns (month -> discount multiplier)
        self.seasonal_patterns = {
            1: 0.8,   # January - post-holiday low sales
            2: 0.9,   # February
            3: 1.0,   # March
            4: 1.1,   # April
            5: 1.0,   # May
            6: 1.2,   # June - summer sales
            7: 1.3,   # July - peak summer
            8: 1.1,   # August
            9: 1.0,   # September
            10: 1.1,  # October
            11: 1.4,  # November - Black Friday
            12: 1.5   # December - Holiday season
        }
        
        # Common dirty data patterns
        self.dirty_patterns = {
            'product_name_variations': [
                lambda name: name.lower(),
                lambda name: name.upper(),
                lambda name: name.replace(' ', '_'),
                lambda name: name.replace(' ', ''),
                lambda name: f"  {name}  ",  # extra spaces
                lambda name: name + " - NEW",
                lambda name: name + " (Used)",
                lambda name: name.replace('e', '3').replace('a', '@'),  # l33t speak
            ],
            'category_variations': {
                'Electronics': ['electronics', 'electronic', 'elec', 'tech', 'ELECTRONICS'],
                'Fashion': ['fashion', 'clothing', 'apparel', 'clothes', 'FASHION'],
                'Home Goods': ['home', 'home goods', 'homegoods', 'home_goods', 'household']
            },
            'region_variations': {
                'North': ['north', 'nort', 'northern', 'n', 'NORTH'],
                'South': ['south', 'sout', 'southern', 's', 'SOUTH'],
                'East': ['east', 'eastern', 'e', 'EAST'],
                'West': ['west', 'western', 'w', 'WEST']
            }
        }
    
    def generate_clean_sample(self, size: int, start_date: str = "2023-01-01", 
                            end_date: str = "2023-12-31") -> pd.DataFrame:
        """
        Generate clean sample data with realistic e-commerce patterns.
        
        Args:
            size: Number of records to generate
            start_date: Start date for sales data (YYYY-MM-DD format)
            end_date: End date for sales data (YYYY-MM-DD format)
            
        Returns:
            DataFrame with clean e-commerce data
            
        Requirements: 6.7
        """
        logger.info(f"Generating {size} clean sample records")
        
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            date_range = (end_dt - start_dt).days
            
            records = []
            
            for i in range(size):
                # Generate sale date with realistic distribution
                days_offset = np.random.exponential(date_range / 4) % date_range
                sale_date = start_dt + timedelta(days=int(days_offset))
                
                # Select category based on weights
                category = np.random.choice(
                    list(self.products.keys()),
                    p=[0.4, 0.35, 0.25]  # Electronics, Fashion, Home Goods
                )
                
                # Select product within category
                products_in_category = self.products[category]
                product_names = list(products_in_category.keys())
                product_weights = [products_in_category[p]['weight'] for p in product_names]
                product_weights = np.array(product_weights) / sum(product_weights)
                
                product_name = np.random.choice(product_names, p=product_weights)
                price_range = products_in_category[product_name]['price_range']
                
                # Generate realistic pricing
                unit_price = np.random.uniform(price_range[0], price_range[1])
                
                # Generate quantity with realistic distribution (most orders are 1-3 items)
                quantity = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                          p=[0.4, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005])
                
                # Generate region
                region_names = list(self.regions.keys())
                region_weights = [self.regions[r]['weight'] for r in region_names]
                region = np.random.choice(region_names, p=region_weights)
                
                # Generate discount with seasonal and regional patterns
                base_discount = self.regions[region]['avg_discount']
                seasonal_multiplier = self.seasonal_patterns[sale_date.month]
                discount_percent = min(0.5, max(0.0, 
                    np.random.normal(base_discount * seasonal_multiplier, 0.05)))
                
                # Calculate revenue
                revenue = quantity * unit_price * (1 - discount_percent)
                
                # Generate customer email
                customer_id = f"{i:06d}"
                customer_email = f"customer_{customer_id}@example.com"
                
                record = {
                    'order_id': f'ORD_{customer_id}',
                    'product_name': product_name,
                    'category': category,
                    'quantity': quantity,
                    'unit_price': round(unit_price, 2),
                    'discount_percent': round(discount_percent, 4),
                    'region': region,
                    'sale_date': sale_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'customer_email': customer_email,
                    'revenue': round(revenue, 2)
                }
                
                records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"Successfully generated {len(df)} clean records")
            return df
            
        except Exception as e:
            logger.error(f"Error generating clean sample data: {e}")
            raise
    
    def generate_dirty_sample(self, size: int, start_date: str = "2023-01-01", 
                            end_date: str = "2023-12-31", 
                            dirty_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate dirty data with common quality issues.
        
        Args:
            size: Number of records to generate
            start_date: Start date for sales data
            end_date: End date for sales data
            dirty_ratio: Proportion of records that should have quality issues
            
        Returns:
            DataFrame with dirty e-commerce data
            
        Requirements: 6.7
        """
        logger.info(f"Generating {size} dirty sample records with {dirty_ratio:.1%} dirty ratio")
        
        # Start with clean data
        df = self.generate_clean_sample(size, start_date, end_date)
        
        # Determine which records to make dirty
        n_dirty = int(size * dirty_ratio)
        dirty_indices = np.random.choice(df.index, size=n_dirty, replace=False)
        
        for idx in dirty_indices:
            # Apply random dirty patterns
            dirty_type = np.random.choice([
                'product_name', 'category', 'region', 'quantity', 
                'discount', 'email', 'price', 'date', 'duplicate'
            ])
            
            if dirty_type == 'product_name':
                # Apply product name variations
                variation_func = np.random.choice(self.dirty_patterns['product_name_variations'])
                df.at[idx, 'product_name'] = variation_func(df.at[idx, 'product_name'])
            
            elif dirty_type == 'category':
                # Use category variations
                original_category = df.at[idx, 'category']
                if original_category in self.dirty_patterns['category_variations']:
                    variations = self.dirty_patterns['category_variations'][original_category]
                    df.at[idx, 'category'] = np.random.choice(variations)
            
            elif dirty_type == 'region':
                # Use region variations
                original_region = df.at[idx, 'region']
                if original_region in self.dirty_patterns['region_variations']:
                    variations = self.dirty_patterns['region_variations'][original_region]
                    df.at[idx, 'region'] = np.random.choice(variations)
            
            elif dirty_type == 'quantity':
                # Invalid quantities - convert to object dtype to handle mixed types
                df['quantity'] = df['quantity'].astype('object')
                invalid_qty = np.random.choice([0, -1, -5, '0', 'invalid', '', None])
                df.at[idx, 'quantity'] = invalid_qty
            
            elif dirty_type == 'discount':
                # Invalid discount percentages - convert to object dtype to handle mixed types
                df['discount_percent'] = df['discount_percent'].astype('object')
                invalid_discount = np.random.choice([1.5, 2.0, -0.1, 'invalid', '', None])
                df.at[idx, 'discount_percent'] = invalid_discount
            
            elif dirty_type == 'email':
                # Invalid email formats
                invalid_emails = [
                    'invalid-email',
                    'user@',
                    '@domain.com',
                    'user.domain.com',
                    '',
                    None,
                    'user@domain',
                    'user name@domain.com'
                ]
                df.at[idx, 'customer_email'] = np.random.choice(invalid_emails)
            
            elif dirty_type == 'price':
                # Invalid prices - convert to object dtype to handle mixed types
                df['unit_price'] = df['unit_price'].astype('object')
                invalid_price = np.random.choice([0, -10, -100, 'invalid', '', None])
                df.at[idx, 'unit_price'] = invalid_price
            
            elif dirty_type == 'date':
                # Invalid date formats - convert to object dtype to handle mixed types
                df['sale_date'] = df['sale_date'].astype('object')
                invalid_dates = [
                    '2023/13/01',  # Invalid month
                    '2023-02-30',  # Invalid day
                    'invalid-date',
                    '01-01-2023',  # Different format
                    '2023.01.01',  # Different format
                    '',
                    None
                ]
                df.at[idx, 'sale_date'] = np.random.choice(invalid_dates)
            
            elif dirty_type == 'duplicate':
                # Create duplicate order IDs
                if idx > 0:
                    df.at[idx, 'order_id'] = df.at[idx-1, 'order_id']
        
        # Add some completely null rows
        n_null_rows = int(size * 0.02)  # 2% null rows
        null_indices = np.random.choice(df.index, size=n_null_rows, replace=False)
        for idx in null_indices:
            null_fields = np.random.choice(df.columns, size=np.random.randint(1, 4), replace=False)
            for field in null_fields:
                df.at[idx, field] = None
        
        logger.info(f"Successfully generated {len(df)} dirty records")
        return df
    
    def generate_anomaly_sample(self, size: int, start_date: str = "2023-01-01", 
                              end_date: str = "2023-12-31", 
                              anomaly_ratio: float = 0.05) -> pd.DataFrame:
        """
        Generate data with known suspicious patterns for anomaly detection testing.
        
        Args:
            size: Number of records to generate
            start_date: Start date for sales data
            end_date: End date for sales data
            anomaly_ratio: Proportion of records that should be anomalous
            
        Returns:
            DataFrame with anomalous e-commerce data
            
        Requirements: 6.7
        """
        logger.info(f"Generating {size} records with {anomaly_ratio:.1%} anomalies")
        
        # Start with clean data
        df = self.generate_clean_sample(size, start_date, end_date)
        
        # Determine which records to make anomalous
        n_anomalies = int(size * anomaly_ratio)
        anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice([
                'extreme_revenue', 'extreme_quantity', 'extreme_discount', 
                'suspicious_combination', 'price_manipulation'
            ])
            
            if anomaly_type == 'extreme_revenue':
                # Generate extremely high revenue transactions
                normal_revenue = df['revenue'].median()
                extreme_multiplier = np.random.uniform(10, 50)
                df.at[idx, 'unit_price'] = df.at[idx, 'unit_price'] * extreme_multiplier
                df.at[idx, 'revenue'] = (df.at[idx, 'quantity'] * 
                                       df.at[idx, 'unit_price'] * 
                                       (1 - df.at[idx, 'discount_percent']))
            
            elif anomaly_type == 'extreme_quantity':
                # Unrealistic quantities
                extreme_qty = np.random.choice([100, 500, 1000, 2000])
                df.at[idx, 'quantity'] = extreme_qty
                df.at[idx, 'revenue'] = (df.at[idx, 'quantity'] * 
                                       df.at[idx, 'unit_price'] * 
                                       (1 - df.at[idx, 'discount_percent']))
            
            elif anomaly_type == 'extreme_discount':
                # Suspicious discount patterns
                extreme_discount = np.random.choice([0.95, 0.99, 0.8, 0.9])
                df.at[idx, 'discount_percent'] = extreme_discount
                df.at[idx, 'revenue'] = (df.at[idx, 'quantity'] * 
                                       df.at[idx, 'unit_price'] * 
                                       (1 - df.at[idx, 'discount_percent']))
            
            elif anomaly_type == 'suspicious_combination':
                # High-value item with extreme discount and high quantity
                df.at[idx, 'unit_price'] = np.random.uniform(1000, 5000)
                df.at[idx, 'quantity'] = np.random.randint(50, 200)
                df.at[idx, 'discount_percent'] = np.random.uniform(0.7, 0.95)
                df.at[idx, 'revenue'] = (df.at[idx, 'quantity'] * 
                                       df.at[idx, 'unit_price'] * 
                                       (1 - df.at[idx, 'discount_percent']))
            
            elif anomaly_type == 'price_manipulation':
                # Unusual pricing patterns
                if np.random.random() < 0.5:
                    # Extremely low price for expensive category
                    if df.at[idx, 'category'] == 'Electronics':
                        df.at[idx, 'unit_price'] = np.random.uniform(0.01, 1.0)
                else:
                    # Round number pricing (potential manipulation)
                    round_prices = [1000, 2000, 5000, 10000]
                    df.at[idx, 'unit_price'] = np.random.choice(round_prices)
                
                df.at[idx, 'revenue'] = (df.at[idx, 'quantity'] * 
                                       df.at[idx, 'unit_price'] * 
                                       (1 - df.at[idx, 'discount_percent']))
        
        logger.info(f"Successfully generated {len(df)} records with anomalies")
        return df
    
    def generate_configurable_dataset(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate dataset based on configuration parameters.
        
        Args:
            config: Configuration dictionary with generation parameters
            
        Returns:
            DataFrame with generated data based on configuration
            
        Requirements: 6.7
        """
        logger.info(f"Generating configurable dataset with config: {config}")
        
        # Extract configuration parameters
        size = config.get('size', 1000)
        data_type = config.get('type', 'clean')  # clean, dirty, anomaly, mixed
        start_date = config.get('start_date', '2023-01-01')
        end_date = config.get('end_date', '2023-12-31')
        dirty_ratio = config.get('dirty_ratio', 0.3)
        anomaly_ratio = config.get('anomaly_ratio', 0.05)
        
        if data_type == 'clean':
            return self.generate_clean_sample(size, start_date, end_date)
        elif data_type == 'dirty':
            return self.generate_dirty_sample(size, start_date, end_date, dirty_ratio)
        elif data_type == 'anomaly':
            return self.generate_anomaly_sample(size, start_date, end_date, anomaly_ratio)
        elif data_type == 'mixed':
            # Generate mixed dataset with all types
            clean_size = int(size * 0.7)
            dirty_size = int(size * 0.25)
            anomaly_size = size - clean_size - dirty_size
            
            clean_df = self.generate_clean_sample(clean_size, start_date, end_date)
            dirty_df = self.generate_dirty_sample(dirty_size, start_date, end_date, 1.0)
            anomaly_df = self.generate_anomaly_sample(anomaly_size, start_date, end_date, 1.0)
            
            # Combine and shuffle
            combined_df = pd.concat([clean_df, dirty_df, anomaly_df], ignore_index=True)
            return combined_df.sample(frac=1).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def save_dataset(self, df: pd.DataFrame, filename: str, 
                    output_dir: str = "data/raw") -> str:
        """
        Save generated dataset to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Output directory
            
        Returns:
            Full path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        full_path = output_path / filename
        df.to_csv(full_path, index=False)
        
        logger.info(f"Saved dataset with {len(df)} records to {full_path}")
        return str(full_path)
    
    def get_generation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for generated dataset.
        
        Args:
            df: Generated DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            # Handle date range safely
            date_range = {'start': None, 'end': None}
            if 'sale_date' in df.columns:
                try:
                    # Filter out non-datetime values for min/max calculation
                    date_series = pd.to_datetime(df['sale_date'], errors='coerce')
                    valid_dates = date_series.dropna()
                    if len(valid_dates) > 0:
                        date_range['start'] = str(valid_dates.min())
                        date_range['end'] = str(valid_dates.max())
                except Exception:
                    pass
            
            # Handle revenue stats safely
            revenue_stats = {'total': 0, 'mean': 0, 'median': 0, 'std': 0}
            if 'revenue' in df.columns:
                try:
                    # Convert to numeric, coercing errors to NaN
                    revenue_numeric = pd.to_numeric(df['revenue'], errors='coerce')
                    valid_revenue = revenue_numeric.dropna()
                    if len(valid_revenue) > 0:
                        revenue_stats = {
                            'total': float(valid_revenue.sum()),
                            'mean': float(valid_revenue.mean()),
                            'median': float(valid_revenue.median()),
                            'std': float(valid_revenue.std())
                        }
                except Exception:
                    pass
            
            # Handle categorical data safely
            categories = {}
            if 'category' in df.columns:
                try:
                    categories = df['category'].value_counts().to_dict()
                    # Convert numpy types to native Python types
                    categories = {str(k): int(v) for k, v in categories.items()}
                except Exception:
                    pass
            
            regions = {}
            if 'region' in df.columns:
                try:
                    regions = df['region'].value_counts().to_dict()
                    # Convert numpy types to native Python types
                    regions = {str(k): int(v) for k, v in regions.items()}
                except Exception:
                    pass
            
            # Handle data quality metrics safely
            null_values = {}
            duplicate_order_ids = 0
            try:
                null_values = df.isnull().sum().to_dict()
                # Convert numpy types to native Python types
                null_values = {str(k): int(v) for k, v in null_values.items()}
            except Exception:
                pass
            
            try:
                if 'order_id' in df.columns:
                    duplicate_order_ids = int(df['order_id'].duplicated().sum())
            except Exception:
                pass
            
            summary = {
                'total_records': len(df),
                'date_range': date_range,
                'categories': categories,
                'regions': regions,
                'revenue_stats': revenue_stats,
                'data_quality': {
                    'null_values': null_values,
                    'duplicate_order_ids': duplicate_order_ids
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'total_records': len(df),
                'error': str(e)
            }