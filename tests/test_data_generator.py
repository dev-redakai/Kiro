"""
Test data generator for creating realistic test datasets.

This module provides utilities to generate clean, dirty, and anomalous
test data for comprehensive testing of the data pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import string


class TestDataGenerator:
    """Generate realistic test datasets for pipeline testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize the test data generator with a random seed."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Realistic product data
        self.products = {
            'Electronics': [
                'iPhone 13 Pro', 'Samsung Galaxy S21', 'iPad Air', 'MacBook Pro',
                'Dell XPS 13', 'Sony WH-1000XM4', 'AirPods Pro', 'Nintendo Switch',
                'PlayStation 5', 'Xbox Series X', 'LG OLED TV', 'Samsung 4K Monitor'
            ],
            'Fashion': [
                'Nike Air Max', 'Adidas Ultraboost', 'Levi\'s Jeans', 'H&M T-Shirt',
                'Zara Dress', 'Gucci Handbag', 'Ray-Ban Sunglasses', 'Rolex Watch',
                'Nike Hoodie', 'Adidas Sneakers', 'Calvin Klein Underwear', 'Tommy Hilfiger Shirt'
            ],
            'Home Goods': [
                'IKEA Sofa', 'Dyson Vacuum', 'KitchenAid Mixer', 'Instant Pot',
                'Philips Air Fryer', 'Roomba Robot Vacuum', 'Nest Thermostat', 'Ring Doorbell',
                'Cuisinart Coffee Maker', 'Vitamix Blender', 'Le Creuset Dutch Oven', 'Shark Steam Mop'
            ]
        }
        
        self.regions = ['North', 'South', 'East', 'West', 'Northeast', 'Southeast', 'Northwest', 'Southwest']
        
        # Price ranges by category
        self.price_ranges = {
            'Electronics': (50, 3000),
            'Fashion': (20, 500),
            'Home Goods': (30, 800)
        }
    
    def generate_clean_sample(self, size: int, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate clean, well-formatted sample data."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        
        data = []
        
        for i in range(size):
            # Select category and product
            category = np.random.choice(list(self.products.keys()))
            product = np.random.choice(self.products[category])
            
            # Generate realistic pricing
            price_min, price_max = self.price_ranges[category]
            unit_price = round(np.random.uniform(price_min, price_max), 2)
            
            # Generate realistic quantities (most orders are small)
            quantity = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                      p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01])
            
            # Generate realistic discounts (most have no discount)
            discount_prob = np.random.random()
            if discount_prob < 0.7:  # 70% no discount
                discount_percent = 0.0
            elif discount_prob < 0.9:  # 20% small discount
                discount_percent = round(np.random.uniform(0.05, 0.15), 3)
            else:  # 10% larger discount
                discount_percent = round(np.random.uniform(0.15, 0.4), 3)
            
            # Generate sale date
            days_back = np.random.randint(0, 365)
            sale_date = start_date + timedelta(days=days_back)
            
            # Generate customer email
            customer_id = i % 1000  # Simulate returning customers
            email = f"customer_{customer_id:04d}@example.com"
            
            record = {
                'order_id': f'ORD_{i:08d}',
                'product_name': product,
                'category': category,
                'quantity': quantity,
                'unit_price': unit_price,
                'discount_percent': discount_percent,
                'region': np.random.choice(self.regions),
                'sale_date': sale_date,
                'customer_email': email
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
        
        return df
    
    def generate_dirty_sample(self, size: int, corruption_rate: float = 0.2) -> pd.DataFrame:
        """Generate sample data with various quality issues."""
        # Start with clean data
        clean_data = self.generate_clean_sample(size)
        
        # Introduce various data quality issues
        dirty_data = clean_data.copy()
        n_corruptions = int(size * corruption_rate)
        
        corruption_indices = np.random.choice(size, n_corruptions, replace=False)
        
        for idx in corruption_indices:
            corruption_type = np.random.choice([
                'missing_product_name', 'invalid_quantity', 'negative_price',
                'invalid_discount', 'malformed_email', 'invalid_date',
                'empty_region', 'duplicate_order_id', 'extreme_values'
            ])
            
            if corruption_type == 'missing_product_name':
                dirty_data.loc[idx, 'product_name'] = np.random.choice(['', None, 'N/A'])
            
            elif corruption_type == 'invalid_quantity':
                dirty_data.loc[idx, 'quantity'] = np.random.choice([0, -1, 'abc', '', None, '10.5'])
            
            elif corruption_type == 'negative_price':
                dirty_data.loc[idx, 'unit_price'] = np.random.choice([-10.5, 0, 'invalid', ''])
            
            elif corruption_type == 'invalid_discount':
                dirty_data.loc[idx, 'discount_percent'] = np.random.choice([1.5, -0.1, 'bad', '150%'])
            
            elif corruption_type == 'malformed_email':
                dirty_data.loc[idx, 'customer_email'] = np.random.choice([
                    'invalid.email', 'user@', '@domain.com', 'user@domain',
                    'user..name@domain.com', '', None
                ])
            
            elif corruption_type == 'invalid_date':
                dirty_data.loc[idx, 'sale_date'] = np.random.choice([
                    'invalid_date', '2023-13-01', '2023-02-30', '', None
                ])
            
            elif corruption_type == 'empty_region':
                dirty_data.loc[idx, 'region'] = np.random.choice(['', None, 'Unknown'])
            
            elif corruption_type == 'duplicate_order_id':
                # Create duplicate order ID
                if idx > 0:
                    dirty_data.loc[idx, 'order_id'] = dirty_data.loc[idx-1, 'order_id']
            
            elif corruption_type == 'extreme_values':
                dirty_data.loc[idx, 'quantity'] = np.random.choice([1000, 50000])
                dirty_data.loc[idx, 'unit_price'] = np.random.choice([0.01, 1000000])
        
        return dirty_data
    
    def generate_anomaly_sample(self, size: int, anomaly_rate: float = 0.05) -> pd.DataFrame:
        """Generate sample data with known anomalous patterns."""
        # Start with clean data
        normal_data = self.generate_clean_sample(size)
        
        # Add specific anomalous patterns
        n_anomalies = int(size * anomaly_rate)
        anomaly_indices = np.random.choice(size, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice([
                'high_value_transaction', 'bulk_purchase', 'extreme_discount',
                'weekend_spike', 'suspicious_email_pattern', 'price_manipulation'
            ])
            
            if anomaly_type == 'high_value_transaction':
                # Extremely high revenue transaction
                normal_data.loc[idx, 'unit_price'] = np.random.uniform(5000, 20000)
                normal_data.loc[idx, 'quantity'] = np.random.randint(1, 5)
                normal_data.loc[idx, 'discount_percent'] = 0.0
            
            elif anomaly_type == 'bulk_purchase':
                # Unusually high quantity
                normal_data.loc[idx, 'quantity'] = np.random.randint(100, 1000)
                normal_data.loc[idx, 'discount_percent'] = np.random.uniform(0.3, 0.6)
            
            elif anomaly_type == 'extreme_discount':
                # Extremely high discount
                normal_data.loc[idx, 'discount_percent'] = np.random.uniform(0.7, 0.95)
                normal_data.loc[idx, 'unit_price'] = np.random.uniform(1000, 5000)
            
            elif anomaly_type == 'weekend_spike':
                # High-value weekend sales
                weekend_date = datetime.now() - timedelta(days=np.random.randint(0, 52) * 7)
                weekend_date = weekend_date.replace(hour=0, minute=0, second=0, microsecond=0)
                weekend_date += timedelta(days=(5 - weekend_date.weekday()) % 7)  # Saturday
                
                normal_data.loc[idx, 'sale_date'] = weekend_date
                normal_data.loc[idx, 'unit_price'] = np.random.uniform(2000, 8000)
                normal_data.loc[idx, 'quantity'] = np.random.randint(5, 20)
            
            elif anomaly_type == 'suspicious_email_pattern':
                # Suspicious email patterns
                suspicious_emails = [
                    'aaa@test.com', 'x@y.z', 'test....test@example.com',
                    'user123456789@temp.com', 'fake@fake.fake'
                ]
                normal_data.loc[idx, 'customer_email'] = np.random.choice(suspicious_emails)
                normal_data.loc[idx, 'unit_price'] = np.random.uniform(1000, 3000)
            
            elif anomaly_type == 'price_manipulation':
                # Suspicious price patterns
                normal_data.loc[idx, 'unit_price'] = 9999.99
                normal_data.loc[idx, 'discount_percent'] = 0.99
                normal_data.loc[idx, 'quantity'] = 1
        
        # Recalculate revenue
        normal_data['revenue'] = (
            normal_data['quantity'] * normal_data['unit_price'] * 
            (1 - normal_data['discount_percent'])
        )
        
        # Add anomaly flags for testing
        normal_data['is_known_anomaly'] = False
        normal_data.loc[anomaly_indices, 'is_known_anomaly'] = True
        
        return normal_data
    
    def generate_time_series_data(self, size: int, days: int = 365) -> pd.DataFrame:
        """Generate time series data with seasonal patterns."""
        start_date = datetime.now() - timedelta(days=days)
        
        data = []
        
        for i in range(size):
            # Generate date with some seasonal patterns
            day_offset = np.random.randint(0, days)
            sale_date = start_date + timedelta(days=day_offset)
            
            # Seasonal effects
            month = sale_date.month
            is_holiday_season = month in [11, 12]  # November, December
            is_summer = month in [6, 7, 8]  # June, July, August
            
            # Weekend effect
            is_weekend = sale_date.weekday() >= 5
            
            # Adjust quantities and prices based on patterns
            base_quantity = np.random.randint(1, 5)
            if is_holiday_season:
                quantity = base_quantity * np.random.randint(2, 4)
            elif is_weekend:
                quantity = base_quantity * np.random.randint(1, 2)
            else:
                quantity = base_quantity
            
            # Category selection with seasonal bias
            if is_holiday_season:
                category = np.random.choice(['Electronics', 'Fashion', 'Home Goods'], 
                                          p=[0.5, 0.3, 0.2])
            elif is_summer:
                category = np.random.choice(['Electronics', 'Fashion', 'Home Goods'], 
                                          p=[0.3, 0.5, 0.2])
            else:
                category = np.random.choice(list(self.products.keys()))
            
            product = np.random.choice(self.products[category])
            price_min, price_max = self.price_ranges[category]
            unit_price = round(np.random.uniform(price_min, price_max), 2)
            
            # Seasonal discount patterns
            if is_holiday_season:
                discount_percent = round(np.random.uniform(0.1, 0.4), 3)
            else:
                discount_percent = round(np.random.uniform(0.0, 0.2), 3)
            
            record = {
                'order_id': f'ORD_{i:08d}',
                'product_name': product,
                'category': category,
                'quantity': quantity,
                'unit_price': unit_price,
                'discount_percent': discount_percent,
                'region': np.random.choice(self.regions),
                'sale_date': sale_date,
                'customer_email': f"customer_{i % 2000:04d}@example.com"
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
        
        return df
    
    def generate_multi_scenario_dataset(self, scenarios: Dict[str, int]) -> pd.DataFrame:
        """Generate dataset with multiple scenarios combined."""
        datasets = []
        
        for scenario, count in scenarios.items():
            if scenario == 'clean':
                data = self.generate_clean_sample(count)
            elif scenario == 'dirty':
                data = self.generate_dirty_sample(count)
            elif scenario == 'anomaly':
                data = self.generate_anomaly_sample(count)
            elif scenario == 'time_series':
                data = self.generate_time_series_data(count)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
            
            data['data_scenario'] = scenario
            datasets.append(data)
        
        # Combine all datasets
        combined_data = pd.concat(datasets, ignore_index=True)
        
        # Shuffle the data
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
        
        # Regenerate order IDs to ensure uniqueness
        combined_data['order_id'] = [f'ORD_{i:08d}' for i in range(len(combined_data))]
        
        return combined_data


class TestDataGeneratorTests:
    """Test the test data generator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = TestDataGenerator(seed=42)
    
    def test_generate_clean_sample(self):
        """Test clean sample generation."""
        data = self.generator.generate_clean_sample(1000)
        
        # Basic structure tests
        assert len(data) == 1000
        assert 'order_id' in data.columns
        assert 'product_name' in data.columns
        assert 'revenue' in data.columns
        
        # Data quality tests
        assert data['order_id'].nunique() == 1000  # All unique
        assert not data['product_name'].isna().any()  # No missing product names
        assert (data['quantity'] > 0).all()  # All positive quantities
        assert (data['unit_price'] > 0).all()  # All positive prices
        assert (data['discount_percent'] >= 0).all()  # Non-negative discounts
        assert (data['discount_percent'] <= 1).all()  # Discounts <= 100%
        assert (data['revenue'] >= 0).all()  # Non-negative revenue
    
    def test_generate_dirty_sample(self):
        """Test dirty sample generation."""
        data = self.generator.generate_dirty_sample(1000, corruption_rate=0.3)
        
        # Should have same structure
        assert len(data) == 1000
        
        # Should have some data quality issues
        has_missing_products = data['product_name'].isna().any() or (data['product_name'] == '').any()
        has_invalid_quantities = (data['quantity'] <= 0).any() if data['quantity'].dtype in ['int64', 'float64'] else True
        has_negative_prices = (data['unit_price'] <= 0).any() if data['unit_price'].dtype in ['int64', 'float64'] else True
        
        # At least some corruption should be present
        assert has_missing_products or has_invalid_quantities or has_negative_prices
    
    def test_generate_anomaly_sample(self):
        """Test anomaly sample generation."""
        data = self.generator.generate_anomaly_sample(1000, anomaly_rate=0.1)
        
        # Should have anomaly flags
        assert 'is_known_anomaly' in data.columns
        assert data['is_known_anomaly'].sum() > 0  # Should have some anomalies
        assert data['is_known_anomaly'].sum() <= 100  # Should not exceed expected rate
        
        # Anomalies should have extreme values
        anomaly_data = data[data['is_known_anomaly']]
        if len(anomaly_data) > 0:
            # Should have some extreme values
            has_high_revenue = (anomaly_data['revenue'] > 10000).any()
            has_high_quantity = (anomaly_data['quantity'] > 50).any()
            has_high_discount = (anomaly_data['discount_percent'] > 0.5).any()
            
            assert has_high_revenue or has_high_quantity or has_high_discount
    
    def test_generate_time_series_data(self):
        """Test time series data generation."""
        data = self.generator.generate_time_series_data(1000, days=365)
        
        # Should span the expected time range
        date_range = data['sale_date'].max() - data['sale_date'].min()
        assert date_range.days >= 300  # Should cover most of the year
        
        # Should have seasonal patterns (hard to test directly, but check structure)
        assert len(data) == 1000
        assert 'sale_date' in data.columns
        assert data['sale_date'].dtype == 'datetime64[ns]'
    
    def test_generate_multi_scenario_dataset(self):
        """Test multi-scenario dataset generation."""
        scenarios = {
            'clean': 500,
            'dirty': 300,
            'anomaly': 200
        }
        
        data = self.generator.generate_multi_scenario_dataset(scenarios)
        
        # Should have correct total size
        assert len(data) == 1000
        
        # Should have scenario labels
        assert 'data_scenario' in data.columns
        assert set(data['data_scenario'].unique()) == set(scenarios.keys())
        
        # Should have approximately correct proportions
        scenario_counts = data['data_scenario'].value_counts()
        assert scenario_counts['clean'] == 500
        assert scenario_counts['dirty'] == 300
        assert scenario_counts['anomaly'] == 200
    
    def test_data_consistency(self):
        """Test data consistency across different generation methods."""
        # Generate same data with same seed
        gen1 = TestDataGenerator(seed=123)
        gen2 = TestDataGenerator(seed=123)
        
        data1 = gen1.generate_clean_sample(100)
        data2 = gen2.generate_clean_sample(100)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_realistic_data_patterns(self):
        """Test that generated data follows realistic patterns."""
        data = self.generator.generate_clean_sample(10000)
        
        # Most orders should be small quantities
        small_orders = (data['quantity'] <= 5).sum()
        assert small_orders / len(data) > 0.7  # At least 70% small orders
        
        # Most orders should have no discount
        no_discount = (data['discount_percent'] == 0).sum()
        assert no_discount / len(data) > 0.6  # At least 60% no discount
        
        # Price ranges should be reasonable by category
        electronics = data[data['category'] == 'Electronics']
        fashion = data[data['category'] == 'Fashion']
        home_goods = data[data['category'] == 'Home Goods']
        
        if len(electronics) > 0:
            assert electronics['unit_price'].min() >= 50
            assert electronics['unit_price'].max() <= 3000
        
        if len(fashion) > 0:
            assert fashion['unit_price'].min() >= 20
            assert fashion['unit_price'].max() <= 500
        
        if len(home_goods) > 0:
            assert home_goods['unit_price'].min() >= 30
            assert home_goods['unit_price'].max() <= 800


if __name__ == '__main__':
    pytest.main([__file__, '-v'])