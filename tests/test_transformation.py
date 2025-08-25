"""
Unit tests for the analytical transformation engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.transformation import AnalyticalTransformer, MetricsCalculator


class TestAnalyticalTransformer:
    """Test cases for AnalyticalTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample e-commerce data for testing."""
        np.random.seed(42)
        n_records = 1000
        
        # Generate sample data
        data = {
            'order_id': [f'ORD_{i:06d}' for i in range(n_records)],
            'product_name': np.random.choice(['iPhone 13', 'Samsung TV', 'Nike Shoes', 'Dell Laptop', 'Sony Headphones'], n_records),
            'category': np.random.choice(['Electronics', 'Fashion', 'Home Goods'], n_records),
            'quantity': np.random.randint(1, 10, n_records),
            'unit_price': np.random.uniform(10, 1000, n_records),
            'discount_percent': np.random.uniform(0, 0.3, n_records),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
            'sale_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)],
            'customer_email': [f'customer_{i}@example.com' for i in range(n_records)]
        }
        
        df = pd.DataFrame(data)
        df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
        
        return df
    
    @pytest.fixture
    def transformer(self):
        """Create AnalyticalTransformer instance."""
        return AnalyticalTransformer()
    
    def test_create_monthly_sales_summary(self, transformer, sample_data):
        """Test monthly sales summary generation."""
        result = transformer.create_monthly_sales_summary(sample_data)
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check required columns exist
        required_cols = ['month', 'total_revenue', 'total_quantity', 'avg_discount', 'unique_orders', 'avg_order_value']
        for col in required_cols:
            assert col in result.columns
        
        # Check that we have data
        assert len(result) > 0
        
        # Check that revenue values are positive
        assert all(result['total_revenue'] >= 0)
        
        # Check that average discount is between 0 and 1
        assert all((result['avg_discount'] >= 0) & (result['avg_discount'] <= 1))
    
    def test_create_top_products_table(self, transformer, sample_data):
        """Test top products table generation."""
        result = transformer.create_top_products_table(sample_data)
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check required columns exist
        required_cols = ['product_name', 'total_revenue', 'total_units', 'revenue_rank', 'units_rank']
        for col in required_cols:
            assert col in result.columns
        
        # Check that we have at most 10 products (or fewer if less unique products)
        unique_products = sample_data['product_name'].nunique()
        expected_max = min(10, unique_products)
        assert len(result) <= expected_max
        
        # Check that rankings are valid
        assert all(result['revenue_rank'] >= 1)
        assert all(result['units_rank'] >= 1)
    
    def test_create_region_wise_performance_table(self, transformer, sample_data):
        """Test regional performance table generation."""
        result = transformer.create_region_wise_performance_table(sample_data)
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check required columns exist
        required_cols = ['region', 'total_revenue', 'avg_order_value', 'market_share_pct']
        for col in required_cols:
            assert col in result.columns
        
        # Check that we have data for each region
        unique_regions = sample_data['region'].nunique()
        assert len(result) == unique_regions
        
        # Check that market share percentages sum to approximately 100
        assert abs(result['market_share_pct'].sum() - 100.0) < 0.1
    
    def test_create_category_discount_map(self, transformer, sample_data):
        """Test category discount map generation."""
        result = transformer.create_category_discount_map(sample_data)
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check required columns exist
        required_cols = ['category', 'avg_discount', 'total_revenue', 'discount_strategy']
        for col in required_cols:
            assert col in result.columns
        
        # Check that we have data for each category
        unique_categories = sample_data['category'].nunique()
        assert len(result) == unique_categories
        
        # Check that discount values are valid
        assert all((result['avg_discount'] >= 0) & (result['avg_discount'] <= 1))


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample e-commerce data for testing."""
        np.random.seed(42)
        n_records = 500
        
        data = {
            'order_id': [f'ORD_{i:06d}' for i in range(n_records)],
            'quantity': np.random.randint(1, 10, n_records),
            'unit_price': np.random.uniform(10, 1000, n_records),
            'discount_percent': np.random.uniform(0, 0.3, n_records),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
            'category': np.random.choice(['Electronics', 'Fashion', 'Home Goods'], n_records),
            'sale_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)],
            'customer_email': [f'customer_{i}@example.com' for i in range(n_records)]
        }
        
        df = pd.DataFrame(data)
        df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
        
        return df
    
    @pytest.fixture
    def calculator(self):
        """Create MetricsCalculator instance."""
        return MetricsCalculator()
    
    def test_calculate_revenue_metrics(self, calculator, sample_data):
        """Test revenue metrics calculation."""
        result = calculator.calculate_revenue_metrics(sample_data)
        
        # Check that result is a dictionary
        assert isinstance(result, dict)
        
        # Check required metrics exist
        required_metrics = ['total_revenue', 'avg_revenue_per_transaction', 'total_transactions']
        for metric in required_metrics:
            assert metric in result
        
        # Check that values are reasonable
        assert result['total_revenue'] > 0
        assert result['avg_revenue_per_transaction'] > 0
        assert result['total_transactions'] == len(sample_data)
    
    def test_calculate_discount_effectiveness_analysis(self, calculator, sample_data):
        """Test discount effectiveness analysis."""
        result = calculator.calculate_discount_effectiveness_analysis(sample_data)
        
        # Check that result is a dictionary
        assert isinstance(result, dict)
        
        # Check required metrics exist
        required_metrics = ['avg_discount_rate', 'discount_penetration_rate', 'total_discount_amount']
        for metric in required_metrics:
            assert metric in result
        
        # Check that values are reasonable
        assert 0 <= result['avg_discount_rate'] <= 1
        assert 0 <= result['discount_penetration_rate'] <= 1
        assert result['total_discount_amount'] >= 0
    
    def test_calculate_regional_performance_calculation_logic(self, calculator, sample_data):
        """Test regional performance calculation logic."""
        result = calculator.calculate_regional_performance_calculation_logic(sample_data)
        
        # Check that result is a dictionary
        assert isinstance(result, dict)
        
        # Check that we have data for each region
        unique_regions = sample_data['region'].nunique()
        assert len(result) == unique_regions
        
        # Check that each region has required metrics
        for region, metrics in result.items():
            assert 'total_revenue' in metrics
            assert 'avg_order_value' in metrics
            assert 'market_share' in metrics
            assert metrics['total_revenue'] >= 0
            assert metrics['avg_order_value'] >= 0
            assert 0 <= metrics['market_share'] <= 1
    
    def test_edge_case_handling(self, calculator):
        """Test edge case handling for null values and division by zero."""
        # Create data with null values
        data_with_nulls = pd.DataFrame({
            'quantity': [1, 2, None, 4],
            'unit_price': [10, 20, 30, None],
            'discount_percent': [0.1, None, 0.2, 0.3],
            'region': ['North', 'South', None, 'East']
        })
        
        # Test that methods handle nulls gracefully
        try:
            calculator.calculate_revenue_metrics(data_with_nulls)
            # Should not raise an exception
        except Exception as e:
            pytest.fail(f"Revenue metrics calculation failed with nulls: {str(e)}")
        
        # Check that null handling was tracked
        stats = calculator.get_calculation_stats()
        assert stats['null_values_handled'] > 0


if __name__ == '__main__':
    pytest.main([__file__])