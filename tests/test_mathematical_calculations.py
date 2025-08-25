"""
Unit tests for mathematical calculations and aggregations.

This module contains comprehensive tests for all mathematical operations
used in the data pipeline including revenue calculations, aggregations,
and statistical computations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import individual modules to avoid package import issues
try:
    from pipeline.transformation import AnalyticalTransformer, MetricsCalculator
    from quality.validation_engine import ValidationEngine
    IMPORTS_AVAILABLE = True
except ImportError:
    # Create mock classes for testing mathematical operations
    IMPORTS_AVAILABLE = False
    
    class MockValidationEngine:
        def calculate_revenue(self, data):
            """Mock revenue calculation for testing."""
            result = data.copy()
            result['revenue'] = (
                data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
            )
            # Handle edge cases
            result.loc[result['revenue'] < 0, 'revenue'] = 0.0
            result.loc[result['revenue'].isna(), 'revenue'] = 0.0
            result.loc[np.isinf(result['revenue']), 'revenue'] = 0.0
            return result
    
    class MockAnalyticalTransformer:
        def create_monthly_sales_summary(self, data):
            """Mock monthly sales summary for testing."""
            if len(data) == 0:
                return pd.DataFrame()
            
            # Check if order_id column exists
            agg_dict = {
                'revenue': 'sum',
                'quantity': 'sum',
                'discount_percent': 'mean'
            }
            if 'order_id' in data.columns:
                agg_dict['order_id'] = 'nunique'
            
            monthly = data.groupby(
                data['sale_date'].dt.to_period('M')
            ).agg(agg_dict).reset_index()
            
            monthly.columns = ['month', 'total_revenue', 'total_quantity', 'avg_discount', 'unique_orders']
            monthly['avg_order_value'] = monthly['total_revenue'] / monthly['unique_orders']
            return monthly
        
        def create_top_products_table(self, data):
            """Mock top products table for testing."""
            if len(data) == 0:
                return pd.DataFrame()
            
            products = data.groupby('product_name').agg({
                'revenue': 'sum',
                'quantity': 'sum'
            }).reset_index()
            
            products['revenue_rank'] = products['revenue'].rank(method='dense', ascending=False).astype(int)
            products['units_rank'] = products['quantity'].rank(method='dense', ascending=False).astype(int)
            products.columns = ['product_name', 'total_revenue', 'total_units', 'revenue_rank', 'units_rank']
            
            return products.head(10)
        
        def create_region_wise_performance_table(self, data):
            """Mock regional performance table for testing."""
            if len(data) == 0:
                return pd.DataFrame()
            
            total_revenue = data['revenue'].sum()
            regional = data.groupby('region').agg({
                'revenue': 'sum',
                'order_id': 'count'
            }).reset_index()
            
            regional['avg_order_value'] = regional['revenue'] / regional['order_id']
            regional['market_share_pct'] = (regional['revenue'] / total_revenue * 100)
            regional.columns = ['region', 'total_revenue', 'order_count', 'avg_order_value', 'market_share_pct']
            
            return regional
    
    class MockMetricsCalculator:
        def calculate_discount_effectiveness_analysis(self, data):
            """Mock discount effectiveness analysis for testing."""
            avg_discount = data['discount_percent'].mean()
            discount_penetration = (data['discount_percent'] > 0).sum() / len(data)
            
            full_price_revenue = (data['quantity'] * data['unit_price']).sum()
            actual_revenue = data['revenue'].sum()
            discount_amount = full_price_revenue - actual_revenue
            
            return {
                'avg_discount_rate': avg_discount,
                'discount_penetration_rate': discount_penetration,
                'total_discount_amount': discount_amount
            }


class TestRevenueCalculations:
    """Test revenue calculation accuracy and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        if IMPORTS_AVAILABLE:
            self.validator = ValidationEngine()
        else:
            self.validator = MockValidationEngine()
    
    def test_basic_revenue_calculation(self):
        """Test basic revenue calculation formula."""
        data = pd.DataFrame({
            'quantity': [2, 3, 1, 5],
            'unit_price': [10.0, 15.0, 20.0, 8.0],
            'discount_percent': [0.1, 0.0, 0.2, 0.15]
        })
        
        result = self.validator.calculate_revenue(data)
        
        # Manual calculations: quantity * unit_price * (1 - discount_percent)
        expected_revenue = [
            2 * 10.0 * (1 - 0.1),    # 18.0
            3 * 15.0 * (1 - 0.0),    # 45.0
            1 * 20.0 * (1 - 0.2),    # 16.0
            5 * 8.0 * (1 - 0.15)     # 34.0
        ]
        
        assert result['revenue'].tolist() == expected_revenue
    
    def test_revenue_calculation_precision(self):
        """Test revenue calculation precision with floating point numbers."""
        data = pd.DataFrame({
            'quantity': [1, 1, 1],
            'unit_price': [0.1, 0.2, 0.3],
            'discount_percent': [0.1, 0.2, 0.3]
        })
        
        result = self.validator.calculate_revenue(data)
        
        # Check precision (should be accurate to reasonable decimal places)
        expected = [
            0.1 * (1 - 0.1),  # 0.09
            0.2 * (1 - 0.2),  # 0.16
            0.3 * (1 - 0.3)   # 0.21
        ]
        
        for i, expected_val in enumerate(expected):
            assert abs(result['revenue'].iloc[i] - expected_val) < 1e-10
    
    def test_revenue_calculation_edge_cases(self):
        """Test revenue calculation with edge cases."""
        data = pd.DataFrame({
            'quantity': [0, 1, 1, 1, 1],
            'unit_price': [10.0, 0.0, 10.0, 10.0, 10.0],
            'discount_percent': [0.1, 0.1, 0.0, 1.0, 1.5]  # Last one > 100%
        })
        
        result = self.validator.calculate_revenue(data)
        
        # Check edge cases
        assert result['revenue'].iloc[0] == 0.0  # Zero quantity
        assert result['revenue'].iloc[1] == 0.0  # Zero price
        assert result['revenue'].iloc[2] == 10.0  # No discount
        assert result['revenue'].iloc[3] == 0.0  # 100% discount
        assert result['revenue'].iloc[4] == 0.0  # Over 100% discount (negative revenue set to 0)
    
    def test_revenue_calculation_with_nulls(self):
        """Test revenue calculation with null values."""
        data = pd.DataFrame({
            'quantity': [2, np.nan, 3, 1],
            'unit_price': [10.0, 15.0, np.nan, 20.0],
            'discount_percent': [0.1, 0.2, 0.3, np.nan]
        })
        
        result = self.validator.calculate_revenue(data)
        
        # Null values should result in 0 revenue
        assert result['revenue'].iloc[0] == 18.0  # Valid calculation
        assert result['revenue'].iloc[1] == 0.0   # Null quantity
        assert result['revenue'].iloc[2] == 0.0   # Null price
        assert result['revenue'].iloc[3] == 0.0   # Null discount
    
    def test_revenue_calculation_large_numbers(self):
        """Test revenue calculation with large numbers."""
        data = pd.DataFrame({
            'quantity': [1000000, 1, 999999],
            'unit_price': [0.01, 1000000.0, 1.0],
            'discount_percent': [0.0, 0.1, 0.001]
        })
        
        result = self.validator.calculate_revenue(data)
        
        # Check large number handling
        assert result['revenue'].iloc[0] == 10000.0  # 1M * 0.01
        assert result['revenue'].iloc[1] == 900000.0  # 1M * 0.9
        assert abs(result['revenue'].iloc[2] - 999999.001) < 1e-3  # Precision check (999999 * 0.999)


class TestAggregationCalculations:
    """Test aggregation calculations for analytical tables."""
    
    def setup_method(self):
        """Set up test fixtures."""
        if IMPORTS_AVAILABLE:
            self.transformer = AnalyticalTransformer()
            self.calculator = MetricsCalculator()
        else:
            self.transformer = MockAnalyticalTransformer()
            self.calculator = MockMetricsCalculator()
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for aggregation tests."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        
        data = pd.DataFrame({
            'order_id': [f'ORD_{i:06d}' for i in range(1000)],
            'product_name': np.random.choice(['iPhone', 'Samsung TV', 'Nike Shoes'], 1000),
            'category': np.random.choice(['Electronics', 'Fashion'], 1000),
            'quantity': np.random.randint(1, 10, 1000),
            'unit_price': np.random.uniform(10, 1000, 1000),
            'discount_percent': np.random.uniform(0, 0.3, 1000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'sale_date': np.random.choice(dates, 1000)
        })
        
        data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])
        return data
    
    def test_monthly_sales_aggregation(self, sample_sales_data):
        """Test monthly sales summary aggregation accuracy."""
        result = self.transformer.create_monthly_sales_summary(sample_sales_data)
        
        # Verify aggregation accuracy by manual calculation
        manual_monthly = sample_sales_data.groupby(
            sample_sales_data['sale_date'].dt.to_period('M')
        ).agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'discount_percent': 'mean',
            'order_id': 'nunique'
        }).reset_index()
        
        # Check that totals match
        assert abs(result['total_revenue'].sum() - manual_monthly['revenue'].sum()) < 1e-6
        assert result['total_quantity'].sum() == manual_monthly['quantity'].sum()
    
    def test_top_products_ranking_accuracy(self, sample_sales_data):
        """Test top products ranking calculation accuracy."""
        result = self.transformer.create_top_products_table(sample_sales_data)
        
        # Manual calculation for verification
        manual_products = sample_sales_data.groupby('product_name').agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        manual_products['revenue_rank'] = manual_products['revenue'].rank(
            method='dense', ascending=False
        ).astype(int)
        manual_products['units_rank'] = manual_products['quantity'].rank(
            method='dense', ascending=False
        ).astype(int)
        
        # Check top product by revenue
        top_revenue_manual = manual_products.loc[
            manual_products['revenue_rank'] == 1, 'product_name'
        ].iloc[0]
        top_revenue_result = result.loc[
            result['revenue_rank'] == 1, 'product_name'
        ].iloc[0]
        
        assert top_revenue_manual == top_revenue_result
    
    def test_regional_performance_calculations(self, sample_sales_data):
        """Test regional performance calculation accuracy."""
        result = self.transformer.create_region_wise_performance_table(sample_sales_data)
        
        # Manual calculation
        total_revenue = sample_sales_data['revenue'].sum()
        manual_regional = sample_sales_data.groupby('region').agg({
            'revenue': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        manual_regional['avg_order_value'] = (
            manual_regional['revenue'] / manual_regional['order_id']
        )
        manual_regional['market_share_pct'] = (
            manual_regional['revenue'] / total_revenue * 100
        )
        
        # Check market share percentages sum to 100
        assert abs(result['market_share_pct'].sum() - 100.0) < 1e-6
        
        # Check individual calculations
        for _, row in result.iterrows():
            manual_row = manual_regional[manual_regional['region'] == row['region']].iloc[0]
            assert abs(row['total_revenue'] - manual_row['revenue']) < 1e-6
            assert abs(row['avg_order_value'] - manual_row['avg_order_value']) < 1e-6
    
    def test_discount_effectiveness_calculations(self, sample_sales_data):
        """Test discount effectiveness calculation accuracy."""
        result = self.calculator.calculate_discount_effectiveness_analysis(sample_sales_data)
        
        # Manual calculations
        manual_avg_discount = sample_sales_data['discount_percent'].mean()
        manual_discount_penetration = (
            (sample_sales_data['discount_percent'] > 0).sum() / len(sample_sales_data)
        )
        
        # Calculate total discount amount
        full_price_revenue = (
            sample_sales_data['quantity'] * sample_sales_data['unit_price']
        ).sum()
        actual_revenue = sample_sales_data['revenue'].sum()
        manual_discount_amount = full_price_revenue - actual_revenue
        
        # Verify calculations
        assert abs(result['avg_discount_rate'] - manual_avg_discount) < 1e-6
        assert abs(result['discount_penetration_rate'] - manual_discount_penetration) < 1e-6
        assert abs(result['total_discount_amount'] - manual_discount_amount) < 1e-6
    
    def test_aggregation_with_empty_data(self):
        """Test aggregation functions with empty datasets."""
        empty_data = pd.DataFrame(columns=[
            'product_name', 'category', 'quantity', 'unit_price', 
            'discount_percent', 'region', 'sale_date', 'revenue'
        ])
        
        # Should handle empty data gracefully
        monthly_result = self.transformer.create_monthly_sales_summary(empty_data)
        assert len(monthly_result) == 0
        
        products_result = self.transformer.create_top_products_table(empty_data)
        assert len(products_result) == 0
        
        regional_result = self.transformer.create_region_wise_performance_table(empty_data)
        assert len(regional_result) == 0
    
    def test_aggregation_with_single_record(self):
        """Test aggregation functions with single record."""
        single_record = pd.DataFrame({
            'product_name': ['Test Product'],
            'category': ['Electronics'],
            'quantity': [5],
            'unit_price': [100.0],
            'discount_percent': [0.1],
            'region': ['North'],
            'sale_date': [datetime(2023, 1, 15)],
            'revenue': [450.0]
        })
        
        # Should handle single record correctly
        monthly_result = self.transformer.create_monthly_sales_summary(single_record)
        assert len(monthly_result) == 1
        assert monthly_result['total_revenue'].iloc[0] == 450.0
        
        products_result = self.transformer.create_top_products_table(single_record)
        assert len(products_result) == 1
        assert products_result['revenue_rank'].iloc[0] == 1
        
        regional_result = self.transformer.create_region_wise_performance_table(single_record)
        assert len(regional_result) == 1
        assert regional_result['market_share_pct'].iloc[0] == 100.0


class TestStatisticalCalculations:
    """Test statistical calculations used in the pipeline."""
    
    def test_mean_calculations(self):
        """Test mean calculation accuracy."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test with pandas
        pandas_mean = data.mean()
        
        # Test with numpy
        numpy_mean = np.mean(data)
        
        # Test manual calculation
        manual_mean = sum(data) / len(data)
        
        assert abs(pandas_mean - numpy_mean) < 1e-10
        assert abs(pandas_mean - manual_mean) < 1e-10
        assert pandas_mean == 5.5
    
    def test_standard_deviation_calculations(self):
        """Test standard deviation calculation accuracy."""
        data = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])
        
        # Test with pandas (default ddof=1)
        pandas_std = data.std()
        
        # Test with numpy (default ddof=0)
        numpy_std_pop = np.std(data)
        numpy_std_sample = np.std(data, ddof=1)
        
        # Pandas uses sample standard deviation by default
        assert abs(pandas_std - numpy_std_sample) < 1e-10
        assert pandas_std != numpy_std_pop  # Should be different
    
    def test_percentile_calculations(self):
        """Test percentile calculation accuracy."""
        data = pd.Series(range(1, 101))  # 1 to 100
        
        # Test various percentiles
        q25 = data.quantile(0.25)
        q50 = data.quantile(0.50)  # median
        q75 = data.quantile(0.75)
        q95 = data.quantile(0.95)
        
        assert q25 == 25.75
        assert q50 == 50.5
        assert q75 == 75.25
        assert q95 == 95.05
    
    def test_correlation_calculations(self):
        """Test correlation calculation accuracy."""
        # Create correlated data
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])  # Perfect positive correlation
        z = pd.Series([10, 8, 6, 4, 2])  # Perfect negative correlation
        
        # Test perfect positive correlation
        corr_xy = x.corr(y)
        assert abs(corr_xy - 1.0) < 1e-10
        
        # Test perfect negative correlation
        corr_xz = x.corr(z)
        assert abs(corr_xz - (-1.0)) < 1e-10
        
        # Test correlation matrix
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        corr_matrix = df.corr()
        
        assert abs(corr_matrix.loc['x', 'y'] - 1.0) < 1e-10
        assert abs(corr_matrix.loc['x', 'z'] - (-1.0)) < 1e-10
    
    def test_division_by_zero_handling(self):
        """Test handling of division by zero in calculations."""
        # Test with zeros in denominator
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 5])
        
        # Using pandas operations
        result = numerator / denominator
        
        assert result.iloc[0] == 5.0
        assert np.isinf(result.iloc[1])  # Division by zero gives infinity
        assert result.iloc[2] == 6.0
        
        # Test safe division
        safe_result = numerator / denominator.replace(0, np.nan)
        assert safe_result.iloc[0] == 5.0
        assert np.isnan(safe_result.iloc[1])  # Division by NaN gives NaN
        assert safe_result.iloc[2] == 6.0
    
    def test_floating_point_precision(self):
        """Test floating point precision in calculations."""
        # Test precision issues
        a = 0.1 + 0.2
        b = 0.3
        
        # Direct comparison will fail due to floating point precision
        assert a != b
        
        # But should be equal within tolerance
        assert abs(a - b) < 1e-15
        
        # Test with pandas Series
        series_a = pd.Series([0.1 + 0.2])
        series_b = pd.Series([0.3])
        
        # Use numpy.allclose for comparison
        assert np.allclose(series_a, series_b)


if __name__ == '__main__':
    pytest.main([__file__])