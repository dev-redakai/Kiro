"""
Unit tests for ValidationEngine class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality.validation_engine import ValidationEngine


class TestValidationEngine:
    """Test cases for ValidationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_detect_duplicate_order_ids_basic(self):
        """Test basic duplicate order ID detection."""
        data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD001', 'ORD003'],
            'product_name': ['Product A', 'Product B', 'Product C', 'Product D']
        })
        
        result = self.engine.detect_duplicate_order_ids(data)
        
        # Check that duplicate flag is added
        assert 'is_duplicate' in result.columns
        
        # Check that duplicates are correctly identified
        expected_duplicates = [True, False, True, False]
        assert result['is_duplicate'].tolist() == expected_duplicates
        
        # Check statistics
        stats = self.engine.get_validation_stats()
        assert stats['duplicate_records'] == 2
        assert stats['unique_duplicate_order_ids'] == 1
    
    def test_detect_duplicate_order_ids_no_duplicates(self):
        """Test duplicate detection with no duplicates."""
        data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD003'],
            'product_name': ['Product A', 'Product B', 'Product C']
        })
        
        result = self.engine.detect_duplicate_order_ids(data)
        
        # Check that no duplicates are found
        assert not result['is_duplicate'].any()
        
        # Check statistics
        stats = self.engine.get_validation_stats()
        assert stats['duplicate_records'] == 0
    
    def test_detect_duplicate_order_ids_missing_column(self):
        """Test duplicate detection with missing order_id column."""
        data = pd.DataFrame({
            'product_name': ['Product A', 'Product B', 'Product C']
        })
        
        result = self.engine.detect_duplicate_order_ids(data)
        
        # Should return original data unchanged
        assert 'is_duplicate' not in result.columns
        pd.testing.assert_frame_equal(result, data)
    
    def test_calculate_revenue_basic(self):
        """Test basic revenue calculation."""
        data = pd.DataFrame({
            'quantity': [2, 3, 1],
            'unit_price': [10.0, 15.0, 20.0],
            'discount_percent': [0.1, 0.0, 0.2]
        })
        
        result = self.engine.calculate_revenue(data)
        
        # Check that revenue column is added
        assert 'revenue' in result.columns
        
        # Check revenue calculations
        expected_revenue = [18.0, 45.0, 16.0]  # 2*10*0.9, 3*15*1.0, 1*20*0.8
        assert result['revenue'].tolist() == expected_revenue
        
        # Check statistics
        stats = self.engine.get_validation_stats()
        assert stats['revenue_calculations_performed'] == 3
    
    def test_calculate_revenue_missing_columns(self):
        """Test revenue calculation with missing columns."""
        data = pd.DataFrame({
            'quantity': [2, 3],
            'unit_price': [10.0, 15.0]
            # Missing discount_percent
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.engine.calculate_revenue(data)
    
    def test_calculate_revenue_invalid_values(self):
        """Test revenue calculation with invalid values."""
        data = pd.DataFrame({
            'quantity': [2, np.inf, 1],
            'unit_price': [10.0, 15.0, np.nan],
            'discount_percent': [0.1, 0.0, 0.2]
        })
        
        result = self.engine.calculate_revenue(data)
        
        # Check that invalid revenues are set to 0
        assert result['revenue'].iloc[1] == 0.0  # inf case
        assert result['revenue'].iloc[2] == 0.0  # nan case
        assert result['revenue'].iloc[0] == 18.0  # valid case
    
    def test_calculate_revenue_negative_values(self):
        """Test revenue calculation with negative results."""
        data = pd.DataFrame({
            'quantity': [2, 1],
            'unit_price': [10.0, 15.0],
            'discount_percent': [1.5, 0.0]  # Discount > 100%
        })
        
        result = self.engine.calculate_revenue(data)
        
        # Negative revenue should be set to 0
        assert result['revenue'].iloc[0] == 0.0
        assert result['revenue'].iloc[1] == 15.0
    
    def test_validate_data_quality_basic(self):
        """Test basic data quality validation."""
        data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD003'],
            'product_name': ['Product A', 'Product B', 'Product C'],
            'category': ['Electronics', 'Fashion', 'Home Goods'],
            'quantity': [2, 3, 1],
            'unit_price': [10.0, 15.0, 20.0],
            'discount_percent': [0.1, 0.0, 0.2],
            'region': ['North', 'South', 'East'],
            'sale_date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            'customer_email': ['user@test.com', 'user2@test.com', None]
        })
        
        results = self.engine.validate_data_quality(data)
        
        # Check basic structure
        assert 'total_records' in results
        assert 'valid_records' in results
        assert 'field_validation_results' in results
        assert 'overall_quality_score' in results
        
        # Check that all required fields are validated
        required_fields = ['order_id', 'product_name', 'category', 'quantity', 
                          'unit_price', 'discount_percent', 'region', 'sale_date']
        for field in required_fields:
            assert field in results['field_validation_results']
        
        # Check quality score is reasonable
        assert 0.0 <= results['overall_quality_score'] <= 1.0
    
    def test_validate_data_quality_with_errors(self):
        """Test data quality validation with various errors."""
        data = pd.DataFrame({
            'order_id': ['ORD001', '', 'ORD@003'],  # Empty and invalid pattern
            'product_name': ['Product A', None, 'B'],  # Null value
            'quantity': [2, -1, 15000],  # Negative and too high
            'unit_price': [10.0, 0.0, 2000000.0],  # Zero and too high
            'discount_percent': [0.1, 1.5, -0.1],  # Over 100% and negative
            'region': ['North', 'S', ''],  # Too short and empty
            'sale_date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            'customer_email': ['user@test.com', 'invalid.email', None]
        })
        
        results = self.engine.validate_data_quality(data)
        
        # Should have validation errors
        assert results['overall_quality_score'] < 1.0
        assert results['valid_records'] < results['total_records']
        
        # Check specific field errors
        assert results['field_validation_results']['order_id']['invalid_values'] > 0
        assert results['field_validation_results']['quantity']['invalid_values'] > 0
        assert results['field_validation_results']['unit_price']['invalid_values'] > 0
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD003'],
            'product_name': ['Product A', 'Product B', None],
            'quantity': [2, 3, 1],
            'unit_price': [10.0, 15.0, 20.0],
            'discount_percent': [0.1, 0.0, 0.2],
            'region': ['North', 'South', 'East'],
            'sale_date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        })
        
        validation_results = self.engine.validate_data_quality(data)
        report = self.engine.generate_validation_report(validation_results)
        
        # Check report structure
        assert 'summary' in report
        assert 'field_analysis' in report
        assert 'data_quality_issues' in report
        assert 'recommendations' in report
        
        # Check summary
        assert report['summary']['total_records_validated'] == 3
        assert 'overall_quality_score' in report['summary']
        
        # Check field analysis
        for field_name in validation_results['field_validation_results']:
            assert field_name in report['field_analysis']
            field_analysis = report['field_analysis'][field_name]
            assert 'validity_rate' in field_analysis
            assert 'null_rate' in field_analysis
    
    def test_validate_chunk_integration(self):
        """Test complete chunk validation integration."""
        chunk = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD001'],  # Has duplicate
            'product_name': ['Product A', 'Product B', 'Product C'],
            'quantity': [2, 3, 1],
            'unit_price': [10.0, 15.0, 20.0],
            'discount_percent': [0.1, 0.0, 0.2],
            'region': ['North', 'South', 'East'],
            'sale_date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        })
        
        result_chunk, validation_results = self.engine.validate_chunk(chunk)
        
        # Check that all operations were performed
        assert 'is_duplicate' in result_chunk.columns
        assert 'revenue' in result_chunk.columns
        assert validation_results['total_records'] == 3
        
        # Check duplicate detection
        assert result_chunk['is_duplicate'].sum() == 2  # Two duplicate records
        
        # Check revenue calculation
        assert all(result_chunk['revenue'] > 0)
    
    def test_get_validation_stats(self):
        """Test validation statistics retrieval."""
        data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD001'],
            'quantity': [2, 3, 1],
            'unit_price': [10.0, 15.0, 20.0],
            'discount_percent': [0.1, 0.0, 0.2]
        })
        
        # Perform operations
        self.engine.detect_duplicate_order_ids(data)
        self.engine.calculate_revenue(data)
        self.engine.validate_data_quality(data)
        
        stats = self.engine.get_validation_stats()
        
        # Check statistics
        assert stats['total_records_processed'] == 3
        assert stats['duplicate_records'] == 2
        assert stats['revenue_calculations_performed'] == 3
        assert stats['unique_duplicate_order_ids'] == 1
        assert 'validation_success_rate' in stats
    
    def test_reset_stats(self):
        """Test statistics reset functionality."""
        data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002'],
            'quantity': [2, 3],
            'unit_price': [10.0, 15.0],
            'discount_percent': [0.1, 0.0]
        })
        
        # Perform operations
        self.engine.detect_duplicate_order_ids(data)
        self.engine.calculate_revenue(data)
        self.engine.validate_data_quality(data)  # This updates total_records
        
        stats_before = self.engine.get_validation_stats()
        self.engine.reset_stats()
        stats_after = self.engine.get_validation_stats()
        
        # Check that stats were reset
        assert stats_before['total_records_processed'] > 0
        assert stats_after['total_records_processed'] == 0
        assert stats_after['duplicate_records'] == 0
        assert stats_after['revenue_calculations_performed'] == 0


if __name__ == '__main__':
    pytest.main([__file__])