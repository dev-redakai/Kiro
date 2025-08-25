"""
Unit tests for the AnomalyDetector class.

This module contains comprehensive tests for anomaly detection functionality
including revenue, quantity, and discount anomaly detection methods.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from src.quality.anomaly_detector import AnomalyDetector


class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sales data for testing."""
        np.random.seed(42)
        data = {
            'order_id': [f'ORD{i:06d}' for i in range(1000)],
            'product_name': [f'Product {i}' for i in range(1000)],
            'category': np.random.choice(['Electronics', 'Fashion', 'Home'], 1000),
            'quantity': np.random.randint(1, 20, 1000),
            'unit_price': np.random.uniform(10, 1000, 1000),
            'discount_percent': np.random.uniform(0, 0.3, 1000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'sale_date': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'customer_email': [f'user{i}@example.com' for i in range(1000)]
        }
        
        df = pd.DataFrame(data)
        df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
        
        # Add some anomalies
        # High revenue anomalies
        df.loc[0:4, 'revenue'] = [50000, 45000, 60000, 55000, 48000]
        
        # High quantity anomalies
        df.loc[5:9, 'quantity'] = [500, 450, 600, 550, 480]
        
        # High discount anomalies
        df.loc[10:14, 'discount_percent'] = [0.9, 0.85, 0.95, 0.88, 0.92]
        
        return df
    
    @pytest.fixture
    def anomaly_detector(self):
        """Create AnomalyDetector instance for testing."""
        return AnomalyDetector(z_score_threshold=3.0, iqr_multiplier=1.5)
    
    def test_init(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector(z_score_threshold=2.5, iqr_multiplier=2.0)
        
        assert detector.z_score_threshold == 2.5
        assert detector.iqr_multiplier == 2.0
        assert detector.detection_stats['total_records_analyzed'] == 0
        assert detector.detection_stats['revenue_anomalies'] == 0
        assert detector.detection_stats['quantity_anomalies'] == 0
        assert detector.detection_stats['discount_anomalies'] == 0
        assert detector.anomaly_records == []
    
    def test_detect_revenue_anomalies_basic(self, anomaly_detector, sample_data):
        """Test basic revenue anomaly detection."""
        result = anomaly_detector.detect_revenue_anomalies(sample_data)
        
        # Check that anomaly columns are added
        assert 'revenue_anomaly_zscore' in result.columns
        assert 'revenue_anomaly_iqr' in result.columns
        assert 'revenue_anomaly_score' in result.columns
        assert 'revenue_anomaly_reason' in result.columns
        
        # Check that some anomalies are detected
        assert result['revenue_anomaly_zscore'].sum() > 0
        assert result['revenue_anomaly_iqr'].sum() > 0
        
        # Check that statistics are updated
        assert anomaly_detector.detection_stats['revenue_anomalies'] > 0
    
    def test_detect_revenue_anomalies_without_revenue_column(self, anomaly_detector, sample_data):
        """Test revenue anomaly detection when revenue column is missing."""
        # Remove revenue column
        data_without_revenue = sample_data.drop('revenue', axis=1)
        
        result = anomaly_detector.detect_revenue_anomalies(data_without_revenue)
        
        # Check that revenue is calculated
        assert 'revenue' in result.columns
        assert 'revenue_anomaly_zscore' in result.columns
        assert 'revenue_anomaly_iqr' in result.columns
    
    def test_detect_revenue_anomalies_missing_required_columns(self, anomaly_detector):
        """Test revenue anomaly detection with missing required columns."""
        # Create data without required columns
        incomplete_data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002'],
            'product_name': ['Product A', 'Product B']
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            anomaly_detector.detect_revenue_anomalies(incomplete_data)
    
    def test_detect_revenue_anomalies_no_valid_data(self, anomaly_detector):
        """Test revenue anomaly detection with no valid data."""
        # Create data with all null revenues
        invalid_data = pd.DataFrame({
            'quantity': [np.nan, np.nan],
            'unit_price': [np.nan, np.nan],
            'discount_percent': [np.nan, np.nan],
            'revenue': [np.nan, np.nan]
        })
        
        result = anomaly_detector.detect_revenue_anomalies(invalid_data)
        
        # Check that no anomalies are detected
        assert result['revenue_anomaly_zscore'].sum() == 0
        assert result['revenue_anomaly_iqr'].sum() == 0
        assert all(result['revenue_anomaly_reason'] == 'No valid data')
    
    def test_detect_quantity_anomalies_basic(self, anomaly_detector, sample_data):
        """Test basic quantity anomaly detection."""
        result = anomaly_detector.detect_quantity_anomalies(sample_data)
        
        # Check that anomaly columns are added
        assert 'quantity_anomaly_zscore' in result.columns
        assert 'quantity_anomaly_iqr' in result.columns
        assert 'quantity_anomaly_score' in result.columns
        assert 'quantity_anomaly_reason' in result.columns
        
        # Check that some anomalies are detected (we added high quantities)
        assert result['quantity_anomaly_zscore'].sum() > 0
        assert result['quantity_anomaly_iqr'].sum() > 0
        
        # Check that statistics are updated
        assert anomaly_detector.detection_stats['quantity_anomalies'] > 0
    
    def test_detect_quantity_anomalies_missing_column(self, anomaly_detector):
        """Test quantity anomaly detection with missing quantity column."""
        data_without_quantity = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002'],
            'product_name': ['Product A', 'Product B']
        })
        
        with pytest.raises(ValueError, match="Missing required column: quantity"):
            anomaly_detector.detect_quantity_anomalies(data_without_quantity)
    
    def test_detect_discount_anomalies_basic(self, anomaly_detector, sample_data):
        """Test basic discount anomaly detection."""
        result = anomaly_detector.detect_discount_anomalies(sample_data)
        
        # Check that anomaly columns are added
        assert 'discount_anomaly_zscore' in result.columns
        assert 'discount_anomaly_iqr' in result.columns
        assert 'discount_anomaly_score' in result.columns
        assert 'discount_anomaly_reason' in result.columns
        
        # Check that some anomalies are detected (we added high discounts)
        assert result['discount_anomaly_zscore'].sum() > 0
        assert result['discount_anomaly_iqr'].sum() > 0
        
        # Check that statistics are updated
        assert anomaly_detector.detection_stats['discount_anomalies'] > 0
    
    def test_detect_discount_anomalies_missing_column(self, anomaly_detector):
        """Test discount anomaly detection with missing discount column."""
        data_without_discount = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002'],
            'product_name': ['Product A', 'Product B']
        })
        
        with pytest.raises(ValueError, match="Missing required column: discount_percent"):
            anomaly_detector.detect_discount_anomalies(data_without_discount)
    
    def test_generate_top_anomaly_records_basic(self, anomaly_detector, sample_data):
        """Test generation of top anomaly records."""
        # First run anomaly detection
        data_with_anomalies = anomaly_detector.detect_all_anomalies(sample_data)
        
        # Generate top anomaly records
        top_anomalies = anomaly_detector.generate_top_anomaly_records(data_with_anomalies, top_n=5)
        
        # Check that we get the expected number of records (or less if fewer anomalies)
        assert len(top_anomalies) <= 5
        assert len(top_anomalies) > 0  # Should have some anomalies
        
        # Check that required columns are present
        expected_columns = ['anomaly_types', 'composite_anomaly_score', 'detailed_anomaly_reason', 'anomaly_rank']
        for col in expected_columns:
            assert col in top_anomalies.columns
        
        # Check that records are sorted by anomaly score
        if len(top_anomalies) > 1:
            scores = top_anomalies['composite_anomaly_score'].values
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_generate_top_anomaly_records_no_anomalies(self, anomaly_detector):
        """Test generation of top anomaly records when no anomalies exist."""
        # Create data with no anomalies
        normal_data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002'],
            'quantity': [5, 6],
            'unit_price': [100, 110],
            'discount_percent': [0.1, 0.15],
            'revenue': [450, 535]
        })
        
        # Add anomaly columns with no anomalies
        for col in ['revenue_anomaly_zscore', 'revenue_anomaly_iqr', 
                   'quantity_anomaly_zscore', 'quantity_anomaly_iqr',
                   'discount_anomaly_zscore', 'discount_anomaly_iqr']:
            normal_data[col] = False
        
        top_anomalies = anomaly_detector.generate_top_anomaly_records(normal_data)
        
        # Should return empty DataFrame
        assert len(top_anomalies) == 0
    
    def test_detect_all_anomalies(self, anomaly_detector, sample_data):
        """Test comprehensive anomaly detection."""
        result = anomaly_detector.detect_all_anomalies(sample_data)
        
        # Check that all anomaly detection methods were run
        expected_columns = [
            'revenue_anomaly_zscore', 'revenue_anomaly_iqr',
            'quantity_anomaly_zscore', 'quantity_anomaly_iqr',
            'discount_anomaly_zscore', 'discount_anomaly_iqr'
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        # Check that statistics are updated
        assert anomaly_detector.detection_stats['total_records_analyzed'] > 0
        assert anomaly_detector.detection_stats['total_anomalies'] > 0
    
    def test_generate_anomaly_report(self, anomaly_detector, sample_data):
        """Test anomaly report generation."""
        # Run anomaly detection first
        anomaly_detector.detect_all_anomalies(sample_data)
        
        report = anomaly_detector.generate_anomaly_report()
        
        # Check report structure
        assert 'summary' in report
        assert 'detection_parameters' in report
        assert 'anomaly_breakdown' in report
        assert 'top_anomaly_records' in report
        assert 'recommendations' in report
        
        # Check summary content
        summary = report['summary']
        assert 'total_records_analyzed' in summary
        assert 'total_anomalies_detected' in summary
        assert 'anomaly_rate' in summary
        
        # Check that anomaly rate is calculated correctly
        expected_rate = summary['total_anomalies_detected'] / summary['total_records_analyzed']
        assert summary['anomaly_rate'] == expected_rate
    
    def test_get_detection_stats(self, anomaly_detector, sample_data):
        """Test detection statistics retrieval."""
        # Run some anomaly detection
        anomaly_detector.detect_revenue_anomalies(sample_data)
        
        stats = anomaly_detector.get_detection_stats()
        
        # Check that all expected keys are present
        expected_keys = [
            'total_records_analyzed', 'total_anomalies_detected',
            'revenue_anomalies', 'quantity_anomalies', 'discount_anomalies',
            'detection_errors', 'anomaly_rate'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Check that values are reasonable
        assert stats['total_records_analyzed'] > 0
        assert stats['revenue_anomalies'] > 0
        assert 0 <= stats['anomaly_rate'] <= 1
    
    def test_reset_stats(self, anomaly_detector, sample_data):
        """Test statistics reset functionality."""
        # Run some anomaly detection to populate stats
        anomaly_detector.detect_all_anomalies(sample_data)
        
        # Verify stats are populated
        assert anomaly_detector.detection_stats['total_records_analyzed'] > 0
        assert anomaly_detector.detection_stats['total_anomalies'] > 0
        
        # Reset stats
        anomaly_detector.reset_stats()
        
        # Verify stats are reset
        assert anomaly_detector.detection_stats['total_records_analyzed'] == 0
        assert anomaly_detector.detection_stats['total_anomalies'] == 0
        assert anomaly_detector.detection_stats['revenue_anomalies'] == 0
        assert anomaly_detector.detection_stats['quantity_anomalies'] == 0
        assert anomaly_detector.detection_stats['discount_anomalies'] == 0
        assert anomaly_detector.anomaly_records == []
    
    def test_zero_variance_data(self, anomaly_detector):
        """Test anomaly detection with zero variance data."""
        # Create data with constant values
        constant_data = pd.DataFrame({
            'quantity': [10] * 100,
            'unit_price': [100] * 100,
            'discount_percent': [0.1] * 100,
            'revenue': [900] * 100
        })
        
        result = anomaly_detector.detect_revenue_anomalies(constant_data)
        
        # Should handle zero variance gracefully
        assert 'revenue_anomaly_zscore' in result.columns
        assert result['revenue_anomaly_zscore'].sum() == 0  # No anomalies with zero variance
        assert all(result['revenue_anomaly_score'] == 0.0)
    
    def test_error_handling(self, anomaly_detector):
        """Test error handling in anomaly detection."""
        # Test with invalid data that should cause an error
        with patch('src.quality.anomaly_detector.logger') as mock_logger:
            # This should raise an error due to missing columns
            with pytest.raises(ValueError):
                anomaly_detector.detect_revenue_anomalies(pd.DataFrame({'invalid': [1, 2, 3]}))
            
            # Check that error was logged
            mock_logger.error.assert_called()
    
    def test_anomaly_reasons(self, anomaly_detector, sample_data):
        """Test that anomaly reasons are properly set."""
        result = anomaly_detector.detect_revenue_anomalies(sample_data)
        
        # Check that reasons are set for anomalous records
        zscore_anomalies = result[result['revenue_anomaly_zscore']]
        iqr_anomalies = result[result['revenue_anomaly_iqr']]
        
        if len(zscore_anomalies) > 0:
            assert all('Z-score' in reason for reason in zscore_anomalies['revenue_anomaly_reason'])
        
        if len(iqr_anomalies) > 0:
            assert all('IQR' in reason for reason in iqr_anomalies['revenue_anomaly_reason'])
    
    def test_custom_thresholds(self):
        """Test anomaly detector with custom thresholds."""
        detector = AnomalyDetector(z_score_threshold=2.0, iqr_multiplier=1.0)
        
        # Create data with mild outliers
        data = pd.DataFrame({
            'quantity': [1, 2, 3, 4, 5, 10],  # 10 is a mild outlier
            'unit_price': [100] * 6,
            'discount_percent': [0.1] * 6,
            'revenue': [90, 180, 270, 360, 450, 900]
        })
        
        result = detector.detect_quantity_anomalies(data)
        
        # With lower thresholds, should detect more anomalies
        assert result['quantity_anomaly_zscore'].sum() > 0 or result['quantity_anomaly_iqr'].sum() > 0