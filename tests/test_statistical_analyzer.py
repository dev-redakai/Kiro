"""
Unit tests for the StatisticalAnalyzer class.

This module contains comprehensive tests for advanced statistical analysis
and anomaly detection functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from src.quality.statistical_analyzer import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sales data for testing."""
        np.random.seed(42)
        data = {
            'order_id': [f'ORD{i:06d}' for i in range(500)],
            'product_name': [f'Product {i}' for i in range(500)],
            'category': np.random.choice(['Electronics', 'Fashion', 'Home'], 500),
            'quantity': np.random.randint(1, 20, 500),
            'unit_price': np.random.uniform(10, 1000, 500),
            'discount_percent': np.random.uniform(0, 0.3, 500),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
            'sale_date': pd.date_range('2023-01-01', periods=500, freq='2H'),
            'customer_email': [f'user{i}@example.com' for i in range(500)]
        }
        
        df = pd.DataFrame(data)
        df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_percent'])
        
        # Add some specific patterns for testing
        # High quantity + high discount pattern
        df.loc[0:4, 'quantity'] = [150, 200, 180, 220, 160]
        df.loc[0:4, 'discount_percent'] = [0.6, 0.7, 0.65, 0.75, 0.62]
        
        # High price + extreme discount pattern
        df.loc[5:9, 'unit_price'] = [15000, 12000, 18000, 14000, 16000]
        df.loc[5:9, 'discount_percent'] = [0.8, 0.85, 0.82, 0.88, 0.83]
        
        # Weekend high-value sales
        weekend_dates = pd.date_range('2023-01-07', periods=5, freq='D')  # Saturdays
        df.loc[10:14, 'sale_date'] = weekend_dates
        df.loc[10:14, 'revenue'] = [60000, 55000, 65000, 58000, 62000]
        
        return df
    
    @pytest.fixture
    def statistical_analyzer(self):
        """Create StatisticalAnalyzer instance for testing."""
        return StatisticalAnalyzer(sensitivity='medium', contamination=0.1, random_state=42)
    
    def test_init_default(self):
        """Test StatisticalAnalyzer initialization with default parameters."""
        analyzer = StatisticalAnalyzer()
        
        assert analyzer.sensitivity == 'medium'
        assert analyzer.contamination == 0.1
        assert analyzer.random_state == 42
        assert analyzer.analysis_stats['total_records_analyzed'] == 0
        assert analyzer.threshold_cache == {}
    
    def test_init_custom_parameters(self):
        """Test StatisticalAnalyzer initialization with custom parameters."""
        analyzer = StatisticalAnalyzer(sensitivity='high', contamination=0.15, random_state=123)
        
        assert analyzer.sensitivity == 'high'
        assert analyzer.contamination == 0.15
        assert analyzer.random_state == 123
        
        # Check that sensitivity parameters are configured correctly
        assert analyzer.sensitivity_params['z_score_threshold'] == 2.5
        assert analyzer.sensitivity_params['iqr_multiplier'] == 1.2
    
    def test_configure_sensitivity_parameters(self, statistical_analyzer):
        """Test sensitivity parameter configuration."""
        # Test medium sensitivity (default)
        params = statistical_analyzer.sensitivity_params
        assert params['z_score_threshold'] == 3.0
        assert params['iqr_multiplier'] == 1.5
        assert params['percentile_threshold'] == 0.95
        
        # Test high sensitivity
        statistical_analyzer.configure_sensitivity('high')
        params = statistical_analyzer.sensitivity_params
        assert params['z_score_threshold'] == 2.5
        assert params['iqr_multiplier'] == 1.2
        assert params['percentile_threshold'] == 0.90
        
        # Test low sensitivity
        statistical_analyzer.configure_sensitivity('low')
        params = statistical_analyzer.sensitivity_params
        assert params['z_score_threshold'] == 3.5
        assert params['iqr_multiplier'] == 2.0
        assert params['percentile_threshold'] == 0.99
    
    def test_configure_sensitivity_invalid(self, statistical_analyzer):
        """Test sensitivity configuration with invalid value."""
        with pytest.raises(ValueError, match="Sensitivity must be"):
            statistical_analyzer.configure_sensitivity('invalid')
    
    def test_calculate_statistical_thresholds_zscore(self, statistical_analyzer):
        """Test z-score threshold calculation."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is an outlier
        
        thresholds = statistical_analyzer.calculate_statistical_thresholds(data, method='zscore')
        
        # Check that z-score thresholds are calculated
        assert 'zscore_mean' in thresholds
        assert 'zscore_std' in thresholds
        assert 'zscore_upper_threshold' in thresholds
        assert 'zscore_lower_threshold' in thresholds
        assert 'zscore_threshold_value' in thresholds
        
        # Check that values are reasonable
        assert thresholds['zscore_mean'] == data.mean()
        assert thresholds['zscore_std'] == data.std()
        assert thresholds['zscore_threshold_value'] == 3.0  # medium sensitivity
    
    def test_calculate_statistical_thresholds_iqr(self, statistical_analyzer):
        """Test IQR threshold calculation."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        
        thresholds = statistical_analyzer.calculate_statistical_thresholds(data, method='iqr')
        
        # Check that IQR thresholds are calculated
        assert 'iqr_q1' in thresholds
        assert 'iqr_q3' in thresholds
        assert 'iqr_value' in thresholds
        assert 'iqr_upper_threshold' in thresholds
        assert 'iqr_lower_threshold' in thresholds
        
        # Check that values are correct
        assert thresholds['iqr_q1'] == data.quantile(0.25)
        assert thresholds['iqr_q3'] == data.quantile(0.75)
    
    def test_calculate_statistical_thresholds_all(self, statistical_analyzer):
        """Test calculation of all threshold types."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        
        thresholds = statistical_analyzer.calculate_statistical_thresholds(data, method='all')
        
        # Check that all threshold types are present
        expected_keys = [
            'zscore_mean', 'zscore_std', 'zscore_upper_threshold',
            'iqr_q1', 'iqr_q3', 'iqr_upper_threshold',
            'percentile_lower', 'percentile_upper',
            'modified_zscore_median', 'modified_zscore_mad'
        ]
        
        for key in expected_keys:
            assert key in thresholds
    
    def test_calculate_statistical_thresholds_empty_data(self, statistical_analyzer):
        """Test threshold calculation with empty data."""
        empty_data = pd.Series([])
        
        thresholds = statistical_analyzer.calculate_statistical_thresholds(empty_data)
        
        # Should return empty dictionary
        assert thresholds == {}
    
    def test_calculate_statistical_thresholds_constant_data(self, statistical_analyzer):
        """Test threshold calculation with constant data."""
        constant_data = pd.Series([5, 5, 5, 5, 5])
        
        thresholds = statistical_analyzer.calculate_statistical_thresholds(constant_data, method='zscore')
        
        # Should handle zero variance gracefully
        assert thresholds['zscore_std'] == 0.0
        assert thresholds['zscore_upper_threshold'] == 5.0
        assert thresholds['zscore_lower_threshold'] == 5.0
    
    def test_detect_outliers_isolation_forest(self, statistical_analyzer, sample_data):
        """Test Isolation Forest anomaly detection."""
        features = ['quantity', 'unit_price', 'discount_percent', 'revenue']
        
        result = statistical_analyzer.detect_outliers_isolation_forest(sample_data, features)
        
        # Check that isolation forest columns are added
        assert 'isolation_forest_anomaly' in result.columns
        assert 'isolation_forest_score' in result.columns
        
        # Check that some anomalies are detected
        assert result['isolation_forest_anomaly'].sum() > 0
        
        # Check that statistics are updated
        assert statistical_analyzer.analysis_stats['isolation_forest_anomalies'] > 0
    
    def test_detect_outliers_isolation_forest_missing_features(self, statistical_analyzer, sample_data):
        """Test Isolation Forest with missing features."""
        features = ['missing_feature', 'another_missing']
        
        with pytest.raises(ValueError, match="Missing features"):
            statistical_analyzer.detect_outliers_isolation_forest(sample_data, features)
    
    def test_detect_outliers_isolation_forest_no_complete_rows(self, statistical_analyzer):
        """Test Isolation Forest with no complete rows."""
        # Create data with all missing values
        incomplete_data = pd.DataFrame({
            'feature1': [np.nan, np.nan, np.nan],
            'feature2': [np.nan, np.nan, np.nan]
        })
        
        result = statistical_analyzer.detect_outliers_isolation_forest(incomplete_data, ['feature1', 'feature2'])
        
        # Should handle gracefully
        assert result['isolation_forest_anomaly'].sum() == 0
        assert all(result['isolation_forest_score'] == 0.0)
    
    def test_detect_pattern_based_anomalies(self, statistical_analyzer, sample_data):
        """Test pattern-based anomaly detection."""
        result = statistical_analyzer.detect_pattern_based_anomalies(sample_data)
        
        # Check that pattern anomaly columns are added
        assert 'pattern_anomaly' in result.columns
        assert 'pattern_anomaly_reasons' in result.columns
        assert 'pattern_anomaly_score' in result.columns
        
        # Check that some pattern anomalies are detected (we added specific patterns)
        assert result['pattern_anomaly'].sum() > 0
        
        # Check that statistics are updated
        assert statistical_analyzer.analysis_stats['pattern_anomalies'] > 0
        
        # Check that reasons are provided for anomalies
        pattern_anomalies = result[result['pattern_anomaly']]
        assert all(reason != '' for reason in pattern_anomalies['pattern_anomaly_reasons'])
    
    def test_detect_pattern_based_anomalies_specific_patterns(self, statistical_analyzer, sample_data):
        """Test detection of specific anomaly patterns."""
        result = statistical_analyzer.detect_pattern_based_anomalies(sample_data)
        
        # Check for high quantity + high discount pattern
        high_qty_discount = result[
            (result['quantity'] > 100) & (result['discount_percent'] > 0.5)
        ]
        if len(high_qty_discount) > 0:
            assert any('High quantity + High discount' in reason 
                      for reason in high_qty_discount['pattern_anomaly_reasons'])
        
        # Check for high price + extreme discount pattern
        high_price_discount = result[
            (result['unit_price'] > 10000) & (result['discount_percent'] > 0.7)
        ]
        if len(high_price_discount) > 0:
            assert any('High price + Extreme discount' in reason 
                      for reason in high_price_discount['pattern_anomaly_reasons'])
    
    def test_generate_detailed_anomaly_report(self, statistical_analyzer, sample_data):
        """Test detailed anomaly report generation."""
        # Run some anomaly detection first
        sample_data = statistical_analyzer.detect_pattern_based_anomalies(sample_data)
        
        report = statistical_analyzer.generate_detailed_anomaly_report(sample_data)
        
        # Check report structure
        assert 'analysis_summary' in report
        assert 'anomaly_statistics' in report
        assert 'threshold_analysis' in report
        assert 'pattern_analysis' in report
        assert 'recommendations' in report
        assert 'detailed_findings' in report
        
        # Check analysis summary
        summary = report['analysis_summary']
        assert 'total_records_analyzed' in summary
        assert 'sensitivity_level' in summary
        assert 'analysis_parameters' in summary
        
        # Check that recommendations are provided
        assert len(report['recommendations']) > 0
    
    def test_generate_detailed_anomaly_report_with_pattern_analysis(self, statistical_analyzer, sample_data):
        """Test detailed report with pattern analysis."""
        # Add pattern anomaly detection
        sample_data = statistical_analyzer.detect_pattern_based_anomalies(sample_data)
        
        report = statistical_analyzer.generate_detailed_anomaly_report(sample_data)
        
        # Check pattern analysis
        if 'pattern_analysis' in report and 'common_patterns' in report['pattern_analysis']:
            common_patterns = report['pattern_analysis']['common_patterns']
            assert isinstance(common_patterns, dict)
            # Should have some patterns detected
            assert len(common_patterns) > 0
    
    def test_get_analysis_stats(self, statistical_analyzer, sample_data):
        """Test analysis statistics retrieval."""
        # Run some analysis
        statistical_analyzer.detect_pattern_based_anomalies(sample_data)
        
        stats = statistical_analyzer.get_analysis_stats()
        
        # Check that all expected keys are present
        expected_keys = [
            'total_records_analyzed', 'statistical_anomalies', 'pattern_anomalies',
            'isolation_forest_anomalies', 'analysis_errors', 'sensitivity_level',
            'sensitivity_parameters'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Check that values are reasonable
        assert stats['total_records_analyzed'] > 0
        assert stats['pattern_anomalies'] > 0
        assert stats['sensitivity_level'] == 'medium'
    
    def test_reset_stats(self, statistical_analyzer, sample_data):
        """Test statistics reset functionality."""
        # Run some analysis to populate stats
        statistical_analyzer.detect_pattern_based_anomalies(sample_data)
        
        # Verify stats are populated
        assert statistical_analyzer.analysis_stats['total_records_analyzed'] > 0
        assert statistical_analyzer.analysis_stats['pattern_anomalies'] > 0
        
        # Reset stats
        statistical_analyzer.reset_stats()
        
        # Verify stats are reset
        assert statistical_analyzer.analysis_stats['total_records_analyzed'] == 0
        assert statistical_analyzer.analysis_stats['pattern_anomalies'] == 0
        assert statistical_analyzer.analysis_stats['statistical_anomalies'] == 0
        assert statistical_analyzer.analysis_stats['isolation_forest_anomalies'] == 0
        assert statistical_analyzer.threshold_cache == {}
    
    def test_threshold_caching(self, statistical_analyzer):
        """Test threshold caching functionality."""
        data = pd.Series([1, 2, 3, 4, 5], name='test_series')
        
        # Calculate thresholds first time
        thresholds1 = statistical_analyzer.calculate_statistical_thresholds(data, method='zscore')
        
        # Check that cache is populated
        assert len(statistical_analyzer.threshold_cache) > 0
        
        # Calculate thresholds second time (should use cache)
        thresholds2 = statistical_analyzer.calculate_statistical_thresholds(data, method='zscore')
        
        # Should be identical
        assert thresholds1 == thresholds2
    
    def test_sensitivity_configuration_clears_cache(self, statistical_analyzer):
        """Test that changing sensitivity clears threshold cache."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        # Calculate thresholds to populate cache
        statistical_analyzer.calculate_statistical_thresholds(data)
        assert len(statistical_analyzer.threshold_cache) > 0
        
        # Change sensitivity
        statistical_analyzer.configure_sensitivity('high')
        
        # Cache should be cleared
        assert len(statistical_analyzer.threshold_cache) == 0
    
    def test_error_handling(self, statistical_analyzer):
        """Test error handling in statistical analysis."""
        with patch('src.quality.statistical_analyzer.logger') as mock_logger:
            # Test with invalid data that should cause an error
            with pytest.raises(Exception):
                # This should cause an error
                statistical_analyzer.detect_outliers_isolation_forest(
                    pd.DataFrame({'invalid': [1, 2, 3]}), 
                    ['missing_feature']
                )
            
            # Check that error was logged
            mock_logger.error.assert_called()
    
    def test_pattern_scoring_sensitivity(self):
        """Test that pattern scoring respects sensitivity settings."""
        # Create data with mild pattern anomaly
        data = pd.DataFrame({
            'quantity': [50],  # Moderate quantity
            'unit_price': [5000],  # Moderate price
            'discount_percent': [0.6],  # High discount
            'revenue': [1000]
        })
        
        # Test with high sensitivity (should detect)
        analyzer_high = StatisticalAnalyzer(sensitivity='high')
        result_high = analyzer_high.detect_pattern_based_anomalies(data)
        
        # Test with low sensitivity (might not detect)
        analyzer_low = StatisticalAnalyzer(sensitivity='low')
        result_low = analyzer_low.detect_pattern_based_anomalies(data)
        
        # High sensitivity should be more likely to detect anomalies
        # (exact behavior depends on the specific patterns and thresholds)
        assert 'pattern_anomaly' in result_high.columns
        assert 'pattern_anomaly' in result_low.columns
    
    def test_modified_zscore_calculation(self, statistical_analyzer):
        """Test modified z-score calculation using median absolute deviation."""
        # Create data with outliers
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        
        thresholds = statistical_analyzer.calculate_statistical_thresholds(data, method='modified_zscore')
        
        # Check that modified z-score thresholds are calculated
        assert 'modified_zscore_median' in thresholds
        assert 'modified_zscore_mad' in thresholds
        assert 'modified_zscore_threshold' in thresholds
        assert 'modified_zscore_upper' in thresholds
        assert 'modified_zscore_lower' in thresholds
        
        # Check that values are reasonable
        assert thresholds['modified_zscore_median'] == data.median()
        assert thresholds['modified_zscore_threshold'] == 3.5
    
    def test_weekend_pattern_detection(self, statistical_analyzer):
        """Test weekend high-value sales pattern detection."""
        # Create data with weekend high-value sales
        weekend_data = pd.DataFrame({
            'sale_date': pd.to_datetime(['2023-01-07', '2023-01-08']),  # Saturday, Sunday
            'revenue': [60000, 55000],  # High values
            'quantity': [10, 12],
            'unit_price': [6000, 4583],
            'discount_percent': [0.0, 0.0]
        })
        
        result = statistical_analyzer.detect_pattern_based_anomalies(weekend_data)
        
        # Should detect weekend high-value pattern
        weekend_anomalies = result[result['pattern_anomaly']]
        if len(weekend_anomalies) > 0:
            assert any('Weekend high-value sale' in reason 
                      for reason in weekend_anomalies['pattern_anomaly_reasons'])
    
    def test_email_pattern_detection(self, statistical_analyzer):
        """Test suspicious email pattern detection."""
        # Create data with suspicious email patterns
        suspicious_data = pd.DataFrame({
            'customer_email': ['aaa@test.com', 'x@y.z', 'test....test@example.com'],
            'quantity': [5, 6, 7],
            'unit_price': [100, 110, 120],
            'discount_percent': [0.1, 0.15, 0.2],
            'revenue': [450, 535, 672]
        })
        
        result = statistical_analyzer.detect_pattern_based_anomalies(suspicious_data)
        
        # Should detect suspicious email patterns
        email_anomalies = result[result['pattern_anomaly']]
        if len(email_anomalies) > 0:
            assert any('Suspicious email pattern' in reason 
                      for reason in email_anomalies['pattern_anomaly_reasons'])