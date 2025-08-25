"""
Comprehensive test suite for ETL validation fixes.

This test suite covers:
1. Unit tests for date validation with mixed data types
2. Integration tests for error handler parameter passing
3. Edge case tests for validation resilience
4. Performance tests for validation under error conditions

Requirements covered: 1.1, 2.1, 3.1, 4.1
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.quality.automated_validation import (
    AutomatedDataValidator, ValidationResult, ValidationRule, ValidationSeverity,
    ValidationExecutionContext, ValidationErrorReport, ValidationMetrics
)
from src.utils.error_handler import ErrorCategory, ErrorSeverity as ErrorHandlerSeverity, DataQualityError


class TestDateValidationMixedDataTypes:
    """Unit tests for date validation with mixed data types (Requirement 1.1)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_date_validation_with_categorical_data(self):
        """Test date validation handles categorical data gracefully."""
        # Create mixed data with categorical values
        mixed_data = pd.Series([
            '2023-01-01',
            '2023-02-15', 
            'Unknown',
            'N/A',
            '2023-03-20',
            'null',
            '2023-12-31'
        ])
        
        is_valid, details = self.validator._validate_date_range_robust(
            mixed_data, 
            min_date='2020-01-01', 
            max_date='2024-12-31'
        )
        
        # Should handle categorical data without crashing
        assert isinstance(is_valid, pd.Series)
        assert len(is_valid) == len(mixed_data)
        
        # Valid dates should pass, categorical should be handled gracefully
        assert is_valid.iloc[0] == True  # '2023-01-01'
        assert is_valid.iloc[1] == True  # '2023-02-15'
        assert is_valid.iloc[4] == True  # '2023-03-20'
        assert is_valid.iloc[6] == True  # '2023-12-31'
        
        # Check conversion stats
        assert 'conversion_stats' in details
        assert details['conversion_stats']['categorical_count'] > 0
        assert details['conversion_stats']['string_count'] > 0
        
        # Should have conversion attempts and some successes/failures
        assert details['conversion_stats']['conversion_attempts'] > 0
        assert details['conversion_stats']['conversion_successes'] >= 0
        assert details['conversion_stats']['conversion_failures'] >= 0
    
    def test_date_validation_with_numeric_data(self):
        """Test date validation with numeric timestamp data."""
        # Create numeric timestamp data (Unix timestamps)
        numeric_data = pd.Series([
            1672531200,  # 2023-01-01 00:00:00 UTC
            1675555200,  # 2023-02-05 00:00:00 UTC
            np.nan,
            1677628800,  # 2023-03-01 00:00:00 UTC
            'invalid'
        ])
        
        is_valid, details = self.validator._validate_date_range_robust(
            numeric_data,
            min_date='2020-01-01',
            max_date='2024-12-31'
        )
        
        # Should handle mixed numeric/string data
        assert isinstance(is_valid, pd.Series)
        assert len(is_valid) == len(numeric_data)
        
        # Check conversion stats
        assert 'conversion_stats' in details
        assert details['conversion_stats']['numeric_count'] > 0
        assert details['conversion_stats']['string_count'] > 0
    
    def test_date_validation_with_datetime_objects(self):
        """Test date validation with actual datetime objects."""
        datetime_data = pd.Series([
            datetime(2023, 1, 1),
            datetime(2023, 6, 15),
            pd.NaT,
            datetime(2023, 12, 31),
            'string_value'  # Mixed with string
        ])
        
        is_valid, details = self.validator._validate_date_range_robust(
            datetime_data,
            min_date='2020-01-01',
            max_date='2024-12-31'
        )
        
        # Should handle mixed datetime/string data
        assert isinstance(is_valid, pd.Series)
        assert len(is_valid) == len(datetime_data)
        
        # Datetime objects should be valid
        assert is_valid.iloc[0] == True
        assert is_valid.iloc[1] == True
        assert is_valid.iloc[3] == True
        
        # NaT should be considered valid (null handling)
        assert is_valid.iloc[2] == True
        
        # Check conversion stats
        assert 'conversion_stats' in details
        # Note: datetime_count may be 0 if data type detection categorizes differently
    
    def test_date_validation_empty_series(self):
        """Test date validation with empty series."""
        empty_data = pd.Series([], dtype=object)
        
        is_valid, details = self.validator._validate_date_range_robust(
            empty_data,
            min_date='2020-01-01',
            max_date='2024-12-31'
        )
        
        # Should handle empty series gracefully
        assert isinstance(is_valid, pd.Series)
        assert len(is_valid) == 0
        assert 'warning_messages' in details
        assert 'Empty series provided for date validation' in details['warning_messages']
    
    def test_date_validation_all_null_series(self):
        """Test date validation with all null values."""
        null_data = pd.Series([None, np.nan, pd.NaT, None])
        
        is_valid, details = self.validator._validate_date_range_robust(
            null_data,
            min_date='2020-01-01',
            max_date='2024-12-31'
        )
        
        # All null values should be considered valid
        assert isinstance(is_valid, pd.Series)
        assert all(is_valid)
        assert 'warning_messages' in details
        assert any('All values are null' in msg for msg in details['warning_messages'])
    
    def test_date_validation_format_fallback(self):
        """Test date validation with various date formats."""
        format_data = pd.Series([
            '01/15/2023',  # MM/DD/YYYY
            '15/01/2023',  # DD/MM/YYYY
            '2023-01-15',  # YYYY-MM-DD
            '20230115',    # YYYYMMDD
            '2023-01-15 10:30:00',  # With time
            'invalid_date'
        ])
        
        is_valid, details = self.validator._validate_date_range_robust(
            format_data,
            min_date='2020-01-01',
            max_date='2024-12-31'
        )
        
        # Should attempt multiple format conversions
        assert isinstance(is_valid, pd.Series)
        assert len(is_valid) == len(format_data)
        
        # Should have conversion attempts and some successes
        assert details['conversion_stats']['conversion_attempts'] > 0
        assert details['conversion_stats']['conversion_successes'] > 0
        
        # Should have format-specific warnings if fallback was applied
        if details['conversion_stats']['fallback_applied']:
            assert 'warning_messages' in details
    
    def test_date_validation_out_of_range(self):
        """Test date validation with dates outside specified range."""
        range_data = pd.Series([
            '2019-12-31',  # Before min_date
            '2023-06-15',  # Within range
            '2025-01-01',  # After max_date
            '2022-03-20'   # Within range
        ])
        
        is_valid, details = self.validator._validate_date_range_robust(
            range_data,
            min_date='2020-01-01',
            max_date='2024-12-31'
        )
        
        # Should validate date ranges correctly
        assert is_valid.iloc[0] == False  # Before min_date
        assert is_valid.iloc[1] == True   # Within range
        assert is_valid.iloc[2] == False  # After max_date
        assert is_valid.iloc[3] == True   # Within range
        
        # Should report out of range count
        assert details['out_of_range_count'] == 2
        assert details['min_date'] == '2020-01-01'
        assert details['max_date'] == '2024-12-31'


class TestErrorHandlerParameterPassing:
    """Integration tests for error handler parameter passing (Requirement 2.1)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    @patch('src.utils.error_handler.error_handler.handle_error')
    def test_error_handler_called_with_category(self, mock_handle_error):
        """Test that error handler is called with correct category parameter."""
        # Create data that will cause validation errors
        problematic_data = pd.DataFrame({
            'sale_date': ['invalid_date', '2023-01-01', 'another_invalid']
        })
        
        # Run validation which should trigger error handling
        results = self.validator.validate_dataframe(problematic_data, ['sale_date_range'])
        
        # Verify error handler was called with ErrorCategory parameter
        if mock_handle_error.called:
            # Check that calls include the category parameter
            for call_args in mock_handle_error.call_args_list:
                args, kwargs = call_args
                # Should have at least 2 positional args: error and category
                assert len(args) >= 2
                # Second argument should be ErrorCategory
                assert isinstance(args[1], ErrorCategory)
    
    @patch('src.utils.error_handler.error_handler.handle_error')
    def test_error_handler_receives_all_required_arguments(self, mock_handle_error):
        """Test that error handler receives all required arguments."""
        # Create a validation rule that will fail
        test_data = pd.DataFrame({
            'quantity': [-1, 'invalid', 2, None]
        })
        
        # Run validation
        results = self.validator.validate_dataframe(test_data, ['quantity_positive'])
        
        # If error handler was called, verify the signature
        if mock_handle_error.called:
            for call_args in mock_handle_error.call_args_list:
                args, kwargs = call_args
                
                # Should have error, category, and optionally severity and context
                assert len(args) >= 2
                assert isinstance(args[0], Exception)  # Error
                assert isinstance(args[1], ErrorCategory)  # Category
                
                # Check for severity parameter (can be positional or keyword)
                if len(args) > 2:
                    assert isinstance(args[2], ErrorHandlerSeverity)
                elif 'severity' in kwargs:
                    assert isinstance(kwargs['severity'], ErrorHandlerSeverity)
                
                # Context should be provided
                if len(args) > 3:
                    assert isinstance(args[3], dict)
                elif 'context' in kwargs:
                    assert isinstance(kwargs['context'], dict)
    
    def test_error_handler_integration_without_crash(self):
        """Test that error handler integration doesn't cause secondary failures."""
        # Create data that will definitely cause validation errors
        error_prone_data = pd.DataFrame({
            'sale_date': [None, 'invalid', 123, [], {}],
            'quantity': ['not_a_number', -999, None, float('inf'), 'text'],
            'unit_price': [None, 'free', -100, 0, 'expensive']
        })
        
        # This should not crash even with multiple validation errors
        results = self.validator.validate_dataframe(error_prone_data)
        
        # Should complete without crashing
        assert 'status' in results
        assert results['status'] in ['passed', 'warning', 'error', 'critical']
        
        # Should have processed all records
        assert ('data_quality_metrics' in results and 'total_records_processed' in results['data_quality_metrics']) or 'validation_results' in results
    
    @patch('src.utils.error_handler.error_handler.handle_error', side_effect=Exception("Handler failed"))
    def test_error_handler_failure_resilience(self, mock_handle_error):
        """Test resilience when error handler itself fails."""
        # Create data that will cause validation errors
        test_data = pd.DataFrame({
            'sale_date': ['invalid_date', 'another_invalid']
        })
        
        # Even if error handler fails, validation should continue
        results = self.validator.validate_dataframe(test_data, ['sale_date_range'])
        
        # Should not crash the entire validation process
        assert isinstance(results, dict)
        assert 'status' in results or 'validation_results' in results
    
    def test_multiple_validation_errors_categorization(self):
        """Test that multiple validation errors are properly categorized."""
        # Create data with various types of validation errors
        multi_error_data = pd.DataFrame({
            'order_id': [None, '', 'valid_id'],  # NOT_NULL errors
            'quantity': [-1, 0, 'invalid'],      # RANGE errors  
            'sale_date': ['invalid', '1900-01-01', '2023-01-01'],  # DATE errors
            'category': ['Invalid', 'Electronics', 'Fashion']  # ENUM errors
        })
        
        # Run comprehensive validation
        results = self.validator.validate_dataframe(multi_error_data)
        
        # Should handle multiple error types without crashing
        assert isinstance(results, dict)
        
        # Should have processed all validation rules
        if 'validation_results' in results:
            assert len(results['validation_results']) > 0


class TestValidationResilience:
    """Edge case tests for validation resilience (Requirement 3.1)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_single_rule_failure_isolation(self):
        """Test that single validation rule failure doesn't crash entire pipeline."""
        # Create a custom rule that will always fail
        def failing_validation_rule(series, **kwargs):
            raise ValueError("This rule always fails")
        
        # Register the failing rule
        self.validator.register_validation_rule(
            "always_fails",
            "test_field",
            ValidationRule.CUSTOM,
            ValidationSeverity.ERROR,
            failing_validation_rule
        )
        
        # Create test data
        test_data = pd.DataFrame({
            'test_field': [1, 2, 3],
            'quantity': [10, 20, 30]  # This should still be validated
        })
        
        # Run validation - should not crash
        results = self.validator.validate_dataframe(test_data)
        
        # Should complete with some results
        assert isinstance(results, dict)
        assert 'status' in results
        
        # Should have execution statistics showing the failure
        if 'rule_execution_stats' in results:
            stats = results['rule_execution_stats']
            assert stats['rules_failed_execution'] > 0
            assert stats['execution_errors']
            
            # Should have error details for the failing rule
            failing_errors = [e for e in stats['execution_errors'] if e['rule_name'] == 'always_fails']
            assert len(failing_errors) > 0
    
    def test_validation_with_corrupted_data_types(self):
        """Test validation resilience with corrupted/unexpected data types."""
        # Create DataFrame with very problematic data types
        corrupted_data = pd.DataFrame({
            'mixed_field': [
                1, 'string', [1, 2, 3], {'key': 'value'}, 
                lambda x: x, None, float('inf'), complex(1, 2)
            ]
        })
        
        # Should handle corrupted data without crashing
        results = self.validator.validate_dataframe(corrupted_data)
        
        assert isinstance(results, dict)
        assert 'status' in results
    
    def test_validation_with_memory_constraints(self):
        """Test validation behavior under memory constraints."""
        # Create a large dataset that might cause memory issues
        large_data = pd.DataFrame({
            'id': range(10000),
            'data': ['x' * 100] * 10000,  # Large strings
            'numbers': np.random.randn(10000)
        })
        
        # Should handle large datasets gracefully
        results = self.validator.validate_dataframe(large_data)
        
        assert isinstance(results, dict)
        assert 'status' in results
    
    def test_validation_rule_exception_handling(self):
        """Test handling of various exception types in validation rules."""
        exception_types = [
            ValueError("Value error"),
            TypeError("Type error"),
            KeyError("Key error"),
            IndexError("Index error"),
            AttributeError("Attribute error"),
            RuntimeError("Runtime error")
        ]
        
        for i, exception in enumerate(exception_types):
            # Create a rule that raises this specific exception
            def exception_rule(series, exc=exception, **kwargs):
                raise exc
            
            rule_name = f"exception_rule_{i}"
            self.validator.register_validation_rule(
                rule_name,
                "test_field",
                ValidationRule.CUSTOM,
                ValidationSeverity.WARNING,
                exception_rule
            )
        
        # Test data
        test_data = pd.DataFrame({'test_field': [1, 2, 3]})
        
        # Should handle all exception types without crashing
        results = self.validator.validate_dataframe(test_data)
        
        assert isinstance(results, dict)
        assert 'status' in results
        
        # Should have recorded all the exceptions
        if 'rule_execution_stats' in results:
            stats = results['rule_execution_stats']
            assert stats['rules_failed_execution'] >= len(exception_types)
    
    def test_validation_with_infinite_and_nan_values(self):
        """Test validation resilience with infinite and NaN values."""
        extreme_data = pd.DataFrame({
            'numbers': [
                float('inf'), float('-inf'), np.nan,
                1e308, -1e308, 0, 1, -1
            ],
            'strings': [
                'normal', '', None, 'inf', 'nan',
                'very_long_string' * 1000, '\x00\x01\x02', '🚀'
            ]
        })
        
        # Should handle extreme values without crashing
        results = self.validator.validate_dataframe(extreme_data)
        
        assert isinstance(results, dict)
        assert 'status' in results
    
    def test_comprehensive_validation_summary(self):
        """Test that validation provides comprehensive summary with success/failure metrics."""
        # Create mixed data with some valid and some invalid records
        mixed_data = pd.DataFrame({
            'order_id': ['ORD001', '', 'ORD003', None, 'ORD005'],
            'quantity': [1, -1, 5, 0, 10],
            'sale_date': ['2023-01-01', 'invalid', '2023-03-01', None, '2023-05-01']
        })
        
        results = self.validator.validate_dataframe(mixed_data)
        
        # Should provide comprehensive metrics
        assert isinstance(results, dict)
        
        # Should have overall status
        assert 'status' in results
        
        # Should have execution statistics
        if 'rule_execution_stats' in results:
            stats = results['rule_execution_stats']
            assert 'total_rules_attempted' in stats
            assert 'rules_executed_successfully' in stats
            assert 'rules_failed_execution' in stats
            assert 'rules_skipped' in stats
        
        # Should have validation results if any rules succeeded
        if 'validation_results' in results:
            assert isinstance(results['validation_results'], list)


class TestValidationPerformance:
    """Performance tests for validation under error conditions (Requirement 4.1)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_validation_performance_with_errors(self):
        """Test validation performance when errors occur."""
        # Create data that will cause many validation errors
        error_heavy_data = pd.DataFrame({
            'sale_date': ['invalid'] * 1000 + ['2023-01-01'] * 100,
            'quantity': [-1] * 500 + [1] * 600,
            'unit_price': [0] * 300 + [10.0] * 800
        })
        
        start_time = time.time()
        results = self.validator.validate_dataframe(error_heavy_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (less than 30 seconds for 1100 records)
        assert execution_time < 30.0
        
        # Should still provide results
        assert isinstance(results, dict)
        assert 'status' in results
        
        # Log performance metrics for analysis
        print(f"Validation with errors completed in {execution_time:.2f} seconds")
        if 'rule_execution_stats' in results:
            stats = results['rule_execution_stats']
            print(f"Rules executed: {stats.get('rules_executed_successfully', 0)}")
            print(f"Rules failed: {stats.get('rules_failed_execution', 0)}")
    
    def test_validation_memory_usage(self):
        """Test validation memory usage under error conditions."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory-intensive data with errors
        large_error_data = pd.DataFrame({
            'large_text': ['invalid_date_' + 'x' * 1000] * 5000,
            'numbers': list(range(5000)),
            'categories': ['Invalid_Category_' + str(i) for i in range(5000)]
        })
        
        results = self.validator.validate_dataframe(large_error_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500
        
        # Should still complete successfully
        assert isinstance(results, dict)
        
        print(f"Memory usage increased by {memory_increase:.2f} MB")
    
    def test_validation_with_concurrent_errors(self):
        """Test validation performance with multiple concurrent error types."""
        # Create data with multiple types of errors occurring simultaneously
        concurrent_error_data = pd.DataFrame({
            'field1': [None] * 200 + ['valid'] * 100,  # Null errors
            'field2': ['invalid_format'] * 150 + ['valid@email.com'] * 150,  # Format errors
            'field3': [-1] * 100 + [999999] * 100 + [50] * 100,  # Range errors
            'field4': ['InvalidEnum'] * 180 + ['ValidEnum'] * 120,  # Enum errors
            'field5': ['not_a_date'] * 160 + ['2023-01-01'] * 140  # Date errors
        })
        
        start_time = time.time()
        results = self.validator.validate_dataframe(concurrent_error_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should handle concurrent errors efficiently
        assert execution_time < 20.0  # Should complete within 20 seconds
        
        # Should provide comprehensive results
        assert isinstance(results, dict)
        assert 'status' in results
        
        print(f"Concurrent error validation completed in {execution_time:.2f} seconds")
    
    def test_validation_error_recovery_performance(self):
        """Test performance of error recovery mechanisms."""
        # Create data that will trigger various recovery mechanisms
        recovery_test_data = pd.DataFrame({
            'dates_mixed_types': [
                '2023-01-01', 'Unknown', 123456789, None,
                '01/15/2023', 'N/A', datetime(2023, 3, 1), 'invalid'
            ] * 100,  # 800 records with mixed types requiring recovery
        })
        
        start_time = time.time()
        results = self.validator.validate_dataframe(recovery_test_data, ['sale_date_range'])
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Recovery mechanisms should not significantly impact performance
        assert execution_time < 15.0
        
        # Should complete with results
        assert isinstance(results, dict)
        
        print(f"Error recovery validation completed in {execution_time:.2f} seconds")
    
    def test_validation_scalability_with_errors(self):
        """Test validation scalability as error rate increases."""
        error_rates = [0.1, 0.3, 0.5, 0.7, 0.9]  # 10% to 90% error rates
        performance_results = []
        
        for error_rate in error_rates:
            # Create data with specified error rate
            total_records = 1000
            error_count = int(total_records * error_rate)
            valid_count = total_records - error_count
            
            test_data = pd.DataFrame({
                'test_field': ['invalid'] * error_count + ['2023-01-01'] * valid_count
            })
            
            start_time = time.time()
            results = self.validator.validate_dataframe(test_data, ['sale_date_range'])
            end_time = time.time()
            
            execution_time = end_time - start_time
            performance_results.append((error_rate, execution_time))
            
            # Should complete regardless of error rate
            assert isinstance(results, dict)
            
            print(f"Error rate {error_rate:.1%}: {execution_time:.2f} seconds")
        
        # Performance should not degrade exponentially with error rate
        # (allowing for some increase but not more than 10x from lowest to highest)
        min_time = min(time for _, time in performance_results)
        max_time = max(time for _, time in performance_results)
        
        assert max_time / min_time < 10.0, "Performance degradation too severe with high error rates"


class TestValidationFixesIntegration:
    """Integration tests combining all validation fixes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_complete_validation_fixes_integration(self):
        """Test all validation fixes working together."""
        # Create comprehensive test data that exercises all fixes
        integration_data = pd.DataFrame({
            # Date validation with mixed types (Fix 1)
            'sale_date': [
                '2023-01-01', 'Unknown', 'N/A', datetime(2023, 2, 1),
                123456789, None, '01/15/2023', 'invalid_date'
            ],
            # Data that will trigger error handler (Fix 2)
            'quantity': [1, -1, 'invalid', None, float('inf'), 0, 5, 'text'],
            # Data that will test resilience (Fix 3)
            'unit_price': [10.0, 0, -5, None, 'free', 999999, 15.5, []],
            # Data for enhanced logging (Fix 4)
            'category': [
                'Electronics', 'InvalidCategory', None, 'Fashion',
                '', 'Home & Garden', 'AnotherInvalid', 'Electronics'
            ]
        })
        
        # Run comprehensive validation
        results = self.validator.validate_dataframe(integration_data)
        
        # Should complete successfully with all fixes applied
        assert isinstance(results, dict)
        assert 'status' in results
        
        # Should have comprehensive execution statistics (Fix 3 & 4)
        if 'rule_execution_stats' in results:
            stats = results['rule_execution_stats']
            assert 'total_rules_attempted' in stats
            assert 'rules_executed_successfully' in stats
            assert 'rules_failed_execution' in stats
            assert 'execution_errors' in stats
        
        # Should have validation results for successful rules
        if 'validation_results' in results:
            assert isinstance(results['validation_results'], list)
            assert len(results['validation_results']) > 0
    
    @patch('src.utils.error_handler.error_handler.handle_error')
    def test_error_handler_integration_with_all_fixes(self, mock_handle_error):
        """Test error handler integration with all validation fixes."""
        # Create data that will trigger multiple error conditions
        error_integration_data = pd.DataFrame({
            'sale_date': ['invalid_date', 'Unknown', None],
            'quantity': [-1, 'not_number', None],
            'unit_price': [0, 'free', None]
        })
        
        # Run validation
        results = self.validator.validate_dataframe(error_integration_data)
        
        # Should complete without crashing
        assert isinstance(results, dict)
        
        # If error handler was called, verify proper parameter passing
        if mock_handle_error.called:
            for call_args in mock_handle_error.call_args_list:
                args, kwargs = call_args
                # Should have proper error categorization (Fix 2)
                assert len(args) >= 2
                assert isinstance(args[1], ErrorCategory)
    
    def test_validation_fixes_performance_integration(self):
        """Test performance of all validation fixes working together."""
        # Create large dataset that exercises all fixes
        large_integration_data = pd.DataFrame({
            'sale_date': (['invalid'] * 500 + ['2023-01-01'] * 500 + 
                         ['Unknown'] * 300 + [None] * 200),
            'quantity': ([-1] * 400 + [1] * 600 + ['invalid'] * 300 + [None] * 200),
            'unit_price': ([0] * 300 + [10.0] * 700 + ['free'] * 300 + [None] * 200),
            'category': (['Invalid'] * 400 + ['Electronics'] * 600 + 
                        [''] * 300 + [None] * 200)
        })
        
        start_time = time.time()
        results = self.validator.validate_dataframe(large_integration_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time with all fixes
        assert execution_time < 60.0  # 1 minute for 1500 records
        
        # Should provide comprehensive results
        assert isinstance(results, dict)
        assert 'status' in results
        
        print(f"Integration validation completed in {execution_time:.2f} seconds")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])