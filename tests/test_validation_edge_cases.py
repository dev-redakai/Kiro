"""
Edge case tests for validation resilience.

This module contains specialized edge case tests that push the validation system
to its limits to ensure robustness under extreme conditions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import gc
import threading
import time
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.quality.automated_validation import (
    AutomatedDataValidator, ValidationRule, ValidationSeverity
)


class TestExtremeDataTypes:
    """Test validation with extreme and unusual data types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_validation_with_complex_objects(self):
        """Test validation with complex Python objects."""
        complex_data = pd.DataFrame({
            'complex_field': [
                {'nested': {'dict': 'value'}},
                [1, 2, [3, 4, {'deep': 'nested'}]],
                lambda x: x + 1,
                type('CustomClass', (), {'attr': 'value'})(),
                set([1, 2, 3]),
                frozenset([4, 5, 6]),
                complex(1, 2),
                bytes([1, 2, 3, 4]),
                bytearray([5, 6, 7, 8])
            ]
        })
        
        # Should handle complex objects without crashing
        results = self.validator.validate_dataframe(complex_data)
        assert isinstance(results, dict)
        assert 'status' in results
    
    def test_validation_with_circular_references(self):
        """Test validation with circular reference objects."""
        # Create objects with circular references
        obj1 = {'name': 'obj1'}
        obj2 = {'name': 'obj2'}
        obj1['ref'] = obj2
        obj2['ref'] = obj1
        
        circular_data = pd.DataFrame({
            'circular_field': [obj1, obj2, 'normal_string', None, obj1]
        })
        
        # Should handle circular references gracefully
        results = self.validator.validate_dataframe(circular_data)
        assert isinstance(results, dict)
        assert 'status' in results
    
    def test_validation_with_unicode_and_encoding_issues(self):
        """Test validation with various Unicode and encoding challenges."""
        unicode_data = pd.DataFrame({
            'unicode_field': [
                '🚀🌟💫',  # Emojis
                'café naïve résumé',  # Accented characters
                '中文测试',  # Chinese characters
                'العربية',  # Arabic
                'русский',  # Cyrillic
                '\x00\x01\x02\x03',  # Control characters
                '\u200b\u200c\u200d',  # Zero-width characters
                'a' * 10000,  # Very long string
                '',  # Empty string
                None
            ]
        })
        
        # Should handle Unicode gracefully
        results = self.validator.validate_dataframe(unicode_data)
        assert isinstance(results, dict)
        assert 'status' in results
    
    def test_validation_with_extreme_numeric_values(self):
        """Test validation with extreme numeric values."""
        extreme_numeric_data = pd.DataFrame({
            'extreme_numbers': [
                float('inf'),
                float('-inf'),
                np.nan,
                1e308,  # Near float64 max
                -1e308,  # Near float64 min
                1e-308,  # Near float64 min positive
                sys.maxsize,  # Max int
                -sys.maxsize - 1,  # Min int
                0,
                1,
                -1,
                complex(float('inf'), float('inf')),
                complex(np.nan, np.nan)
            ]
        })
        
        # Should handle extreme numeric values
        results = self.validator.validate_dataframe(extreme_numeric_data)
        assert isinstance(results, dict)
        assert 'status' in results


class TestConcurrencyAndThreading:
    """Test validation under concurrent access and threading conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
        self.results = []
        self.errors = []
    
    def validation_worker(self, worker_id, data):
        """Worker function for concurrent validation testing."""
        try:
            result = self.validator.validate_dataframe(data)
            self.results.append((worker_id, result))
        except Exception as e:
            self.errors.append((worker_id, e))
    
    def test_concurrent_validation_access(self):
        """Test validation with concurrent access from multiple threads."""
        # Create test data
        test_data = pd.DataFrame({
            'field1': ['value1', 'value2', 'value3'] * 100,
            'field2': [1, 2, 3] * 100,
            'field3': ['2023-01-01', '2023-01-02', '2023-01-03'] * 100
        })
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=self.validation_worker,
                args=(i, test_data.copy())
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check results
        assert len(self.errors) == 0, f"Concurrent validation errors: {self.errors}"
        assert len(self.results) == 5, f"Expected 5 results, got {len(self.results)}"
        
        # All results should be valid
        for worker_id, result in self.results:
            assert isinstance(result, dict)
            assert 'status' in result
    
    def test_validation_with_shared_state_modification(self):
        """Test validation behavior when shared state is modified concurrently."""
        def modify_validator_state():
            """Function to modify validator state during validation."""
            time.sleep(0.1)  # Let validation start
            # Try to modify validator state
            try:
                self.validator.register_validation_rule(
                    "concurrent_rule",
                    "test_field",
                    ValidationRule.CUSTOM,
                    ValidationSeverity.WARNING,
                    lambda x, **kwargs: (pd.Series(True, index=x.index), {})
                )
            except:
                pass  # Expected to potentially fail
        
        # Start state modification in background
        modifier_thread = threading.Thread(target=modify_validator_state)
        modifier_thread.start()
        
        # Run validation
        test_data = pd.DataFrame({'test_field': [1, 2, 3] * 100})
        result = self.validator.validate_dataframe(test_data)
        
        # Wait for modifier thread
        modifier_thread.join()
        
        # Should complete without crashing
        assert isinstance(result, dict)
        assert 'status' in result


class TestMemoryAndResourceLimits:
    """Test validation under memory and resource constraints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_validation_with_memory_pressure(self):
        """Test validation behavior under memory pressure."""
        # Force garbage collection before test
        gc.collect()
        
        # Create memory-intensive data
        memory_intensive_data = pd.DataFrame({
            'large_strings': ['x' * 10000] * 1000,
            'large_lists': [[i] * 1000 for i in range(1000)],
            'repeated_data': ['memory_test'] * 1000
        })
        
        # Run validation under memory pressure
        result = self.validator.validate_dataframe(memory_intensive_data)
        
        # Should complete despite memory pressure
        assert isinstance(result, dict)
        assert 'status' in result
        
        # Clean up
        del memory_intensive_data
        gc.collect()
    
    def test_validation_with_very_large_datasets(self):
        """Test validation with datasets approaching memory limits."""
        # Create large dataset (but not too large for CI)
        large_data = pd.DataFrame({
            'id': range(50000),
            'data': [f'data_{i}' for i in range(50000)],
            'numbers': np.random.randn(50000)
        })
        
        # Should handle large datasets
        result = self.validator.validate_dataframe(large_data)
        
        assert isinstance(result, dict)
        assert 'status' in result
        
        # Clean up
        del large_data
        gc.collect()
    
    @patch('psutil.virtual_memory')
    def test_validation_with_low_memory_simulation(self, mock_memory):
        """Test validation behavior when system memory is low."""
        # Mock low memory condition
        mock_memory.return_value.available = 100 * 1024 * 1024  # 100MB available
        
        test_data = pd.DataFrame({
            'field1': ['test'] * 1000,
            'field2': list(range(1000))
        })
        
        # Should adapt to low memory conditions
        result = self.validator.validate_dataframe(test_data)
        
        assert isinstance(result, dict)
        assert 'status' in result


class TestValidationRuleEdgeCases:
    """Test edge cases in validation rule execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_validation_rule_with_infinite_loop(self):
        """Test validation rule that could cause infinite loop."""
        def potentially_infinite_rule(series, **kwargs):
            # Simulate a rule that might loop indefinitely
            count = 0
            while count < 1000000:  # Large but finite loop
                count += 1
                if count > 100:  # Break early for test
                    break
            return pd.Series(True, index=series.index), {'loop_count': count}
        
        self.validator.register_validation_rule(
            "infinite_loop_test",
            "test_field",
            ValidationRule.CUSTOM,
            ValidationSeverity.WARNING,
            potentially_infinite_rule
        )
        
        test_data = pd.DataFrame({'test_field': [1, 2, 3]})
        
        # Should complete within reasonable time
        start_time = time.time()
        result = self.validator.validate_dataframe(test_data, ['infinite_loop_test'])
        end_time = time.time()
        
        # Should not take too long
        assert (end_time - start_time) < 10.0  # Less than 10 seconds
        assert isinstance(result, dict)
    
    def test_validation_rule_with_recursive_calls(self):
        """Test validation rule that makes recursive calls."""
        def recursive_rule(series, depth=0, **kwargs):
            if depth > 10:  # Limit recursion depth
                return pd.Series(True, index=series.index), {'max_depth_reached': True}
            
            # Simulate recursive processing
            if len(series) > 1:
                # Process in smaller chunks recursively
                mid = len(series) // 2
                left_result, left_details = recursive_rule(series[:mid], depth + 1)
                right_result, right_details = recursive_rule(series[mid:], depth + 1)
                
                # Combine results
                combined_result = pd.concat([left_result, right_result])
                combined_details = {'depth': depth, 'left': left_details, 'right': right_details}
                return combined_result, combined_details
            else:
                return pd.Series(True, index=series.index), {'depth': depth}
        
        self.validator.register_validation_rule(
            "recursive_test",
            "test_field",
            ValidationRule.CUSTOM,
            ValidationSeverity.INFO,
            recursive_rule
        )
        
        test_data = pd.DataFrame({'test_field': list(range(100))})
        
        # Should handle recursion without stack overflow
        result = self.validator.validate_dataframe(test_data, ['recursive_test'])
        
        assert isinstance(result, dict)
        assert 'status' in result
    
    def test_validation_rule_with_external_dependencies(self):
        """Test validation rule that depends on external resources."""
        def external_dependency_rule(series, **kwargs):
            # Simulate rule that might depend on external resource
            try:
                # This might fail if external resource is unavailable
                import random
                time.sleep(0.001)  # Simulate network delay
                
                # Simulate external validation
                external_result = random.choice([True, False])
                return pd.Series(external_result, index=series.index), {'external_check': True}
            except Exception as e:
                # Fallback when external dependency fails
                return pd.Series(False, index=series.index), {'external_error': str(e)}
        
        self.validator.register_validation_rule(
            "external_dependency_test",
            "test_field",
            ValidationRule.CUSTOM,
            ValidationSeverity.WARNING,
            external_dependency_rule
        )
        
        test_data = pd.DataFrame({'test_field': [1, 2, 3, 4, 5]})
        
        # Should handle external dependency issues gracefully
        result = self.validator.validate_dataframe(test_data, ['external_dependency_test'])
        
        assert isinstance(result, dict)
        assert 'status' in result


class TestDataCorruptionScenarios:
    """Test validation with various data corruption scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_validation_with_corrupted_dataframe_structure(self):
        """Test validation with corrupted DataFrame structure."""
        # Create DataFrame and then corrupt its structure
        normal_data = pd.DataFrame({
            'field1': [1, 2, 3],
            'field2': ['a', 'b', 'c']
        })
        
        # Simulate various corruption scenarios
        corruption_scenarios = [
            # Empty DataFrame
            pd.DataFrame(),
            # DataFrame with no columns
            pd.DataFrame(index=[0, 1, 2]),
            # DataFrame with mismatched index
            pd.DataFrame({'field1': [1, 2]}, index=[0, 1, 2]),
        ]
        
        for corrupted_data in corruption_scenarios:
            # Should handle corrupted structure gracefully
            result = self.validator.validate_dataframe(corrupted_data)
            assert isinstance(result, dict)
    
    def test_validation_with_mixed_data_corruption(self):
        """Test validation with mixed types of data corruption."""
        # Create data with various corruption patterns
        corrupted_data = pd.DataFrame({
            'mixed_corruption': [
                1, 'string', None, [], {}, 
                float('inf'), np.nan, complex(1, 2),
                lambda x: x, type('Test', (), {})()
            ]
        })
        
        # Should handle mixed corruption types
        result = self.validator.validate_dataframe(corrupted_data)
        assert isinstance(result, dict)
        assert 'status' in result
    
    def test_validation_with_index_corruption(self):
        """Test validation with corrupted DataFrame index."""
        # Create DataFrame with problematic index
        problematic_indices = [
            # Duplicate index values
            pd.Index([0, 1, 1, 2, 2]),
            # Mixed type index
            pd.Index([0, 'a', 1.5, None, complex(1, 2)]),
            # Very large index
            pd.Index(range(10000)),
        ]
        
        for idx in problematic_indices:
            try:
                data = pd.DataFrame({
                    'field1': ['test'] * len(idx),
                    'field2': list(range(len(idx)))
                }, index=idx)
                
                result = self.validator.validate_dataframe(data)
                assert isinstance(result, dict)
            except Exception:
                # Some index corruptions might prevent DataFrame creation
                # This is acceptable as long as validation doesn't crash
                pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])