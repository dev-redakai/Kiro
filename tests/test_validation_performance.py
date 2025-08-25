"""
Performance tests for validation under error conditions.

This module contains performance benchmarks and stress tests to ensure
the validation system performs well even under adverse conditions.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
import gc
from datetime import datetime, timedelta
import sys
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.quality.automated_validation import (
    AutomatedDataValidator, ValidationRule, ValidationSeverity
)


class TestValidationPerformanceBenchmarks:
    """Performance benchmarks for validation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
        self.performance_results = []
    
    def teardown_method(self):
        """Clean up after tests."""
        # Force garbage collection
        gc.collect()
    
    def measure_performance(self, test_name, data, rules=None):
        """Measure validation performance and resource usage."""
        process = psutil.Process(os.getpid())
        
        # Measure initial state
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu_percent = process.cpu_percent()
        
        # Run validation with timing
        start_time = time.time()
        result = self.validator.validate_dataframe(data, rules)
        end_time = time.time()
        
        # Measure final state
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu_percent = process.cpu_percent()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = final_memory - initial_memory
        
        performance_data = {
            'test_name': test_name,
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'records_processed': len(data),
            'records_per_second': len(data) / execution_time if execution_time > 0 else 0,
            'validation_status': result.get('status', 'unknown')
        }
        
        self.performance_results.append(performance_data)
        
        # Print performance summary
        print(f"\n{test_name} Performance:")
        print(f"  Execution Time: {execution_time:.3f}s")
        print(f"  Records/Second: {performance_data['records_per_second']:.0f}")
        print(f"  Memory Delta: {memory_delta:.2f}MB")
        print(f"  Status: {performance_data['validation_status']}")
        
        return result, performance_data
    
    def test_baseline_performance_clean_data(self):
        """Benchmark validation performance with clean data."""
        # Create clean, valid data
        clean_data = pd.DataFrame({
            'order_id': [f'ORD{i:06d}' for i in range(10000)],
            'product_name': [f'Product {i}' for i in range(10000)],
            'category': ['Electronics'] * 10000,
            'quantity': np.random.randint(1, 100, 10000),
            'unit_price': np.random.uniform(1.0, 1000.0, 10000),
            'discount_percent': np.random.uniform(0.0, 0.3, 10000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 10000),
            'sale_date': pd.date_range('2023-01-01', periods=10000, freq='H'),
            'customer_email': [f'user{i}@example.com' for i in range(10000)]
        })
        
        result, perf_data = self.measure_performance("Baseline Clean Data", clean_data)
        
        # Performance expectations for clean data
        assert perf_data['execution_time'] < 30.0  # Should complete within 30 seconds
        assert perf_data['records_per_second'] > 100  # At least 100 records/second
        assert perf_data['memory_delta'] < 200  # Memory increase < 200MB
        assert result['status'] in ['passed', 'warning']
    
    def test_performance_with_high_error_rate(self):
        """Benchmark validation performance with high error rates."""
        # Create data with 70% error rate
        error_rate = 0.7
        total_records = 5000
        error_count = int(total_records * error_rate)
        valid_count = total_records - error_count
        
        high_error_data = pd.DataFrame({
            'order_id': [''] * error_count + [f'ORD{i:06d}' for i in range(valid_count)],
            'quantity': [-1] * error_count + list(range(1, valid_count + 1)),
            'unit_price': [0] * error_count + [10.0] * valid_count,
            'sale_date': ['invalid'] * error_count + ['2023-01-01'] * valid_count,
            'category': ['InvalidCategory'] * error_count + ['Electronics'] * valid_count
        })
        
        result, perf_data = self.measure_performance("High Error Rate (70%)", high_error_data)
        
        # Performance should degrade gracefully with high error rates
        assert perf_data['execution_time'] < 60.0  # Should complete within 60 seconds
        assert perf_data['records_per_second'] > 50   # At least 50 records/second
        assert perf_data['memory_delta'] < 300  # Memory increase < 300MB
    
    def test_performance_with_mixed_data_types(self):
        """Benchmark validation performance with mixed data types."""
        # Create data with mixed types that require conversion
        mixed_data = pd.DataFrame({
            'mixed_dates': (
                ['2023-01-01'] * 1000 +
                ['Unknown'] * 500 +
                [datetime(2023, 1, 1)] * 500 +
                [123456789] * 500 +
                ['01/15/2023'] * 500 +
                [None] * 500 +
                ['invalid'] * 500
            ),
            'mixed_numbers': (
                [1] * 1000 +
                ['2'] * 500 +
                [3.0] * 500 +
                ['invalid'] * 500 +
                [None] * 500 +
                [float('inf')] * 500 +
                ['text'] * 500
            )
        })
        
        result, perf_data = self.measure_performance("Mixed Data Types", mixed_data)
        
        # Mixed data types require more processing but should still be reasonable
        assert perf_data['execution_time'] < 45.0  # Should complete within 45 seconds
        assert perf_data['records_per_second'] > 75   # At least 75 records/second
        assert perf_data['memory_delta'] < 250  # Memory increase < 250MB
    
    def test_performance_scaling_with_dataset_size(self):
        """Test how validation performance scales with dataset size."""
        dataset_sizes = [1000, 2500, 5000, 7500, 10000]
        scaling_results = []
        
        for size in dataset_sizes:
            # Create dataset of specified size
            test_data = pd.DataFrame({
                'field1': ['valid'] * (size // 2) + ['invalid'] * (size // 2),
                'field2': list(range(size)),
                'field3': ['2023-01-01'] * (size // 2) + ['invalid_date'] * (size // 2)
            })
            
            result, perf_data = self.measure_performance(f"Scaling Test ({size} records)", test_data)
            scaling_results.append((size, perf_data['execution_time'], perf_data['records_per_second']))
        
        # Analyze scaling characteristics
        # Performance should scale roughly linearly (not exponentially)
        for i in range(1, len(scaling_results)):
            prev_size, prev_time, prev_rps = scaling_results[i-1]
            curr_size, curr_time, curr_rps = scaling_results[i]
            
            size_ratio = curr_size / prev_size
            time_ratio = curr_time / prev_time if prev_time > 0 else 1
            
            # Time should not increase more than 3x the size ratio (allowing for overhead)
            assert time_ratio < size_ratio * 3, f"Performance degraded too much: {time_ratio} vs {size_ratio}"
            
            print(f"Size {prev_size} -> {curr_size}: Time ratio {time_ratio:.2f}, Size ratio {size_ratio:.2f}")
    
    def test_performance_with_memory_constraints(self):
        """Test validation performance under memory constraints."""
        # Create memory-intensive data
        memory_intensive_data = pd.DataFrame({
            'large_text': ['x' * 1000] * 3000,  # Large strings
            'large_numbers': np.random.randn(3000),
            'large_dates': pd.date_range('2020-01-01', periods=3000, freq='D'),
            'mixed_large': [{'key': 'value' * 100}] * 1500 + ['string' * 100] * 1500
        })
        
        result, perf_data = self.measure_performance("Memory Intensive", memory_intensive_data)
        
        # Should handle memory-intensive data reasonably
        assert perf_data['execution_time'] < 40.0  # Should complete within 40 seconds
        assert perf_data['memory_delta'] < 400  # Memory increase < 400MB
        assert result['status'] in ['passed', 'warning', 'error']  # Should not crash
    
    def test_performance_with_concurrent_validation_rules(self):
        """Test performance when many validation rules are active."""
        # Register many validation rules
        for i in range(20):
            rule_name = f"performance_rule_{i}"
            
            def make_rule(rule_id):
                def perf_rule(series, **kwargs):
                    # Simulate some processing time
                    time.sleep(0.001)  # 1ms delay per rule
                    return pd.Series(True, index=series.index), {'rule_id': rule_id}
                return perf_rule
            
            self.validator.register_validation_rule(
                rule_name,
                "test_field",
                ValidationRule.CUSTOM,
                ValidationSeverity.INFO,
                make_rule(i)
            )
        
        # Create test data
        test_data = pd.DataFrame({
            'test_field': ['test_value'] * 1000
        })
        
        # Get list of performance rules
        perf_rules = [f"performance_rule_{i}" for i in range(20)]
        
        result, perf_data = self.measure_performance("Many Rules", test_data, perf_rules)
        
        # Should handle many rules efficiently
        assert perf_data['execution_time'] < 60.0  # Should complete within 60 seconds
        assert result['status'] in ['passed', 'warning', 'error']
    
    def test_performance_with_validation_errors_and_recovery(self):
        """Test performance when validation errors trigger recovery mechanisms."""
        # Create data that will trigger various recovery mechanisms
        recovery_data = pd.DataFrame({
            'dates_needing_recovery': (
                ['Unknown'] * 500 +  # Categorical data requiring recovery
                ['01/15/2023'] * 500 +  # Format conversion recovery
                [123456789] * 500 +  # Numeric timestamp recovery
                ['invalid'] * 500 +  # Complete failure cases
                [None] * 500  # Null handling
            ),
            'numbers_needing_recovery': (
                ['not_a_number'] * 500 +
                [float('inf')] * 500 +
                ['123.45'] * 500 +  # String numbers
                [None] * 500 +
                ['text'] * 500
            )
        })
        
        result, perf_data = self.measure_performance("Error Recovery", recovery_data)
        
        # Recovery mechanisms should not severely impact performance
        assert perf_data['execution_time'] < 50.0  # Should complete within 50 seconds
        assert perf_data['records_per_second'] > 50   # At least 50 records/second
        assert result['status'] in ['passed', 'warning', 'error']
    
    def test_performance_summary_and_analysis(self):
        """Analyze overall performance characteristics."""
        if not self.performance_results:
            pytest.skip("No performance results to analyze")
        
        # Calculate performance statistics
        execution_times = [r['execution_time'] for r in self.performance_results]
        records_per_second = [r['records_per_second'] for r in self.performance_results]
        memory_deltas = [r['memory_delta'] for r in self.performance_results]
        
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_records_per_second = sum(records_per_second) / len(records_per_second)
        avg_memory_delta = sum(memory_deltas) / len(memory_deltas)
        
        print(f"\n=== Performance Summary ===")
        print(f"Tests Run: {len(self.performance_results)}")
        print(f"Average Execution Time: {avg_execution_time:.3f}s")
        print(f"Average Records/Second: {avg_records_per_second:.0f}")
        print(f"Average Memory Delta: {avg_memory_delta:.2f}MB")
        print(f"Max Execution Time: {max(execution_times):.3f}s")
        print(f"Min Records/Second: {min(records_per_second):.0f}")
        print(f"Max Memory Delta: {max(memory_deltas):.2f}MB")
        
        # Performance assertions
        assert avg_execution_time < 45.0, "Average execution time too high"
        assert avg_records_per_second > 60, "Average throughput too low"
        assert avg_memory_delta < 300, "Average memory usage too high"
        assert max(execution_times) < 70.0, "Maximum execution time too high"


class TestValidationStressTests:
    """Stress tests for validation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AutomatedDataValidator()
    
    def test_stress_test_continuous_validation(self):
        """Stress test with continuous validation over time."""
        # Run validation continuously for a period
        start_time = time.time()
        test_duration = 10  # 10 seconds
        validation_count = 0
        
        while time.time() - start_time < test_duration:
            # Create small test dataset
            test_data = pd.DataFrame({
                'field1': ['test'] * 100,
                'field2': list(range(100)),
                'field3': ['2023-01-01'] * 50 + ['invalid'] * 50
            })
            
            # Run validation
            result = self.validator.validate_dataframe(test_data)
            validation_count += 1
            
            # Should not crash
            assert isinstance(result, dict)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
        
        print(f"Completed {validation_count} validations in {test_duration} seconds")
        assert validation_count > 50, "Should complete at least 50 validations in 10 seconds"
    
    def test_stress_test_memory_pressure(self):
        """Stress test under memory pressure conditions."""
        # Create progressively larger datasets
        for size_multiplier in [1, 2, 3, 4, 5]:
            dataset_size = 1000 * size_multiplier
            
            # Create large dataset
            large_data = pd.DataFrame({
                'large_field': ['x' * 100] * dataset_size,
                'numbers': list(range(dataset_size)),
                'mixed_data': (['valid'] * (dataset_size // 2) + 
                              ['invalid'] * (dataset_size // 2))
            })
            
            # Monitor memory before validation
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Run validation
            result = self.validator.validate_dataframe(large_data)
            
            # Monitor memory after validation
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before
            
            print(f"Size {dataset_size}: Memory increase {memory_increase:.2f}MB")
            
            # Should complete without crashing
            assert isinstance(result, dict)
            
            # Memory increase should be reasonable
            assert memory_increase < 500, f"Memory increase too high: {memory_increase}MB"
            
            # Clean up
            del large_data
            gc.collect()
    
    def test_stress_test_error_conditions(self):
        """Stress test with various error conditions."""
        error_scenarios = [
            # High null rate
            pd.DataFrame({'field': [None] * 800 + ['valid'] * 200}),
            
            # High invalid data rate
            pd.DataFrame({'field': ['invalid'] * 900 + ['valid'] * 100}),
            
            # Mixed problematic data
            pd.DataFrame({'field': [
                None, 'invalid', [], {}, lambda x: x, 
                float('inf'), np.nan, complex(1, 2)
            ] * 125}),  # 1000 records total
            
            # Very long strings
            pd.DataFrame({'field': ['x' * 10000] * 1000}),
            
            # Unicode and encoding issues
            pd.DataFrame({'field': ['🚀' * 100, 'café' * 100, '中文' * 100] * 334})
        ]
        
        for i, scenario_data in enumerate(error_scenarios):
            print(f"Running error scenario {i + 1}/{len(error_scenarios)}")
            
            # Should handle each error scenario
            result = self.validator.validate_dataframe(scenario_data)
            
            assert isinstance(result, dict)
            assert 'status' in result
            
            # Clean up
            del scenario_data
            gc.collect()


if __name__ == '__main__':
    # Run performance tests with detailed output
    pytest.main([__file__, '-v', '-s', '--tb=short'])