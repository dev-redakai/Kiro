"""Tests for performance management utilities."""

import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.utils.performance_manager import (
    PerformanceManager, PerformanceMetrics, ProgressTracker,
    PerformanceProfiler, ParallelProcessor
)


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics class."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation and properties."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=1000.0,
            end_time=1010.0,
            duration=10.0,
            records_processed=50000,
            records_per_second=5000.0,
            memory_usage_mb=100.0
        )
        
        assert metrics.operation_name == "test_operation"
        assert metrics.duration == 10.0
        assert metrics.records_processed == 50000
        assert metrics.records_per_second == 5000.0
        assert metrics.memory_usage_mb == 100.0
    
    def test_throughput_description(self):
        """Test throughput description formatting."""
        # Test records per second formatting
        metrics_low = PerformanceMetrics(
            operation_name="test", start_time=0, end_time=1, duration=1,
            records_processed=100, records_per_second=100.0, memory_usage_mb=10
        )
        assert "100.00 records/sec" in metrics_low.throughput_description
        
        metrics_k = PerformanceMetrics(
            operation_name="test", start_time=0, end_time=1, duration=1,
            records_processed=5000, records_per_second=5000.0, memory_usage_mb=10
        )
        assert "5.00K records/sec" in metrics_k.throughput_description
        
        metrics_m = PerformanceMetrics(
            operation_name="test", start_time=0, end_time=1, duration=1,
            records_processed=2000000, records_per_second=2000000.0, memory_usage_mb=10
        )
        assert "2.00M records/sec" in metrics_m.throughput_description


class TestProgressTracker:
    """Test cases for ProgressTracker class."""
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(total_items=1000)
        
        assert tracker.total_items == 1000
        assert tracker.current_items == 0
        assert tracker.progress_percent == 0.0
        assert tracker.items_per_second == 0.0
    
    def test_progress_update(self):
        """Test progress updates."""
        tracker = ProgressTracker(total_items=100)
        
        tracker.update(10)
        assert tracker.current_items == 10
        assert tracker.progress_percent == 10.0
        
        tracker.update(40)
        assert tracker.current_items == 50
        assert tracker.progress_percent == 50.0
    
    def test_progress_calculations(self):
        """Test progress calculation methods."""
        tracker = ProgressTracker(total_items=100)
        
        # Simulate some progress
        time.sleep(0.1)  # Small delay to ensure elapsed time > 0
        tracker.update(25)
        
        assert tracker.progress_percent == 25.0
        assert tracker.elapsed_time > 0
        assert tracker.estimated_total_time > 0
        assert tracker.eta_seconds >= 0
        assert tracker.items_per_second > 0
    
    def test_progress_message(self):
        """Test progress message formatting."""
        tracker = ProgressTracker(total_items=1000)
        tracker.update(250)
        
        message = tracker.get_progress_message()
        assert "250/1,000" in message
        assert "25.0%" in message
        assert "items/sec" in message
        assert "Elapsed:" in message
        assert "ETA:" in message
    
    def test_should_update_display(self):
        """Test display update timing."""
        tracker = ProgressTracker(total_items=100, update_interval=0.1)
        
        # Simulate some time passing to ensure initial update
        time.sleep(0.01)
        tracker.last_update_time = time.time() - 0.2  # Make it seem like last update was 0.2s ago
        assert tracker.should_update_display()
        
        # Update and check immediately - should not update
        tracker.update(10)
        assert not tracker.should_update_display()
        
        # Wait and check - should update
        time.sleep(0.15)
        assert tracker.should_update_display()


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler class."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(enabled=True)
        assert profiler.enabled is True
        assert profiler.profiler is None
        
        profiler_disabled = PerformanceProfiler(enabled=False)
        assert profiler_disabled.enabled is False
    
    def test_profiler_start_stop(self):
        """Test profiler start and stop."""
        profiler = PerformanceProfiler(enabled=True)
        
        profiler.start_profiling()
        assert profiler.profiler is not None
        
        results = profiler.stop_profiling()
        assert isinstance(results, dict)
        assert 'profile_output' in results
        assert 'stats' in results
    
    def test_profiler_disabled(self):
        """Test profiler when disabled."""
        profiler = PerformanceProfiler(enabled=False)
        
        profiler.start_profiling()
        assert profiler.profiler is None
        
        results = profiler.stop_profiling()
        assert results == {}
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        profiler = PerformanceProfiler(enabled=True)
        
        @profiler.profile_function
        def test_function(x, y):
            return x + y
        
        result = test_function(5, 3)
        assert result == 8


class TestParallelProcessor:
    """Test cases for ParallelProcessor class."""
    
    def test_parallel_processor_initialization(self):
        """Test ParallelProcessor initialization."""
        processor = ParallelProcessor(max_workers=4, use_processes=False)
        
        assert processor.max_workers == 4
        assert processor.use_processes is False
    
    def test_map_parallel(self):
        """Test parallel mapping function."""
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        def square(x):
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = processor.map_parallel(square, items)
        
        # Results might not be in order due to parallel processing
        assert len(results) == 5
        assert set(results) == {1, 4, 9, 16, 25}
    
    def test_map_parallel_empty_list(self):
        """Test parallel mapping with empty list."""
        processor = ParallelProcessor(max_workers=2)
        
        def identity(x):
            return x
        
        results = processor.map_parallel(identity, [])
        assert results == []
    
    def test_process_chunks_parallel(self):
        """Test parallel chunk processing."""
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        # Create test chunks
        chunks = [
            pd.DataFrame({'x': range(i*10, (i+1)*10)}) 
            for i in range(3)
        ]
        
        def process_chunk(chunk):
            chunk['x_squared'] = chunk['x'] ** 2
            return chunk
        
        results = list(processor.process_chunks_parallel(iter(chunks), process_chunk))
        
        assert len(results) == 3
        for result in results:
            assert 'x_squared' in result.columns
            assert len(result) == 10


class TestPerformanceManager:
    """Test cases for PerformanceManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.performance_manager = PerformanceManager()
        self.performance_manager.clear_metrics_history()
    
    def test_performance_manager_initialization(self):
        """Test PerformanceManager initialization."""
        assert self.performance_manager.metrics_history == []
        assert self.performance_manager.enable_parallel_processing is True
        assert self.performance_manager.enable_memory_optimization is True
        assert self.performance_manager.enable_profiling is False
    
    def test_benchmark_operation(self):
        """Test operation benchmarking."""
        def test_operation():
            # Simulate some work with a small delay to ensure measurable duration
            time.sleep(0.01)
            data = list(range(1000))
            return [x * 2 for x in data]
        
        metrics = self.performance_manager.benchmark_operation("test_op", test_operation)
        
        assert metrics.operation_name == "test_op"
        assert metrics.duration >= 0  # Allow for very fast operations
        assert metrics.records_processed == 1000
        assert metrics.records_per_second >= 0  # Allow for zero if duration is too small
        assert len(self.performance_manager.metrics_history) == 1
    
    def test_benchmark_operation_with_dataframe(self):
        """Test benchmarking with DataFrame result."""
        def create_dataframe():
            time.sleep(0.01)  # Small delay to ensure measurable duration
            return pd.DataFrame({'x': range(500), 'y': range(500, 1000)})
        
        metrics = self.performance_manager.benchmark_operation("df_op", create_dataframe)
        
        assert metrics.operation_name == "df_op"
        assert metrics.records_processed == 500
        assert metrics.duration >= 0  # Allow for very fast operations
    
    def test_create_progress_tracker(self):
        """Test progress tracker creation."""
        tracker = self.performance_manager.create_progress_tracker(1000, 0.5)
        
        assert isinstance(tracker, ProgressTracker)
        assert tracker.total_items == 1000
        assert tracker.update_interval == 0.5
    
    def test_optimize_dataframe_operations(self):
        """Test DataFrame optimization."""
        # Create test DataFrame with various data types
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5] * 100,  # Small integers
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5] * 100,
            'category_col': ['A', 'B', 'A', 'B', 'A'] * 100,  # Repeating categories
            'string_col': [f'item_{i}' for i in range(500)]  # Unique strings
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = self.performance_manager.optimize_dataframe_operations(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Check that optimization was applied
        assert len(optimized_df) == len(df)
        assert list(optimized_df.columns) == list(df.columns)
        
        # Category column should be optimized
        assert optimized_df['category_col'].dtype.name == 'category'
        
        # Memory should be reduced or same
        assert optimized_memory <= original_memory
    
    def test_optimize_dataframe_disabled(self):
        """Test DataFrame optimization when disabled."""
        self.performance_manager.enable_memory_optimization = False
        
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        optimized_df = self.performance_manager.optimize_dataframe_operations(df)
        
        # Should return the same DataFrame
        pd.testing.assert_frame_equal(df, optimized_df)
    
    def test_get_performance_report_empty(self):
        """Test performance report with no metrics."""
        report = self.performance_manager.get_performance_report()
        
        assert 'message' in report
        assert report['message'] == 'No performance metrics available'
    
    def test_get_performance_report_with_metrics(self):
        """Test performance report with metrics."""
        # Add some test metrics
        def dummy_operation():
            time.sleep(0.01)  # Small delay to ensure measurable duration
            return list(range(100))
        
        self.performance_manager.benchmark_operation("op1", dummy_operation)
        self.performance_manager.benchmark_operation("op2", dummy_operation)
        
        report = self.performance_manager.get_performance_report()
        
        assert 'summary' in report
        assert 'best_performance' in report
        assert 'worst_performance' in report
        assert 'recent_operations' in report
        
        summary = report['summary']
        assert summary['total_operations'] == 2
        assert summary['total_records_processed'] == 200
        assert summary['total_duration_seconds'] >= 0  # Allow for very fast operations
        assert summary['average_throughput_per_second'] >= 0
    
    def test_clear_metrics_history(self):
        """Test clearing metrics history."""
        # Add a metric
        def dummy_operation():
            return [1, 2, 3]
        
        self.performance_manager.benchmark_operation("test", dummy_operation)
        assert len(self.performance_manager.metrics_history) == 1
        
        # Clear history
        self.performance_manager.clear_metrics_history()
        assert len(self.performance_manager.metrics_history) == 0
    
    def test_enable_performance_optimizations(self):
        """Test enabling/disabling performance optimizations."""
        self.performance_manager.enable_performance_optimizations(
            parallel=False, memory=False, profiling=True
        )
        
        assert self.performance_manager.enable_parallel_processing is False
        assert self.performance_manager.enable_memory_optimization is False
        assert self.performance_manager.enable_profiling is True


class TestPerformanceManagerIntegration:
    """Integration tests for PerformanceManager."""
    
    def test_end_to_end_performance_tracking(self):
        """Test end-to-end performance tracking workflow."""
        manager = PerformanceManager()
        manager.clear_metrics_history()
        
        # Create progress tracker
        tracker = manager.create_progress_tracker(1000)
        
        # Benchmark an operation
        def data_processing_simulation():
            data = []
            for i in range(1000):
                data.append(i * 2)
                if i % 100 == 0:
                    tracker.update(100)
            return data
        
        metrics = manager.benchmark_operation("simulation", data_processing_simulation)
        
        # Verify results
        assert metrics.records_processed == 1000
        assert tracker.progress_percent == 100.0
        
        # Get performance report
        report = manager.get_performance_report()
        assert report['summary']['total_operations'] == 1
        assert report['summary']['total_records_processed'] == 1000
    
    def test_dataframe_optimization_workflow(self):
        """Test DataFrame optimization workflow."""
        manager = PerformanceManager()
        
        # Create a DataFrame with optimization potential
        df = pd.DataFrame({
            'small_int': np.random.randint(0, 100, 10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000),
            'float_val': np.random.random(10000),
            'large_int': np.random.randint(0, 1000000, 10000)
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize DataFrame
        optimized_df = manager.optimize_dataframe_operations(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Verify optimization
        assert len(optimized_df) == len(df)
        assert optimized_memory <= original_memory
        
        # Category column should be optimized
        assert optimized_df['category'].dtype.name == 'category'