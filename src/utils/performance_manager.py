"""Performance optimization utilities for large-scale data processing."""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional, Iterator, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from functools import wraps
import cProfile
import pstats
import io

from .logger import get_module_logger
from .memory_manager import memory_manager
from .config import config_manager


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    records_processed: int
    records_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float = 0.0
    
    @property
    def throughput_description(self) -> str:
        """Get human-readable throughput description."""
        if self.records_per_second > 1000000:
            return f"{self.records_per_second / 1000000:.2f}M records/sec"
        elif self.records_per_second > 1000:
            return f"{self.records_per_second / 1000:.2f}K records/sec"
        else:
            return f"{self.records_per_second:.2f} records/sec"


@dataclass
class ProgressTracker:
    """Progress tracking for long-running operations."""
    total_items: int
    current_items: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    update_interval: float = 1.0  # seconds
    
    def update(self, items_processed: int = 1) -> None:
        """Update progress with number of items processed."""
        self.current_items += items_processed
        self.last_update_time = time.time()
    
    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_items == 0:
            return 0.0
        return min(100.0, (self.current_items / self.total_items) * 100)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_total_time(self) -> float:
        """Estimate total time based on current progress."""
        if self.current_items == 0:
            return 0.0
        return (self.elapsed_time / self.current_items) * self.total_items
    
    @property
    def eta_seconds(self) -> float:
        """Get estimated time to completion in seconds."""
        return max(0.0, self.estimated_total_time - self.elapsed_time)
    
    @property
    def items_per_second(self) -> float:
        """Get processing rate in items per second."""
        if self.elapsed_time == 0:
            return 0.0
        return self.current_items / self.elapsed_time
    
    def should_update_display(self) -> bool:
        """Check if display should be updated based on interval."""
        return time.time() - self.last_update_time >= self.update_interval
    
    def get_progress_message(self) -> str:
        """Get formatted progress message."""
        eta_str = self._format_time(self.eta_seconds)
        elapsed_str = self._format_time(self.elapsed_time)
        
        return (
            f"Progress: {self.current_items:,}/{self.total_items:,} "
            f"({self.progress_percent:.1f}%) | "
            f"Rate: {self.items_per_second:.1f} items/sec | "
            f"Elapsed: {elapsed_str} | "
            f"ETA: {eta_str}"
        )
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"


class PerformanceProfiler:
    """Performance profiler for detailed analysis."""
    
    def __init__(self, enabled: bool = True):
        """Initialize profiler."""
        self.enabled = enabled
        self.profiler = None
        self.logger = get_module_logger("performance_profiler")
    
    def start_profiling(self) -> None:
        """Start performance profiling."""
        if not self.enabled:
            return
        
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.logger.debug("Started performance profiling")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results."""
        if not self.enabled or not self.profiler:
            return {}
        
        self.profiler.disable()
        
        # Capture profiling results
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        
        self.logger.debug("Stopped performance profiling")
        
        return {
            'profile_output': profile_output,
            'stats': ps
        }
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a specific function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()
                
                # Log profiling results
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats(10)
                
                self.logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        
        return wrapper


class ParallelProcessor:
    """Parallel processing utilities without distributed frameworks."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers. If None, uses CPU count.
            use_processes: Whether to use processes instead of threads.
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.use_processes = use_processes
        self.logger = get_module_logger("parallel_processor")
        
        self.logger.info(f"Initialized parallel processor with {self.max_workers} workers "
                        f"({'processes' if use_processes else 'threads'})")
    
    def process_chunks_parallel(self, 
                              chunks: Iterator[pd.DataFrame], 
                              process_func: Callable[[pd.DataFrame], pd.DataFrame],
                              progress_callback: Optional[Callable[[int], None]] = None) -> Iterator[pd.DataFrame]:
        """Process data chunks in parallel.
        
        Args:
            chunks: Iterator of data chunks
            process_func: Function to process each chunk
            progress_callback: Optional callback for progress updates
            
        Yields:
            Processed chunks
        """
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit initial batch of tasks
            futures = {}
            chunk_count = 0
            
            try:
                # Submit first batch
                for _ in range(self.max_workers * 2):  # Buffer 2x workers
                    try:
                        chunk = next(chunks)
                        future = executor.submit(process_func, chunk)
                        futures[future] = chunk_count
                        chunk_count += 1
                    except StopIteration:
                        break
                
                # Process results and submit new tasks
                while futures:
                    # Wait for at least one task to complete
                    for future in as_completed(futures):
                        chunk_id = futures.pop(future)
                        
                        try:
                            result = future.result()
                            yield result
                            
                            if progress_callback:
                                progress_callback(1)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing chunk {chunk_id}: {e}")
                            continue
                        
                        # Submit next chunk if available
                        try:
                            chunk = next(chunks)
                            new_future = executor.submit(process_func, chunk)
                            futures[new_future] = chunk_count
                            chunk_count += 1
                        except StopIteration:
                            pass
                        
                        break  # Process one result at a time
                        
            except Exception as e:
                self.logger.error(f"Error in parallel processing: {e}")
                # Cancel remaining futures
                for future in futures:
                    future.cancel()
                raise
    
    def map_parallel(self, func: Callable, items: List[Any]) -> List[Any]:
        """Map function over items in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, item) for item in items]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel map: {e}")
                    results.append(None)
            
            return results


class PerformanceManager:
    """Main performance management system."""
    
    def __init__(self):
        """Initialize performance manager."""
        self.logger = get_module_logger("performance_manager")
        self.config = config_manager.config
        self.metrics_history: List[PerformanceMetrics] = []
        self.profiler = PerformanceProfiler()
        self.parallel_processor = ParallelProcessor()
        
        # Performance optimization settings
        self.enable_parallel_processing = True
        self.enable_memory_optimization = True
        self.enable_profiling = False
    
    def benchmark_operation(self, 
                          operation_name: str, 
                          operation_func: Callable,
                          *args, **kwargs) -> PerformanceMetrics:
        """Benchmark a specific operation.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to benchmark
            *args, **kwargs: Arguments for the function
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Starting benchmark for: {operation_name}")
        
        # Get initial memory usage
        initial_memory = memory_manager.get_memory_stats().process_memory
        
        # Start profiling if enabled
        if self.enable_profiling:
            self.profiler.start_profiling()
        
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            
            # Try to get record count from result
            records_processed = 0
            if hasattr(result, '__len__'):
                records_processed = len(result)
            elif isinstance(result, pd.DataFrame):
                records_processed = len(result)
            elif isinstance(result, (list, tuple)):
                records_processed = len(result)
            
        except Exception as e:
            self.logger.error(f"Error in benchmarked operation {operation_name}: {e}")
            raise
        finally:
            end_time = time.time()
            
            # Stop profiling
            if self.enable_profiling:
                profile_results = self.profiler.stop_profiling()
                if profile_results:
                    self.logger.debug(f"Profile results for {operation_name}:\n{profile_results['profile_output']}")
        
        # Calculate metrics
        duration = end_time - start_time
        final_memory = memory_manager.get_memory_stats().process_memory
        memory_usage_mb = (final_memory - initial_memory) / (1024**2)
        
        records_per_second = records_processed / duration if duration > 0 else 0
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            records_processed=records_processed,
            records_per_second=records_per_second,
            memory_usage_mb=memory_usage_mb
        )
        
        self.metrics_history.append(metrics)
        
        self.logger.info(
            f"Benchmark completed for {operation_name}: "
            f"{duration:.2f}s, {records_processed:,} records, "
            f"{metrics.throughput_description}, "
            f"{memory_usage_mb:+.2f} MB memory"
        )
        
        return metrics
    
    def create_progress_tracker(self, total_items: int, update_interval: float = 1.0) -> ProgressTracker:
        """Create a progress tracker for long-running operations.
        
        Args:
            total_items: Total number of items to process
            update_interval: Update interval in seconds
            
        Returns:
            Progress tracker instance
        """
        return ProgressTracker(total_items=total_items, update_interval=update_interval)
    
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for better performance.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        if not self.enable_memory_optimization:
            return df
        
        self.logger.debug(f"Optimizing DataFrame with {len(df)} rows")
        
        # Optimize data types
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    optimized_df[col] = optimized_df[col].astype('uint8')
                elif col_max < 65535:
                    optimized_df[col] = optimized_df[col].astype('uint16')
                elif col_max < 4294967295:
                    optimized_df[col] = optimized_df[col].astype('uint32')
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize string columns to category if beneficial
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].dtype == 'object':
                unique_count = optimized_df[col].nunique()
                total_count = len(optimized_df[col])
                
                # Convert to category if less than 50% unique values
                if unique_count / total_count < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')
        
        # Calculate memory savings
        original_memory = df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        memory_savings = original_memory - optimized_memory
        
        if memory_savings > 0:
            self.logger.info(f"DataFrame optimization saved {memory_savings / (1024**2):.2f} MB "
                           f"({(memory_savings / original_memory) * 100:.1f}% reduction)")
        
        return optimized_df
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {'message': 'No performance metrics available'}
        
        # Calculate aggregate statistics
        total_operations = len(self.metrics_history)
        total_records = sum(m.records_processed for m in self.metrics_history)
        total_duration = sum(m.duration for m in self.metrics_history)
        avg_throughput = np.mean([m.records_per_second for m in self.metrics_history])
        
        # Find best and worst performing operations
        best_throughput = max(self.metrics_history, key=lambda m: m.records_per_second)
        worst_throughput = min(self.metrics_history, key=lambda m: m.records_per_second)
        
        # Memory usage statistics
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        avg_memory = np.mean(memory_usage)
        max_memory = max(memory_usage)
        
        report = {
            'summary': {
                'total_operations': total_operations,
                'total_records_processed': total_records,
                'total_duration_seconds': total_duration,
                'average_throughput_per_second': avg_throughput,
                'average_memory_usage_mb': avg_memory,
                'max_memory_usage_mb': max_memory
            },
            'best_performance': {
                'operation': best_throughput.operation_name,
                'throughput': best_throughput.throughput_description,
                'duration': best_throughput.duration
            },
            'worst_performance': {
                'operation': worst_throughput.operation_name,
                'throughput': worst_throughput.throughput_description,
                'duration': worst_throughput.duration
            },
            'recent_operations': [
                {
                    'operation': m.operation_name,
                    'duration': m.duration,
                    'records': m.records_processed,
                    'throughput': m.throughput_description,
                    'memory_mb': m.memory_usage_mb
                }
                for m in self.metrics_history[-10:]  # Last 10 operations
            ]
        }
        
        return report
    
    def clear_metrics_history(self) -> None:
        """Clear performance metrics history."""
        self.metrics_history.clear()
        self.logger.info("Cleared performance metrics history")
    
    def enable_performance_optimizations(self, 
                                       parallel: bool = True, 
                                       memory: bool = True, 
                                       profiling: bool = False) -> None:
        """Enable or disable performance optimizations.
        
        Args:
            parallel: Enable parallel processing
            memory: Enable memory optimizations
            profiling: Enable performance profiling
        """
        self.enable_parallel_processing = parallel
        self.enable_memory_optimization = memory
        self.enable_profiling = profiling
        
        self.logger.info(f"Performance optimizations: "
                        f"parallel={parallel}, memory={memory}, profiling={profiling}")


# Global performance manager instance
performance_manager = PerformanceManager()