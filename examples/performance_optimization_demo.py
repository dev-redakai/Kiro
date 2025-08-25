"""
Demonstration of memory management and performance optimization features.

This script shows how the MemoryManager and PerformanceManager work together
to optimize data processing for large datasets.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import memory_manager, performance_manager, get_logger
from src.pipeline.ingestion import ChunkedCSVReader, DataIngestionManager

# Set up logging
logger = get_logger("performance_demo")


def create_sample_data(file_path: str, num_rows: int = 100000):
    """Create a sample CSV file for testing."""
    logger.info(f"Creating sample data with {num_rows:,} rows")
    
    # Generate sample e-commerce data
    np.random.seed(42)  # For reproducible results
    
    data = {
        'order_id': [f'ORD{i:06d}' for i in range(num_rows)],
        'product_name': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Camera'], num_rows),
        'category': np.random.choice(['Electronics', 'Fashion', 'Home'], num_rows),
        'quantity': np.random.randint(1, 10, num_rows),
        'unit_price': np.round(np.random.uniform(10, 1000, num_rows), 2),
        'discount_percent': np.round(np.random.uniform(0, 0.3, num_rows), 3),
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_rows),
        'sale_date': pd.date_range('2023-01-01', periods=num_rows, freq='1H').strftime('%Y-%m-%d'),
        'customer_email': [f'customer{i}@example.com' for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    file_size_mb = Path(file_path).stat().st_size / (1024**2)
    logger.info(f"Created sample file: {file_size_mb:.1f} MB")
    
    return file_path


def demonstrate_memory_management():
    """Demonstrate memory management features."""
    logger.info("=== MEMORY MANAGEMENT DEMONSTRATION ===")
    
    # Get initial memory stats
    initial_stats = memory_manager.get_memory_stats()
    logger.info(f"Initial memory usage: {initial_stats.process_memory / (1024**2):.1f} MB")
    
    # Create some data to demonstrate memory monitoring
    with memory_manager.memory_checkpoint("data_creation"):
        large_data = [np.random.random(10000) for _ in range(100)]
        logger.info(f"Created large data structure with {len(large_data)} arrays")
    
    # Check memory usage
    memory_status = memory_manager.check_memory_usage()
    logger.info(f"Memory status: {memory_status['status']}, "
               f"Usage: {memory_status['process_memory_mb']:.1f} MB")
    
    # Demonstrate garbage collection
    del large_data
    gc_stats = memory_manager.force_garbage_collection()
    logger.info(f"Garbage collection freed: {gc_stats['memory_freed_mb']:.2f} MB")
    
    # Demonstrate optimal chunk size calculation
    file_size = 500 * 1024 * 1024  # 500MB file
    optimal_chunk_size = memory_manager.get_optimal_chunk_size(file_size, 50000)
    logger.info(f"Optimal chunk size for 500MB file: {optimal_chunk_size:,} rows")
    
    # Generate memory report
    memory_report = memory_manager.get_memory_report()
    logger.info(f"System memory: {memory_report['system_memory']['total_gb']:.1f} GB total, "
               f"{memory_report['system_memory']['available_gb']:.1f} GB available")


def demonstrate_performance_optimization():
    """Demonstrate performance optimization features."""
    logger.info("=== PERFORMANCE OPTIMIZATION DEMONSTRATION ===")
    
    # Create test data
    def create_test_dataframe():
        """Create a test DataFrame for optimization."""
        return pd.DataFrame({
            'small_int': np.random.randint(0, 100, 50000),
            'category': np.random.choice(['A', 'B', 'C'], 50000),
            'float_val': np.random.random(50000),
            'large_int': np.random.randint(0, 1000000, 50000)
        })
    
    # Benchmark DataFrame creation
    metrics = performance_manager.benchmark_operation("create_dataframe", create_test_dataframe)
    logger.info(f"DataFrame creation: {metrics.throughput_description}, "
               f"Duration: {metrics.duration:.3f}s")
    
    # Demonstrate DataFrame optimization
    df = create_test_dataframe()
    original_memory = df.memory_usage(deep=True).sum()
    
    optimized_df = performance_manager.optimize_dataframe_operations(df)
    optimized_memory = optimized_df.memory_usage(deep=True).sum()
    
    memory_savings = original_memory - optimized_memory
    logger.info(f"DataFrame optimization saved: {memory_savings / (1024**2):.2f} MB "
               f"({(memory_savings / original_memory) * 100:.1f}% reduction)")
    
    # Demonstrate progress tracking
    logger.info("Demonstrating progress tracking...")
    tracker = performance_manager.create_progress_tracker(1000, 0.5)
    
    for i in range(0, 1000, 100):
        time.sleep(0.1)  # Simulate work
        tracker.update(100)
        if tracker.should_update_display():
            logger.info(tracker.get_progress_message())
    
    # Demonstrate parallel processing
    def square_number(x):
        time.sleep(0.001)  # Simulate some work
        return x * x
    
    numbers = list(range(100))
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [square_number(x) for x in numbers]
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    parallel_results = performance_manager.parallel_processor.map_parallel(square_number, numbers)
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1
    logger.info(f"Parallel processing speedup: {speedup:.2f}x "
               f"(Sequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s)")


def demonstrate_integrated_pipeline():
    """Demonstrate integrated pipeline with optimizations."""
    logger.info("=== INTEGRATED PIPELINE DEMONSTRATION ===")
    
    # Create sample data
    sample_file = "data/temp/performance_demo_sample.csv"
    Path(sample_file).parent.mkdir(parents=True, exist_ok=True)
    
    create_sample_data(sample_file, 50000)
    
    # Initialize ingestion manager
    ingestion_manager = DataIngestionManager({
        'chunk_size': 10000,
        'max_retries': 2
    })
    
    # Process data with optimizations
    logger.info("Processing data with memory and performance optimizations...")
    
    total_rows = 0
    chunk_count = 0
    
    try:
        for chunk in ingestion_manager.ingest_data(sample_file):
            chunk_count += 1
            total_rows += len(chunk)
            
            # Simulate some processing
            processed_chunk = chunk.copy()
            processed_chunk['revenue'] = (
                processed_chunk['quantity'] * 
                processed_chunk['unit_price'] * 
                (1 - processed_chunk['discount_percent'])
            )
            
            if chunk_count % 3 == 0:
                logger.info(f"Processed chunk {chunk_count}, total rows: {total_rows:,}")
    
    except Exception as e:
        logger.error(f"Error in pipeline demonstration: {e}")
    
    # Get ingestion statistics
    stats = ingestion_manager.get_ingestion_stats()
    logger.info(f"Ingestion completed: {stats['total_rows_processed']:,} rows, "
               f"{stats['average_rows_per_second']:,.0f} rows/sec, "
               f"Peak memory: {stats['memory_peak_mb']:.1f} MB")
    
    # Get performance report
    perf_report = performance_manager.get_performance_report()
    if 'summary' in perf_report:
        summary = perf_report['summary']
        logger.info(f"Performance summary: {summary['total_operations']} operations, "
                   f"Average throughput: {summary['average_throughput_per_second']:,.0f} records/sec")
    
    # Clean up
    Path(sample_file).unlink(missing_ok=True)


def main():
    """Run all demonstrations."""
    logger.info("Starting Performance Optimization Demonstration")
    logger.info("=" * 60)
    
    try:
        # Enable performance optimizations
        performance_manager.enable_performance_optimizations(
            parallel=True, 
            memory=True, 
            profiling=False  # Disable profiling for cleaner output
        )
        
        # Run demonstrations
        demonstrate_memory_management()
        print()
        
        demonstrate_performance_optimization()
        print()
        
        demonstrate_integrated_pipeline()
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise
    
    finally:
        # Generate final reports
        logger.info("=" * 60)
        logger.info("FINAL REPORTS")
        logger.info("=" * 60)
        
        # Memory report
        memory_report = memory_manager.get_memory_report()
        logger.info(f"Final memory usage: {memory_report['process_memory']['rss_mb']:.1f} MB")
        
        # Performance report
        perf_report = performance_manager.get_performance_report()
        if 'summary' in perf_report:
            logger.info(f"Total operations benchmarked: {perf_report['summary']['total_operations']}")
        
        logger.info("Performance Optimization Demonstration completed successfully!")


if __name__ == "__main__":
    main()