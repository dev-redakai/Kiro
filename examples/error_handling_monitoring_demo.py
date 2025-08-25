"""
Demonstration of the comprehensive error handling and monitoring system.

This example shows how to use the error handling and monitoring components
together in a typical data processing scenario.
"""

import pandas as pd
import time
import random
from datetime import datetime
from pathlib import Path

# Import our error handling and monitoring components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.error_handler import (
    ErrorHandler, DataQualityError, PipelineMemoryError, ProcessingError,
    error_handler, handle_data_quality_error
)
from src.utils.monitoring import (
    pipeline_monitor, start_monitoring, stop_monitoring,
    log_performance_metric, log_data_quality_metric, time_operation
)
from src.utils.logger import get_module_logger
from src.utils.logging_config import advanced_logging


def simulate_data_processing_with_errors():
    """Simulate a data processing pipeline with various error scenarios."""
    
    logger = get_module_logger("demo")
    logger.info("Starting error handling and monitoring demonstration")
    
    # Start monitoring
    start_monitoring()
    
    try:
        # Simulate data ingestion with potential errors
        with time_operation("data_ingestion", 10000) as timer:
            logger.info("Starting data ingestion simulation")
            
            # Simulate some processing time
            time.sleep(1)
            
            # Simulate a memory error scenario
            try:
                # This would normally be actual memory-intensive operation
                if random.random() < 0.3:  # 30% chance of memory error
                    raise PipelineMemoryError(
                        "Insufficient memory for chunk processing",
                        current_usage=8.5,
                        max_usage=8.0
                    )
                
                logger.info("Data ingestion completed successfully")
                timer.update_records_count(10000)
                
            except PipelineMemoryError as e:
                logger.error(f"Memory error during ingestion: {e}")
                
                # Handle the error - this will apply recovery strategies
                recovery_result = error_handler.handle_memory_error(e, {
                    'operation': 'data_ingestion',
                    'chunk_size': 50000
                })
                
                if recovery_result and recovery_result.get('new_chunk_size'):
                    logger.info(f"Recovered with new chunk size: {recovery_result['new_chunk_size']}")
                    timer.update_records_count(5000)  # Reduced due to smaller chunks
        
        # Simulate data cleaning with quality issues
        with time_operation("data_cleaning", 8000) as timer:
            logger.info("Starting data cleaning simulation")
            
            valid_records = 0
            invalid_records = 0
            
            # Simulate processing records with quality issues
            for i in range(100):
                try:
                    # Simulate various data quality issues
                    if random.random() < 0.1:  # 10% chance of data quality error
                        error_types = [
                            ("Invalid quantity", "quantity", "-5", 0),
                            ("Invalid email", "customer_email", "invalid-email", None),
                            ("Invalid discount", "discount_percent", "1.5", 0.15)
                        ]
                        
                        error_msg, field, value, correction = random.choice(error_types)
                        
                        raise DataQualityError(
                            error_msg,
                            field=field,
                            value=value,
                            correction=correction
                        )
                    
                    valid_records += 1
                    
                except DataQualityError as e:
                    # Handle data quality error
                    correction = handle_data_quality_error(e, {
                        'record_id': i,
                        'processing_stage': 'cleaning'
                    })
                    
                    if correction == "SKIP_RECORD":
                        invalid_records += 1
                        logger.debug(f"Skipped invalid record {i}")
                    elif correction is not None:
                        valid_records += 1
                        logger.debug(f"Applied correction for record {i}: {correction}")
                    else:
                        invalid_records += 1
                
                # Simulate processing time
                if i % 20 == 0:
                    time.sleep(0.1)
            
            # Log data quality metrics
            log_data_quality_metric(
                "data_cleaning_validation",
                total_records=100,
                valid_records=valid_records,
                invalid_records=invalid_records,
                details={
                    'processing_stage': 'cleaning',
                    'validation_rules_applied': ['quantity', 'email', 'discount']
                }
            )
            
            logger.info(f"Data cleaning completed: {valid_records} valid, {invalid_records} invalid")
            timer.update_records_count(valid_records)
        
        # Simulate transformation with processing errors
        with time_operation("data_transformation", 7500) as timer:
            logger.info("Starting data transformation simulation")
            
            try:
                # Simulate processing time
                time.sleep(0.8)
                
                # Simulate a processing error
                if random.random() < 0.2:  # 20% chance of processing error
                    raise ProcessingError(
                        "Failed to calculate aggregated metrics",
                        operation="monthly_aggregation",
                        data_info={'records': 7500, 'columns': 12}
                    )
                
                logger.info("Data transformation completed successfully")
                timer.update_records_count(7500)
                
            except ProcessingError as e:
                logger.error(f"Processing error during transformation: {e}")
                
                # Handle processing error
                recovery_result = error_handler.handle_processing_error(e, {
                    'operation': 'data_transformation',
                    'retry_count': 0
                })
                
                if recovery_result and recovery_result.get('use_fallback'):
                    logger.info("Using fallback transformation method")
                    time.sleep(0.5)  # Simulate fallback processing
                    timer.update_records_count(7000)  # Slightly reduced output
        
        # Log some performance metrics
        log_performance_metric("memory_efficiency", 85.5, "percent", "memory")
        log_performance_metric("cpu_utilization", 45.2, "percent", "cpu")
        log_performance_metric("disk_io_rate", 125.8, "MB/s", "io")
        
        # Simulate anomaly detection
        logger.info("Running anomaly detection")
        anomalies_found = random.randint(5, 15)
        total_records = 7000
        
        logger.log_anomaly_detection(anomalies_found, total_records)
        
        log_performance_metric("anomaly_detection_rate", 
                             (anomalies_found / total_records) * 100, 
                             "percent", "quality")
        
        logger.info("Pipeline processing completed successfully")
        
    except Exception as e:
        logger.exception(f"Unexpected error in pipeline: {e}")
        
    finally:
        # Stop monitoring
        stop_monitoring()
        
        # Print monitoring summary
        print_monitoring_summary()


def print_monitoring_summary():
    """Print a summary of monitoring data collected during the demo."""
    
    print("\n" + "="*60)
    print("MONITORING SUMMARY")
    print("="*60)
    
    # Processing summary
    processing_summary = pipeline_monitor.get_processing_summary()
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"  Total Operations: {processing_summary.get('total_operations', 0)}")
    print(f"  Completed: {processing_summary.get('completed_operations', 0)}")
    print(f"  Failed: {processing_summary.get('failed_operations', 0)}")
    print(f"  Success Rate: {processing_summary.get('success_rate', 0):.1f}%")
    print(f"  Total Records: {processing_summary.get('total_records_processed', 0):,}")
    print(f"  Avg Throughput: {processing_summary.get('average_throughput', 0):.2f} records/sec")
    
    # Data quality summary
    data_quality_summary = pipeline_monitor.get_data_quality_summary()
    if data_quality_summary.get('message') != 'No data quality metrics recorded':
        print(f"\n🎯 DATA QUALITY SUMMARY:")
        print(f"  Total Records Analyzed: {data_quality_summary.get('total_records_analyzed', 0):,}")
        print(f"  Valid Records: {data_quality_summary.get('total_valid_records', 0):,}")
        print(f"  Overall Error Rate: {data_quality_summary.get('overall_error_rate', 0):.2f}%")
    
    # Performance summary
    performance_summary = pipeline_monitor.get_performance_summary()
    if performance_summary.get('message') != 'No performance metrics recorded':
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"  Total Metrics: {performance_summary.get('total_metrics', 0)}")
        by_category = performance_summary.get('by_category', {})
        for category, stats in by_category.items():
            print(f"  {category.title()}: {stats['count']} metrics, avg: {stats['avg_value']:.2f}")
    
    # Error summary
    error_summary = error_handler.get_error_summary()
    if error_summary.get('message') != 'No errors recorded':
        print(f"\n❌ ERROR SUMMARY:")
        print(f"  Total Errors: {error_summary.get('total_errors', 0)}")
        print(f"  Resolved: {error_summary.get('resolved_count', 0)}")
        
        error_types = error_summary.get('by_category', {})
        if error_types:
            print("  By Category:")
            for category, count in error_types.items():
                print(f"    {category}: {count}")
    
    # System metrics
    system_metrics = pipeline_monitor.system_monitor.get_current_metrics()
    print(f"\n🖥️ SYSTEM HEALTH:")
    print(f"  Memory Usage: {system_metrics.get('memory_percent', 0):.1f}%")
    print(f"  CPU Usage: {system_metrics.get('cpu_percent', 0):.1f}%")
    print(f"  Available Memory: {system_metrics.get('memory_available_gb', 0):.1f} GB")
    
    print("\n" + "="*60)


def demonstrate_error_recovery():
    """Demonstrate specific error recovery scenarios."""
    
    print("\n" + "="*60)
    print("ERROR RECOVERY DEMONSTRATION")
    print("="*60)
    
    logger = get_module_logger("error_demo")
    
    # Test data quality error recovery
    print("\n1. Testing Data Quality Error Recovery:")
    try:
        raise DataQualityError(
            "Invalid quantity value",
            field="quantity",
            value="-10",
            correction=0
        )
    except DataQualityError as e:
        result = handle_data_quality_error(e)
        print(f"   Error: {e}")
        print(f"   Recovery result: {result}")
    
    # Test memory error recovery
    print("\n2. Testing Memory Error Recovery:")
    try:
        raise PipelineMemoryError(
            "Out of memory during chunk processing",
            current_usage=7.8,
            max_usage=8.0
        )
    except PipelineMemoryError as e:
        result = error_handler.handle_memory_error(e, {'chunk_size': 100000})
        print(f"   Error: {e}")
        print(f"   Recovery result: {result}")
    
    # Test processing error recovery
    print("\n3. Testing Processing Error Recovery:")
    try:
        raise ProcessingError(
            "Transformation failed due to invalid data format",
            operation="data_transformation"
        )
    except ProcessingError as e:
        result = error_handler.handle_processing_error(e, {'retry_count': 0})
        print(f"   Error: {e}")
        print(f"   Recovery result: {result}")
    
    print("\n" + "="*60)


def export_monitoring_reports():
    """Export monitoring reports to files."""
    
    print("\n📄 Exporting monitoring reports...")
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Export monitoring report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    monitoring_report_path = reports_dir / f"monitoring_report_{timestamp}.json"
    pipeline_monitor.export_monitoring_report(str(monitoring_report_path))
    
    # Export error log
    error_log_path = reports_dir / f"error_log_{timestamp}.json"
    error_handler.export_error_log(str(error_log_path))
    
    print(f"   Monitoring report: {monitoring_report_path}")
    print(f"   Error log: {error_log_path}")


if __name__ == "__main__":
    print("🚀 Starting Error Handling and Monitoring Demonstration")
    print("This demo will simulate various error scenarios and show recovery mechanisms.")
    
    # Run the main demonstration
    simulate_data_processing_with_errors()
    
    # Demonstrate specific error recovery scenarios
    demonstrate_error_recovery()
    
    # Export reports
    export_monitoring_reports()
    
    print("\n✅ Demonstration completed!")
    print("Check the logs/ directory for detailed logs and reports/ for exported reports.")