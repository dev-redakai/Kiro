"""
End-to-end integration tests for the complete data pipeline.

This module tests the complete workflow from data ingestion through dashboard
integration, validating all components work together properly.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pipeline.etl_main import ETLMain
    from dashboard.data_provider import DataProvider
    from utils.test_data_generator import TestDataGenerator
    from utils.config import ConfigManager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for end-to-end testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create complete directory structure
    directories = [
        'data/raw',
        'data/processed', 
        'data/output/csv',
        'data/output/parquet',
        'data/temp',
        'data/checkpoints',
        'config',
        'logs',
        'reports'
    ]
    
    for directory in directories:
        Path(temp_dir, directory).mkdir(parents=True, exist_ok=True)
    
    # Create basic config file
    config_content = f"""
# Pipeline Configuration
chunk_size: 5000
max_memory_usage: 2147483648  # 2GB
anomaly_threshold: 3.0
output_formats:
  - csv
  - parquet

# Data paths
input_data_path: "{temp_dir}/data/raw"
processed_data_path: "{temp_dir}/data/processed"
output_data_path: "{temp_dir}/data/output"
temp_data_path: "{temp_dir}/data/temp"

# Logging
log_level: INFO
log_file: "{temp_dir}/logs/pipeline.log"
"""
    
    config_file = Path(temp_dir, 'config', 'pipeline_config.yaml')
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    yield temp_dir, str(config_file)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def large_sample_dataset():
    """Generate large sample dataset for testing."""
    np.random.seed(42)
    n_records = 50000  # Large enough to test chunking
    
    # Generate realistic e-commerce data
    products = [
        'iPhone 13 Pro', 'Samsung Galaxy S21', 'Dell XPS 13', 'HP Pavilion',
        'Nike Air Max', 'Adidas Ultraboost', 'Sony WH-1000XM4', 'Apple AirPods',
        'LG OLED TV', 'Samsung QLED TV', 'Canon EOS R5', 'Nikon D850',
        'MacBook Pro', 'Surface Laptop', 'iPad Pro', 'Samsung Tab S7'
    ]
    
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate base data
    data = {
        'order_id': [f'ORD_{i:08d}' for i in range(n_records)],
        'product_name': np.random.choice(products, n_records),
        'category': np.random.choice(categories, n_records),
        'quantity': np.random.randint(1, 20, n_records),
        'unit_price': np.random.uniform(10, 2000, n_records),
        'discount_percent': np.random.uniform(0, 0.4, n_records),
        'region': np.random.choice(regions, n_records),
        'sale_date': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_records)
        ],
        'customer_email': [f'customer_{i}@example.com' for i in range(n_records)]
    }
    
    # Add some data quality issues for testing
    dirty_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
    
    df = pd.DataFrame(data)
    
    # Introduce quality issues
    for idx in dirty_indices[:len(dirty_indices)//5]:
        df.loc[idx, 'quantity'] = 0  # Invalid quantity
    
    for idx in dirty_indices[len(dirty_indices)//5:2*len(dirty_indices)//5]:
        df.loc[idx, 'unit_price'] = -10  # Invalid price
    
    for idx in dirty_indices[2*len(dirty_indices)//5:3*len(dirty_indices)//5]:
        df.loc[idx, 'discount_percent'] = 1.5  # Invalid discount
    
    for idx in dirty_indices[3*len(dirty_indices)//5:]:
        df.loc[idx, 'product_name'] = ''  # Missing product name
    
    return df


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestCompleteWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_full_pipeline_execution(self, temp_workspace, large_sample_dataset):
        """Test complete pipeline execution from ingestion to output."""
        temp_dir, config_file = temp_workspace
        
        # Step 1: Save sample data
        input_file = Path(temp_dir, 'data', 'raw', 'sample_ecommerce_data.csv')
        large_sample_dataset.to_csv(input_file, index=False)
        
        logger.info(f"Created input file with {len(large_sample_dataset)} records")
        
        # Step 2: Initialize and run ETL pipeline
        etl_pipeline = ETLMain(config_path=config_file)
        
        start_time = time.time()
        success = etl_pipeline.run_pipeline(str(input_file))
        end_time = time.time()
        
        # Verify pipeline completed successfully
        assert success, "Pipeline should complete successfully"
        
        processing_time = end_time - start_time
        logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
        
        # Step 3: Verify pipeline status
        status = etl_pipeline.get_status()
        assert status['current_stage'] == 'Completed'
        assert status['error_count'] == 0
        assert status['total_records_processed'] > 0
        
        # Step 4: Verify analytical tables were created
        analytical_tables = etl_pipeline.get_analytical_tables()
        
        expected_tables = [
            'monthly_sales_summary',
            'top_products', 
            'region_wise_performance',
            'category_discount_map',
            'anomaly_records'
        ]
        
        for table_name in expected_tables:
            assert table_name in analytical_tables, f"Missing table: {table_name}"
            assert len(analytical_tables[table_name]) > 0, f"Empty table: {table_name}"
        
        # Step 5: Verify output files were created
        output_dir = Path(temp_dir, 'data', 'output')
        
        # Check for CSV outputs
        csv_files = list((output_dir / 'csv').glob('*.csv'))
        assert len(csv_files) > 0, "No CSV output files created"
        
        # Check for Parquet outputs
        parquet_files = list((output_dir / 'parquet').glob('*.parquet'))
        assert len(parquet_files) > 0, "No Parquet output files created"
        
        # Step 6: Verify data quality
        # Load one of the output files to verify data quality
        main_output_file = None
        for csv_file in csv_files:
            if 'processed_data' in csv_file.name:
                main_output_file = csv_file
                break
        
        assert main_output_file is not None, "Main processed data file not found"
        
        processed_data = pd.read_csv(main_output_file)
        
        # Verify data quality improvements
        assert 'revenue' in processed_data.columns, "Revenue column should be calculated"
        assert processed_data['quantity'].min() > 0, "All quantities should be positive"
        assert processed_data['unit_price'].min() > 0, "All prices should be positive"
        assert processed_data['discount_percent'].max() <= 1.0, "Discounts should be <= 100%"
        
        # Step 7: Test dashboard data integration
        data_provider = DataProvider()
        
        # Load analytical tables through data provider
        dashboard_tables = data_provider.load_analytical_tables()
        
        # Verify dashboard can load the data
        assert len(dashboard_tables) > 0, "Dashboard should load analytical tables"
        
        # Verify dashboard metrics
        dashboard_metrics = data_provider.get_dashboard_metrics()
        assert 'total_revenue' in dashboard_metrics, "Dashboard metrics should include revenue"
        assert dashboard_metrics['total_revenue'] > 0, "Total revenue should be positive"
        
        logger.info("Complete workflow test passed successfully")
    
    def test_pipeline_performance_with_large_dataset(self, temp_workspace):
        """Test pipeline performance with large dataset."""
        temp_dir, config_file = temp_workspace
        
        # Generate larger dataset
        np.random.seed(42)
        n_records = 100000  # 100K records
        
        large_data = pd.DataFrame({
            'order_id': [f'ORD_{i:08d}' for i in range(n_records)],
            'product_name': np.random.choice(['Product A', 'Product B', 'Product C'], n_records),
            'category': np.random.choice(['Electronics', 'Fashion'], n_records),
            'quantity': np.random.randint(1, 10, n_records),
            'unit_price': np.random.uniform(10, 500, n_records),
            'discount_percent': np.random.uniform(0, 0.3, n_records),
            'region': np.random.choice(['North', 'South'], n_records),
            'sale_date': [datetime.now() - timedelta(days=i % 365) for i in range(n_records)],
            'customer_email': [f'customer_{i}@example.com' for i in range(n_records)]
        })
        
        input_file = Path(temp_dir, 'data', 'raw', 'large_dataset.csv')
        large_data.to_csv(input_file, index=False)
        
        # Run pipeline with performance monitoring
        etl_pipeline = ETLMain(config_path=config_file)
        
        start_time = time.time()
        success = etl_pipeline.run_pipeline(str(input_file))
        end_time = time.time()
        
        assert success, "Pipeline should handle large dataset"
        
        processing_time = end_time - start_time
        records_per_second = n_records / processing_time
        
        logger.info(f"Processed {n_records} records in {processing_time:.2f} seconds")
        logger.info(f"Throughput: {records_per_second:.2f} records/second")
        
        # Performance assertions
        assert processing_time < 300, f"Processing took too long: {processing_time:.2f} seconds"
        assert records_per_second > 100, f"Throughput too low: {records_per_second:.2f} rec/sec"
        
        # Verify memory efficiency (pipeline should complete without memory errors)
        status = etl_pipeline.get_status()
        assert status['error_count'] == 0, "No memory errors should occur"
    
    def test_dashboard_integration(self, temp_workspace, large_sample_dataset):
        """Test dashboard integration with pipeline outputs."""
        temp_dir, config_file = temp_workspace
        
        # Run pipeline first
        input_file = Path(temp_dir, 'data', 'raw', 'dashboard_test_data.csv')
        large_sample_dataset.to_csv(input_file, index=False)
        
        etl_pipeline = ETLMain(config_path=config_file)
        success = etl_pipeline.run_pipeline(str(input_file))
        assert success, "Pipeline should complete for dashboard test"
        
        # Test dashboard data provider
        data_provider = DataProvider()
        
        # Test loading analytical tables
        tables = data_provider.load_analytical_tables()
        
        required_tables = [
            'monthly_sales_summary',
            'top_products',
            'region_wise_performance', 
            'category_discount_map',
            'anomaly_records'
        ]
        
        for table_name in required_tables:
            assert table_name in tables, f"Dashboard missing table: {table_name}"
            
            table_data = tables[table_name]
            assert isinstance(table_data, pd.DataFrame), f"Table {table_name} should be DataFrame"
            assert len(table_data) > 0, f"Table {table_name} should not be empty"
        
        # Test dashboard metrics calculation
        metrics = data_provider.get_dashboard_metrics()
        
        expected_metrics = ['total_revenue', 'total_orders', 'avg_order_value']
        for metric in expected_metrics:
            assert metric in metrics, f"Dashboard missing metric: {metric}"
            assert metrics[metric] > 0, f"Metric {metric} should be positive"
        
        # Test data refresh functionality
        refresh_success = data_provider.refresh_data()
        assert refresh_success, "Data refresh should succeed"
        
        logger.info("Dashboard integration test passed")
    
    def test_error_recovery_and_monitoring(self, temp_workspace):
        """Test error handling and recovery mechanisms."""
        temp_dir, config_file = temp_workspace
        
        # Create problematic dataset
        problematic_data = pd.DataFrame({
            'order_id': ['ORD_001', 'ORD_002', '', None, 'ORD_005'],
            'product_name': ['Product A', '', None, 'Product D', 'Product E'],
            'quantity': ['5', '0', '-1', 'invalid', '10'],
            'unit_price': ['99.99', '0', '-50', 'bad_price', '199.99'],
            'discount_percent': ['0.1', '2.0', '-0.5', 'invalid', '0.2']
        })
        
        input_file = Path(temp_dir, 'data', 'raw', 'problematic_data.csv')
        problematic_data.to_csv(input_file, index=False)
        
        # Run pipeline with error monitoring
        etl_pipeline = ETLMain(config_path=config_file)
        
        # Pipeline should handle errors gracefully
        success = etl_pipeline.run_pipeline(str(input_file))
        
        # Even with problematic data, pipeline should attempt to complete
        status = etl_pipeline.get_status()
        
        # Check that pipeline attempted processing
        assert status['total_records_processed'] >= 0, "Pipeline should process some records"
        
        # Check that errors were logged but didn't crash the pipeline
        if not success:
            assert len(status['errors']) > 0, "Errors should be logged"
            logger.info(f"Pipeline handled {len(status['errors'])} errors gracefully")
        
        logger.info("Error recovery test completed")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestDataQualityValidation:
    """Test data quality validation throughout the pipeline."""
    
    def test_data_cleaning_effectiveness(self, temp_workspace):
        """Test that data cleaning effectively improves data quality."""
        temp_dir, config_file = temp_workspace
        
        # Create dataset with known quality issues
        dirty_data = pd.DataFrame({
            'order_id': ['ORD_001', 'ORD_002', 'ORD_002', 'ORD_004'],  # Duplicate
            'product_name': ['iPhone', 'samsung tv', 'DELL LAPTOP', ''],  # Case issues, empty
            'category': ['electronics', 'Electronics', 'ELECTRONICS', 'fashion'],  # Case issues
            'quantity': ['5', '0', '-2', '10'],  # Invalid values
            'unit_price': ['99.99', '0', '-50', '299.99'],  # Invalid values
            'discount_percent': ['0.1', '1.5', '-0.1', '0.2'],  # Invalid values
            'region': ['north', 'SOUTH', 'East', 'west'],  # Case issues
            'sale_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
            'customer_email': ['user@example.com', 'invalid-email', 'test@test.com', '']
        })
        
        input_file = Path(temp_dir, 'data', 'raw', 'dirty_data.csv')
        dirty_data.to_csv(input_file, index=False)
        
        # Run pipeline
        etl_pipeline = ETLMain(config_path=config_file)
        success = etl_pipeline.run_pipeline(str(input_file))
        
        assert success, "Pipeline should handle dirty data"
        
        # Load processed output
        output_files = list(Path(temp_dir, 'data', 'output', 'csv').glob('processed_data_*.csv'))
        assert len(output_files) > 0, "Processed data file should exist"
        
        processed_data = pd.read_csv(output_files[0])
        
        # Verify data quality improvements
        assert processed_data['quantity'].min() > 0, "All quantities should be positive"
        assert processed_data['unit_price'].min() > 0, "All prices should be positive"
        assert processed_data['discount_percent'].min() >= 0, "Discounts should be non-negative"
        assert processed_data['discount_percent'].max() <= 1.0, "Discounts should be <= 100%"
        
        # Verify revenue calculation
        assert 'revenue' in processed_data.columns, "Revenue should be calculated"
        expected_revenue = (processed_data['quantity'] * 
                          processed_data['unit_price'] * 
                          (1 - processed_data['discount_percent']))
        
        # Allow for small floating point differences
        revenue_diff = abs(processed_data['revenue'] - expected_revenue).max()
        assert revenue_diff < 0.01, "Revenue calculation should be accurate"
        
        logger.info("Data quality validation test passed")
    
    def test_anomaly_detection_accuracy(self, temp_workspace):
        """Test anomaly detection accuracy."""
        temp_dir, config_file = temp_workspace
        
        # Create dataset with known anomalies
        np.random.seed(42)
        normal_data = pd.DataFrame({
            'order_id': [f'ORD_{i:06d}' for i in range(1000)],
            'product_name': np.random.choice(['Product A', 'Product B'], 1000),
            'category': ['Electronics'] * 1000,
            'quantity': np.random.randint(1, 10, 1000),
            'unit_price': np.random.uniform(50, 200, 1000),
            'discount_percent': np.random.uniform(0, 0.2, 1000),
            'region': np.random.choice(['North', 'South'], 1000),
            'sale_date': [datetime.now() - timedelta(days=i % 30) for i in range(1000)],
            'customer_email': [f'customer_{i}@example.com' for i in range(1000)]
        })
        
        # Add known anomalies
        anomaly_indices = [100, 200, 300, 400, 500]
        for idx in anomaly_indices:
            normal_data.loc[idx, 'unit_price'] = 10000  # Extremely high price
            normal_data.loc[idx, 'quantity'] = 100  # Extremely high quantity
        
        input_file = Path(temp_dir, 'data', 'raw', 'anomaly_test_data.csv')
        normal_data.to_csv(input_file, index=False)
        
        # Run pipeline
        etl_pipeline = ETLMain(config_path=config_file)
        success = etl_pipeline.run_pipeline(str(input_file))
        
        assert success, "Pipeline should complete anomaly detection"
        
        # Check anomaly detection results
        analytical_tables = etl_pipeline.get_analytical_tables()
        anomaly_records = analytical_tables.get('anomaly_records', pd.DataFrame())
        
        assert len(anomaly_records) > 0, "Should detect some anomalies"
        
        # Verify that high-value records are flagged as anomalies
        if 'revenue' in anomaly_records.columns:
            high_revenue_anomalies = anomaly_records[anomaly_records['revenue'] > 500000]
            assert len(high_revenue_anomalies) > 0, "High revenue records should be flagged"
        
        logger.info(f"Detected {len(anomaly_records)} anomalies")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
class TestSystemIntegration:
    """Test system-level integration aspects."""
    
    def test_configuration_management(self, temp_workspace):
        """Test configuration management across components."""
        temp_dir, config_file = temp_workspace
        
        # Test custom configuration
        custom_config = f"""
chunk_size: 2000
max_memory_usage: 1073741824  # 1GB
anomaly_threshold: 2.5
output_formats:
  - csv

input_data_path: "{temp_dir}/data/raw"
processed_data_path: "{temp_dir}/data/processed"
output_data_path: "{temp_dir}/data/output"
temp_data_path: "{temp_dir}/data/temp"

log_level: DEBUG
"""
        
        custom_config_file = Path(temp_dir, 'config', 'custom_config.yaml')
        with open(custom_config_file, 'w') as f:
            f.write(custom_config)
        
        # Initialize pipeline with custom config
        etl_pipeline = ETLMain(config_path=str(custom_config_file))
        
        # Verify configuration is loaded correctly
        assert etl_pipeline.config.chunk_size == 2000, "Custom chunk size should be loaded"
        assert etl_pipeline.config.anomaly_threshold == 2.5, "Custom anomaly threshold should be loaded"
        
        logger.info("Configuration management test passed")
    
    def test_logging_and_monitoring(self, temp_workspace, large_sample_dataset):
        """Test logging and monitoring functionality."""
        temp_dir, config_file = temp_workspace
        
        # Run pipeline with logging
        input_file = Path(temp_dir, 'data', 'raw', 'logging_test_data.csv')
        large_sample_dataset.head(10000).to_csv(input_file, index=False)
        
        etl_pipeline = ETLMain(config_path=config_file)
        success = etl_pipeline.run_pipeline(str(input_file))
        
        assert success, "Pipeline should complete with logging"
        
        # Check that log files were created
        log_files = list(Path(temp_dir, 'logs').glob('*.log'))
        assert len(log_files) > 0, "Log files should be created"
        
        # Check log content
        log_content = ""
        for log_file in log_files:
            with open(log_file, 'r') as f:
                log_content += f.read()
        
        # Verify important log messages are present
        assert "Starting ETL pipeline" in log_content, "Pipeline start should be logged"
        assert "ETL pipeline completed" in log_content, "Pipeline completion should be logged"
        
        logger.info("Logging and monitoring test passed")
    
    def test_output_format_compatibility(self, temp_workspace, large_sample_dataset):
        """Test output format compatibility."""
        temp_dir, config_file = temp_workspace
        
        # Run pipeline
        input_file = Path(temp_dir, 'data', 'raw', 'format_test_data.csv')
        large_sample_dataset.head(5000).to_csv(input_file, index=False)
        
        etl_pipeline = ETLMain(config_path=config_file)
        success = etl_pipeline.run_pipeline(str(input_file))
        
        assert success, "Pipeline should complete"
        
        # Test CSV format
        csv_files = list(Path(temp_dir, 'data', 'output', 'csv').glob('*.csv'))
        assert len(csv_files) > 0, "CSV files should be created"
        
        # Verify CSV can be read back
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            assert len(df) > 0, f"CSV file {csv_file.name} should contain data"
        
        # Test Parquet format
        parquet_files = list(Path(temp_dir, 'data', 'output', 'parquet').glob('*.parquet'))
        assert len(parquet_files) > 0, "Parquet files should be created"
        
        # Verify Parquet can be read back
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            assert len(df) > 0, f"Parquet file {parquet_file.name} should contain data"
        
        logger.info("Output format compatibility test passed")


def run_integration_tests():
    """Run all integration tests."""
    if not IMPORTS_AVAILABLE:
        print("Skipping integration tests - required modules not available")
        return False
    
    # Run pytest with verbose output
    result = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x'  # Stop on first failure
    ])
    
    return result == 0


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)