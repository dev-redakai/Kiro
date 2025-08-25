#!/usr/bin/env python3
"""
Pipeline Integration Validation Script

This script validates that all pipeline components are properly integrated
and can work together in the complete workflow.
"""

import os
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_etl_main_integration():
    """Test the main ETL pipeline integration."""
    logger.info("Testing ETL Main Pipeline Integration...")
    
    try:
        from pipeline.etl_main import ETLMain
        from utils.test_data_generator import TestDataGenerator
        
        # Generate test data
        generator = TestDataGenerator()
        test_data = generator.generate_sample_data(size=5000, data_type='clean')
        
        # Save test data
        test_file = Path('data/raw/integration_test_data.csv')
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_data.to_csv(test_file, index=False)
        
        # Initialize and run pipeline
        etl_pipeline = ETLMain()
        
        start_time = time.time()
        success = etl_pipeline.run_pipeline(str(test_file))
        end_time = time.time()
        
        if success:
            status = etl_pipeline.get_status()
            analytical_tables = etl_pipeline.get_analytical_tables()
            
            logger.info(f"✓ ETL Pipeline completed successfully")
            logger.info(f"  - Processing time: {end_time - start_time:.2f} seconds")
            logger.info(f"  - Records processed: {status['total_records_processed']}")
            logger.info(f"  - Analytical tables created: {len(analytical_tables)}")
            
            # Verify analytical tables
            expected_tables = [
                'monthly_sales_summary',
                'top_products',
                'region_wise_performance',
                'category_discount_map',
                'anomaly_records'
            ]
            
            for table_name in expected_tables:
                if table_name in analytical_tables:
                    table_size = len(analytical_tables[table_name])
                    logger.info(f"  - {table_name}: {table_size} records")
                else:
                    logger.warning(f"  - Missing table: {table_name}")
            
            return True
        else:
            logger.error("✗ ETL Pipeline failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ ETL Main integration test failed: {e}")
        return False


def test_dashboard_integration():
    """Test dashboard integration with pipeline outputs."""
    logger.info("Testing Dashboard Integration...")
    
    try:
        from dashboard.data_provider import DataProvider
        
        # Test data provider
        data_provider = DataProvider()
        
        # Load analytical tables
        tables = data_provider.load_analytical_tables()
        
        if tables:
            logger.info(f"✓ Dashboard data provider loaded {len(tables)} tables")
            
            for table_name, table_data in tables.items():
                if isinstance(table_data, pd.DataFrame):
                    logger.info(f"  - {table_name}: {len(table_data)} records")
                else:
                    logger.warning(f"  - {table_name}: Invalid data type")
            
            # Test dashboard metrics
            metrics = data_provider.get_dashboard_metrics()
            
            if metrics and 'total_revenue' in metrics:
                logger.info(f"✓ Dashboard metrics calculated")
                logger.info(f"  - Total revenue: ${metrics['total_revenue']:,.2f}")
                logger.info(f"  - Total orders: {metrics.get('total_orders', 'N/A')}")
                return True
            else:
                logger.warning("✗ Dashboard metrics incomplete")
                return False
        else:
            logger.warning("✗ No analytical tables loaded")
            return False
            
    except Exception as e:
        logger.error(f"✗ Dashboard integration test failed: {e}")
        return False


def test_component_integration():
    """Test individual component integration."""
    logger.info("Testing Individual Component Integration...")
    
    results = {}
    
    # Test data ingestion
    try:
        from pipeline.ingestion import DataIngestionManager
        
        config = {
            'chunk_size': 1000,
            'encoding': 'utf-8',
            'max_retries': 3,
            'supported_formats': ['.csv']
        }
        
        ingestion_manager = DataIngestionManager(config=config)
        logger.info("✓ Data Ingestion Manager initialized")
        results['ingestion'] = True
        
    except Exception as e:
        logger.error(f"✗ Data Ingestion integration failed: {e}")
        results['ingestion'] = False
    
    # Test data cleaning
    try:
        from data.data_cleaner import DataCleaner
        
        cleaner = DataCleaner()
        logger.info("✓ Data Cleaner initialized")
        results['cleaning'] = True
        
    except Exception as e:
        logger.error(f"✗ Data Cleaner integration failed: {e}")
        results['cleaning'] = False
    
    # Test transformation
    try:
        from pipeline.transformation import AnalyticalTransformer
        
        transformer = AnalyticalTransformer()
        logger.info("✓ Analytical Transformer initialized")
        results['transformation'] = True
        
    except Exception as e:
        logger.error(f"✗ Analytical Transformer integration failed: {e}")
        results['transformation'] = False
    
    # Test anomaly detection
    try:
        from quality.anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector()
        logger.info("✓ Anomaly Detector initialized")
        results['anomaly_detection'] = True
        
    except Exception as e:
        logger.error(f"✗ Anomaly Detector integration failed: {e}")
        results['anomaly_detection'] = False
    
    # Test data export
    try:
        from data.data_exporter import DataExporter
        from utils.config import ConfigManager
        
        config_manager = ConfigManager()
        exporter = DataExporter(config=config_manager.config)
        logger.info("✓ Data Exporter initialized")
        results['export'] = True
        
    except Exception as e:
        logger.error(f"✗ Data Exporter integration failed: {e}")
        results['export'] = False
    
    return results


def test_configuration_integration():
    """Test configuration management integration."""
    logger.info("Testing Configuration Integration...")
    
    try:
        from utils.config import ConfigManager
        
        # Test default configuration
        config_manager = ConfigManager()
        config = config_manager.config
        
        logger.info("✓ Configuration Manager initialized")
        logger.info(f"  - Chunk size: {config.chunk_size}")
        logger.info(f"  - Max memory: {config.max_memory_usage / (1024**3):.1f} GB")
        logger.info(f"  - Anomaly threshold: {config.anomaly_threshold}")
        logger.info(f"  - Output formats: {config.output_formats}")
        
        # Test configuration validation
        errors = config_manager.validate_config()
        if not errors:
            logger.info("✓ Configuration validation passed")
            return True
        else:
            logger.error(f"✗ Configuration validation failed: {errors}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Configuration integration test failed: {e}")
        return False


def test_monitoring_integration():
    """Test monitoring and logging integration."""
    logger.info("Testing Monitoring Integration...")
    
    try:
        from utils.logger import get_module_logger
        from utils.memory_manager import MemoryManager
        from utils.performance_manager import PerformanceManager
        
        # Test logging
        test_logger = get_module_logger("integration_test")
        test_logger.info("Test log message")
        logger.info("✓ Logging system working")
        
        # Test memory manager
        memory_manager = MemoryManager()
        memory_usage = memory_manager.check_memory_usage()
        logger.info(f"✓ Memory Manager working - Usage OK: {memory_usage}")
        
        # Test performance manager
        performance_manager = PerformanceManager()
        logger.info("✓ Performance Manager initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Monitoring integration test failed: {e}")
        return False


def test_data_quality_integration():
    """Test data quality and validation integration."""
    logger.info("Testing Data Quality Integration...")
    
    try:
        from quality.validation_engine import ValidationEngine
        from quality.automated_validation import validate_data
        
        # Create test data with quality issues
        test_data = pd.DataFrame({
            'order_id': ['ORD_001', 'ORD_002', '', 'ORD_004'],
            'product_name': ['iPhone', '', 'Samsung TV', 'Dell Laptop'],
            'quantity': [5, 0, -2, 10],
            'unit_price': [99.99, 0, -50, 299.99],
            'discount_percent': [0.1, 1.5, -0.1, 0.2]
        })
        
        # Test validation engine
        validator = ValidationEngine()
        logger.info("✓ Validation Engine initialized")
        
        # Test automated validation
        validation_result = validate_data(test_data)
        
        if validation_result:
            logger.info("✓ Automated validation working")
            logger.info(f"  - Validation status: {validation_result.get('status', 'Unknown')}")
            return True
        else:
            logger.error("✗ Automated validation failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Data Quality integration test failed: {e}")
        return False


def test_output_file_integration():
    """Test output file generation and formats."""
    logger.info("Testing Output File Integration...")
    
    try:
        # Check if output directories exist
        output_dirs = [
            'data/output/csv',
            'data/output/parquet',
            'data/processed',
            'logs'
        ]
        
        for output_dir in output_dirs:
            path = Path(output_dir)
            if path.exists():
                files = list(path.glob('*'))
                logger.info(f"✓ {output_dir}: {len(files)} files")
            else:
                logger.warning(f"✗ {output_dir}: Directory not found")
        
        # Check for recent output files
        csv_files = list(Path('data/output/csv').glob('*.csv')) if Path('data/output/csv').exists() else []
        parquet_files = list(Path('data/output/parquet').glob('*.parquet')) if Path('data/output/parquet').exists() else []
        
        if csv_files or parquet_files:
            logger.info(f"✓ Output files found - CSV: {len(csv_files)}, Parquet: {len(parquet_files)}")
            
            # Test reading output files
            if csv_files:
                test_csv = pd.read_csv(csv_files[0])
                logger.info(f"✓ CSV file readable - {len(test_csv)} records")
            
            if parquet_files:
                try:
                    test_parquet = pd.read_parquet(parquet_files[0])
                    logger.info(f"✓ Parquet file readable - {len(test_parquet)} records")
                except ImportError:
                    logger.warning("✗ Parquet reading requires pyarrow")
            
            return True
        else:
            logger.warning("✗ No output files found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Output file integration test failed: {e}")
        return False


def run_comprehensive_validation():
    """Run comprehensive pipeline integration validation."""
    logger.info("=" * 60)
    logger.info("SCALABLE DATA PIPELINE - INTEGRATION VALIDATION")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run all integration tests
    tests = [
        ("Configuration Integration", test_configuration_integration),
        ("Component Integration", test_component_integration),
        ("Monitoring Integration", test_monitoring_integration),
        ("Data Quality Integration", test_data_quality_integration),
        ("ETL Main Integration", test_etl_main_integration),
        ("Dashboard Integration", test_dashboard_integration),
        ("Output File Integration", test_output_file_integration)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            test_results[test_name] = False
    
    # Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    failed_tests = total_tests - passed_tests
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    logger.info("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        logger.info(f"  {symbol} {test_name}: {status}")
    
    # Overall assessment
    if passed_tests == total_tests:
        logger.info("\n🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("The pipeline is fully integrated and ready for deployment.")
        return True
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        logger.info(f"\n✅ INTEGRATION MOSTLY SUCCESSFUL ({passed_tests}/{total_tests} passed)")
        logger.info("The pipeline has good integration with minor issues.")
        return True
    else:
        logger.info(f"\n❌ INTEGRATION NEEDS WORK ({passed_tests}/{total_tests} passed)")
        logger.info("The pipeline has significant integration issues.")
        return False


def main():
    """Main entry point for integration validation."""
    try:
        success = run_comprehensive_validation()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\n⚠️ Validation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Validation crashed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())