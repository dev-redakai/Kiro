#!/usr/bin/env python3
"""Test enhanced logging and error reporting for validation system."""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.quality.automated_validation import AutomatedDataValidator
from src.utils.logger import get_module_logger

def create_test_data_with_issues():
    """Create test data with various validation issues for testing enhanced logging."""
    np.random.seed(42)
    
    # Create 1000 records with intentional issues for comprehensive logging testing
    num_records = 1000
    
    data = {
        'order_id': [f'ORD{i:06d}' for i in range(1, num_records-2)] + [None, None, 'ORD000001'],  # Duplicates and nulls
        'product_name': (['Product A'] * 500 + ['Product B'] * 400 + [None] * 50 + [''] * 30 + ['X' * 250] * 20)[:num_records],  # Various issues
        'category': (['Electronics'] * 400 + ['Fashion'] * 300 + ['InvalidCategory'] * 200 + [None] * 100)[:num_records],  # Invalid categories
        'quantity': (list(range(1, 501)) + list(range(1, 401)) + [-5, -10, 0] * 34 + [None] * 35)[:num_records],  # Negative and null values
        'unit_price': ([round(np.random.uniform(10, 1000), 2) for _ in range(800)] + [-50.0] * 100 + [None] * 100)[:num_records],  # Negative prices
        'discount_percent': ([round(np.random.uniform(0, 0.5), 3) for _ in range(700)] + [1.5, 2.0] * 100 + [None] * 100)[:num_records],  # Invalid discounts
        'region': (['North'] * 300 + ['South'] * 300 + ['InvalidRegion'] * 300 + [None] * 100)[:num_records],  # Invalid regions
        'customer_email': ([f'user{i}@example.com' for i in range(500)] + ['invalid-email'] * 200 + [None] * 300)[:num_records],  # Invalid emails
        'sale_date': ([datetime.now() - timedelta(days=i) for i in range(800)] + ['2050-01-01'] * 100 + [None] * 100)[:num_records],  # Future dates
        'revenue': [0] * num_records  # Will be calculated, but intentionally wrong for business logic validation
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate revenue correctly for some records, incorrectly for others
    for i in range(len(df)):
        if pd.notna(df.loc[i, 'quantity']) and pd.notna(df.loc[i, 'unit_price']) and pd.notna(df.loc[i, 'discount_percent']):
            if i < 500:  # First 500 records have correct revenue
                df.loc[i, 'revenue'] = df.loc[i, 'quantity'] * df.loc[i, 'unit_price'] * (1 - df.loc[i, 'discount_percent'])
            else:  # Rest have incorrect revenue for business logic validation
                df.loc[i, 'revenue'] = df.loc[i, 'quantity'] * df.loc[i, 'unit_price'] * 0.5  # Intentionally wrong
    
    return df

def test_enhanced_validation_logging():
    """Test the enhanced logging and error reporting functionality."""
    logger = get_module_logger("test_validation_logging")
    logger.info("Starting enhanced validation logging test")
    
    try:
        # Create test data with various issues
        logger.info("Creating test data with validation issues")
        df = create_test_data_with_issues()
        logger.info(f"Created test dataset with {len(df)} records and {len(df.columns)} columns")
        
        # Initialize validator
        logger.info("Initializing automated data validator")
        validator = AutomatedDataValidator()
        
        # Run validation with enhanced logging
        logger.info("Running validation with enhanced logging and error reporting")
        validation_results = validator.validate_dataframe(df)
        
        # Test structured error report creation
        logger.info("Creating structured error report")
        error_report = validator.create_structured_error_report("logs/validation_error_report.json")
        
        # Test debugging report export
        logger.info("Exporting debugging report")
        debug_report_path = validator.export_debugging_report("logs")
        
        # Log test results
        logger.info("Enhanced validation logging test completed successfully")
        logger.info(f"Validation status: {validation_results.get('status', 'unknown')}")
        logger.info(f"Error report created with {len(error_report.get('actionable_recommendations', []))} recommendations")
        logger.info(f"Debug report exported to: {debug_report_path}")
        
        # Display key metrics from enhanced logging
        if validator.validation_metrics:
            metrics = validator.validation_metrics
            logger.info(
                f"Enhanced Metrics Summary:\n"
                f"  Data Quality Score: {metrics.overall_data_quality_score:.2f}%\n"
                f"  Total Rules Executed: {metrics.total_rules_executed}\n"
                f"  Failed Rules: {metrics.failed_rules}\n"
                f"  Memory Usage: {metrics.memory_usage_mb:.2f}MB\n"
                f"  Execution Time: {metrics.total_execution_time:.2f}s"
            )
        
        # Display error reports summary
        if validator.error_reports:
            logger.info(f"Generated {len(validator.error_reports)} detailed error reports")
            for i, report in enumerate(validator.error_reports[:3]):  # Show first 3
                logger.info(
                    f"  Error Report {i+1}: {report.rule_name} - {report.error_type}\n"
                    f"    Field: {report.field_name}\n"
                    f"    Total Errors: {report.total_errors}\n"
                    f"    Error Percentage: {report.error_percentage:.2f}%\n"
                    f"    Suggestions: {len(report.suggested_actions)}"
                )
        
        # Test specific logging features
        logger.info("Testing specific enhanced logging features")
        
        # Test validation context logging
        logger.log_validation_context(
            "test_rule", "test_field", "object", 
            ["sample1", "sample2"], 0.123,
            ["problematic1", "problematic2"]
        )
        
        # Test validation error logging with context
        logger.log_validation_error_with_context(
            "test_rule", "test_field", "TestError", "Test error message",
            ["bad_value1", "bad_value2"], 
            ["Fix data source", "Add validation"], 0.456
        )
        
        # Test metrics summary logging
        logger.log_validation_metrics_summary(
            10, 8, 2, 1000, 950, 50, 2.5, 128.5, 95.0
        )
        
        # Test problematic values analysis logging
        logger.log_problematic_values_analysis(
            "test_field", 50, 
            {"missing_values": 30, "invalid_format": 20},
            ["bad1", "bad2", "bad3"],
            ["Fix missing values", "Validate format"]
        )
        
        # Test actionable insights logging
        logger.log_actionable_insights([
            "Performance issue detected in rule X",
            "Data quality below threshold in field Y",
            "Memory usage is high, consider optimization"
        ])
        
        # Test structured error report logging
        logger.log_structured_error_report(
            "logs/test_report.json", 25, "problematic_field", 5
        )
        
        logger.info("All enhanced logging features tested successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced validation logging test failed: {e}")
        logger.exception("Full error details:")
        return False

def main():
    """Main test function."""
    print("Testing Enhanced Validation Logging and Error Reporting")
    print("=" * 60)
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the test
    success = test_enhanced_validation_logging()
    
    if success:
        print("\n✅ Enhanced validation logging test completed successfully!")
        print("\nCheck the following files for detailed output:")
        print("  - logs/pipeline.log (main log file)")
        print("  - logs/validation_error_report.json (structured error report)")
        print("  - logs/validation_debug_report_*.json (debugging report)")
    else:
        print("\n❌ Enhanced validation logging test failed!")
        print("Check logs/pipeline.log for error details")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)