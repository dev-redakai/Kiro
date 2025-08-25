#!/usr/bin/env python3
"""Test script for the pipeline monitoring and health check system."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.pipeline_health import pipeline_health_monitor, start_health_monitoring, stop_health_monitoring
from src.utils.pipeline_alerting import pipeline_alerting_system, start_alerting, stop_alerting
from src.utils.pipeline_recovery import pipeline_recovery_manager, start_recovery_monitoring, stop_recovery_monitoring
from src.utils.monitoring import pipeline_monitor, start_monitoring, stop_monitoring
from src.utils.monitoring_dashboard import get_dashboard_data, get_health_status
from src.quality.automated_validation import automated_validator, validate_data
from src.utils.logger import get_module_logger
import pandas as pd


def test_monitoring_system():
    """Test the monitoring system components."""
    logger = get_module_logger("test_monitoring")
    
    print("Testing Pipeline Monitoring and Health Check System")
    print("=" * 60)
    
    try:
        # Test 1: Start monitoring systems
        print("\n1. Starting monitoring systems...")
        start_monitoring()
        start_health_monitoring()
        start_alerting()
        start_recovery_monitoring()
        print("✓ All monitoring systems started")
        
        # Wait a moment for systems to initialize
        time.sleep(2)
        
        # Test 2: Check health status
        print("\n2. Checking health status...")
        health_status = get_health_status()
        print(f"✓ Health status: {health_status.get('status', 'unknown')}")
        print(f"  - Components checked: {len(health_status.get('checks', {}))}")
        print(f"  - Active alerts: {health_status.get('alerts', 0)}")
        
        # Test 3: Get dashboard data
        print("\n3. Getting dashboard data...")
        dashboard_data = get_dashboard_data(force_refresh=True)
        print("✓ Dashboard data retrieved")
        print(f"  - System status: {dashboard_data.get('system_status', {}).get('overall_status', 'unknown')}")
        print(f"  - Timestamp: {dashboard_data.get('timestamp', 'unknown')}")
        
        # Test 4: Test automated validation
        print("\n4. Testing automated data validation...")
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD003', 'ORD001'],  # Duplicate
            'product_name': ['Product A', 'Product B', '', 'Product C'],  # Empty value
            'category': ['Electronics', 'Fashion', 'InvalidCategory', 'Electronics'],  # Invalid category
            'quantity': [1, 2, -1, 5],  # Negative quantity
            'unit_price': [10.0, 20.0, 0.0, 15.0],  # Zero price
            'discount_percent': [0.1, 0.2, 1.5, 0.0],  # Invalid discount > 1.0
            'region': ['North', 'South', 'InvalidRegion', 'East'],  # Invalid region
            'sale_date': ['2024-01-01', '2024-02-01', '2024-13-01', '2024-03-01'],  # Invalid date
            'customer_email': ['user@example.com', 'invalid-email', None, 'user2@example.com'],
            'revenue': [9.0, 16.0, 0.0, 15.0]
        })
        
        validation_result = validate_data(sample_data)
        print("✓ Data validation completed")
        print(f"  - Validation status: {validation_result.get('status', 'unknown')}")
        print(f"  - Rules run: {validation_result.get('total_rules_run', 0)}")
        print(f"  - Critical failures: {validation_result.get('critical_failures', 0)}")
        print(f"  - Error failures: {validation_result.get('error_failures', 0)}")
        print(f"  - Warning failures: {validation_result.get('warning_failures', 0)}")
        
        # Test 5: Test recovery system
        print("\n5. Testing recovery system...")
        recovery_summary = pipeline_recovery_manager.get_recovery_summary()
        print("✓ Recovery system status retrieved")
        print(f"  - Total checkpoints: {recovery_summary.get('total_checkpoints', 0)}")
        print(f"  - Monitoring active: {recovery_summary.get('monitoring_active', False)}")
        print(f"  - Recovery success rate: {recovery_summary.get('recovery_success_rate', 0):.1f}%")
        
        # Test 6: Test alerting system
        print("\n6. Testing alerting system...")
        alert_summary = pipeline_alerting_system.get_alert_summary(hours=1)
        print("✓ Alerting system status retrieved")
        print(f"  - Total alerts (1h): {alert_summary.get('total_alerts', 0)}")
        print(f"  - Active alerts: {alert_summary.get('active_alerts', 0)}")
        print(f"  - Resolution rate: {alert_summary.get('resolution_rate', 0):.1f}%")
        
        # Test 7: Export monitoring report
        print("\n7. Exporting monitoring report...")
        report_path = "logs/test_monitoring_report.json"
        Path("logs").mkdir(exist_ok=True)
        
        from src.utils.monitoring_dashboard import export_monitoring_report
        export_monitoring_report(report_path)
        
        if Path(report_path).exists():
            print(f"✓ Monitoring report exported to {report_path}")
            file_size = Path(report_path).stat().st_size
            print(f"  - Report size: {file_size} bytes")
        else:
            print("✗ Failed to export monitoring report")
        
        print("\n" + "=" * 60)
        print("✓ All monitoring system tests completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"\n✗ Test failed: {e}")
        return False
    
    finally:
        # Clean up: stop monitoring systems
        print("\nStopping monitoring systems...")
        try:
            stop_monitoring()
            stop_health_monitoring()
            stop_alerting()
            stop_recovery_monitoring()
            print("✓ All monitoring systems stopped")
        except Exception as e:
            print(f"Warning: Error stopping monitoring systems: {e}")


def main():
    """Main test function."""
    success = test_monitoring_system()
    
    if success:
        print("\n🎉 Pipeline monitoring and health check system is working correctly!")
        return 0
    else:
        print("\n❌ Pipeline monitoring and health check system test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())