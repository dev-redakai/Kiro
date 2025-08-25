# Pipeline Monitoring and Health Check System Implementation

## Overview

This document describes the comprehensive pipeline monitoring and health check system implemented for task 12.2. The system provides real-time monitoring, automated health checks, data quality validation, performance alerting, and recovery mechanisms for failed pipeline runs.

## Components Implemented

### 1. Pipeline Health Monitor (`src/utils/pipeline_health.py`)

**Purpose**: Provides comprehensive health monitoring for all pipeline components.

**Key Features**:
- Real-time health checks for system resources (memory, CPU, disk)
- Component-specific health monitoring (ingestion, cleaning, transformation, etc.)
- Configurable health check intervals and thresholds
- Automatic recovery action triggering for failed health checks
- Health status history tracking

**Health Checks Implemented**:
- System memory usage monitoring (warning at 80%, critical at 90%)
- CPU usage monitoring (warning at 80%, critical at 90%)
- Disk space monitoring (warning at 85%, critical at 95%)
- Data quality overall health assessment
- Pipeline performance monitoring
- Data directory accessibility checks

### 2. Automated Data Quality Validation (`src/quality/automated_validation.py`)

**Purpose**: Provides automated validation of data quality with configurable rules.

**Key Features**:
- 13 pre-configured validation rules for e-commerce data
- Support for multiple validation types (not_null, unique, range, format, enum, statistical, business_logic)
- Configurable error rate thresholds
- Detailed validation reporting with sample invalid values
- Integration with monitoring system for quality metrics

**Validation Rules Implemented**:
- Order ID validation (not null, unique)
- Product name validation (not null, length constraints)
- Category enumeration validation
- Quantity range and statistical validation
- Unit price validation
- Discount percentage range validation
- Region enumeration validation
- Email format validation
- Date range validation
- Revenue calculation business logic validation

### 3. Pipeline Recovery System (`src/utils/pipeline_recovery.py`)

**Purpose**: Provides automated recovery mechanisms for pipeline failures.

**Key Features**:
- Checkpoint creation and restoration system
- Multiple recovery strategies (retry, checkpoint restore, partial recovery, fallback methods)
- Failure type classification and appropriate recovery action selection
- Recovery attempt tracking and success rate monitoring
- Automatic cleanup of old checkpoints

**Recovery Actions Implemented**:
- Memory error recovery (garbage collection, chunk size reduction)
- I/O error recovery with exponential backoff
- Data corruption recovery from checkpoints
- Processing error recovery (skip problematic data)
- Timeout error recovery (increase timeouts)
- System error recovery (component restart)

### 4. Pipeline Alerting System (`src/utils/pipeline_alerting.py`)

**Purpose**: Provides real-time alerting for pipeline performance and health issues.

**Key Features**:
- Multiple alert channels (log, file, email, webhook, dashboard)
- Configurable alert rules with cooldown periods
- Alert severity levels (info, warning, error, critical)
- Alert acknowledgment and resolution tracking
- Alert history and trend analysis

**Alert Rules Implemented**:
- High memory usage alerts
- Critical memory usage alerts
- High CPU usage alerts
- Low processing throughput alerts
- High error rate alerts
- Data quality degradation alerts
- Pipeline failure alerts
- Low disk space alerts

### 5. Monitoring Dashboard Integration (`src/utils/monitoring_dashboard.py`)

**Purpose**: Provides comprehensive data for monitoring dashboards and APIs.

**Key Features**:
- Real-time dashboard data aggregation
- System status overview with color-coded indicators
- Performance metrics and trends
- Resource usage monitoring
- Active alerts summary
- Recent activities tracking
- Health check API endpoint

### 6. Monitoring CLI (`src/utils/monitoring_cli.py`)

**Purpose**: Provides command-line interface for monitoring operations.

**Available Commands**:
- `status` - Show pipeline status overview
- `health` - Show health status details
- `start-monitoring` - Start monitoring systems
- `stop-monitoring` - Stop monitoring systems
- `export-report` - Export comprehensive monitoring report
- `dashboard` - Get dashboard data
- `alerts` - Show active alerts

## Integration with ETL Pipeline

The monitoring system is fully integrated with the main ETL pipeline (`src/pipeline/etl_main.py`):

### Automatic Monitoring Startup
- Health monitoring starts automatically when pipeline runs
- Alerting system activates for real-time issue detection
- Recovery monitoring enables automatic failure handling

### Checkpoint Integration
- Checkpoints created before and after major processing stages
- Intermediate checkpoints during long-running operations
- Automatic checkpoint cleanup to manage disk space

### Data Quality Integration
- Automated validation runs after data cleaning stage
- Quality metrics logged to monitoring system
- Quality issues trigger appropriate alerts

### Recovery Integration
- Automatic failure classification and recovery attempt
- Recovery success tracking and reporting
- Graceful degradation when recovery fails

## Configuration

The monitoring system uses the existing pipeline configuration (`config/pipeline_config.yaml`) with additional monitoring-specific settings:

```yaml
# Monitoring thresholds
memory_warning_threshold: 0.8
memory_critical_threshold: 0.9
cpu_warning_threshold: 80.0
cpu_critical_threshold: 90.0
disk_warning_threshold: 85.0
disk_critical_threshold: 95.0

# Data quality thresholds
data_quality_warning_threshold: 5.0
data_quality_critical_threshold: 10.0

# Alert settings
alert_cooldown_minutes: 15
email_alerts_enabled: false
```

## Usage Examples

### Starting Monitoring Systems
```bash
# Start all monitoring components
python -m src.utils.monitoring_cli start-monitoring

# Start specific components only
python -m src.utils.monitoring_cli start-monitoring --components health alerting
```

### Checking System Status
```bash
# Show status in table format
python -m src.utils.monitoring_cli status

# Show status in JSON format
python -m src.utils.monitoring_cli status --format json
```

### Health Monitoring
```bash
# Show health status
python -m src.utils.monitoring_cli health

# Export comprehensive monitoring report
python -m src.utils.monitoring_cli export-report --output monitoring_report.json
```

### Programmatic Usage
```python
from src.utils.pipeline_health import start_health_monitoring, get_pipeline_health
from src.utils.monitoring_dashboard import get_dashboard_data
from src.quality.automated_validation import validate_data

# Start monitoring
start_health_monitoring()

# Get current health status
health_status = get_pipeline_health()
print(f"Overall status: {health_status.overall_status.value}")

# Get dashboard data
dashboard_data = get_dashboard_data()

# Validate data quality
validation_result = validate_data(dataframe)
```

## Testing

A comprehensive test script (`test_monitoring_system.py`) verifies all monitoring components:

```bash
python test_monitoring_system.py
```

The test covers:
- Monitoring system startup/shutdown
- Health status checking
- Dashboard data retrieval
- Automated data validation
- Recovery system functionality
- Alerting system operation
- Report generation

## Files Created/Modified

### New Files Created:
- `src/utils/pipeline_health.py` - Health monitoring system
- `src/quality/automated_validation.py` - Data quality validation
- `src/utils/pipeline_recovery.py` - Recovery mechanisms
- `src/utils/pipeline_alerting.py` - Alerting system
- `src/utils/monitoring_dashboard.py` - Dashboard integration
- `src/utils/monitoring_cli.py` - CLI interface
- `test_monitoring_system.py` - Comprehensive test script
- `docs/MONITORING_SYSTEM_IMPLEMENTATION.md` - This documentation

### Modified Files:
- `src/pipeline/etl_main.py` - Integrated monitoring systems

## Performance Impact

The monitoring system is designed to have minimal performance impact:
- Health checks run in background threads
- Configurable check intervals (default: 30-60 seconds)
- Efficient data structures for metrics storage
- Automatic cleanup of old data to prevent memory leaks
- Optional components can be disabled if not needed

## Future Enhancements

Potential improvements for the monitoring system:
- Web-based dashboard UI
- Integration with external monitoring systems (Prometheus, Grafana)
- Machine learning-based anomaly detection
- Advanced alerting channels (Slack, PagerDuty)
- Distributed monitoring for multi-node deployments
- Custom metric collection and visualization

## Conclusion

The implemented monitoring and health check system provides comprehensive visibility into pipeline operations, enabling proactive issue detection, automated recovery, and detailed performance analysis. The system is production-ready and can be easily extended with additional monitoring capabilities as needed.