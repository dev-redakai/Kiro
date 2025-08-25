# Comprehensive Error Handling and Monitoring System

This directory contains a comprehensive error handling and monitoring system for the data pipeline, implementing robust error recovery strategies and detailed monitoring capabilities.

## Components

### 1. Error Handler (`error_handler.py`)

The error handler provides comprehensive error management with automatic recovery strategies:

#### Error Categories
- **Data Quality**: Invalid data, missing values, format issues
- **Memory**: Out-of-memory conditions, memory leaks
- **Processing**: Transformation failures, calculation errors
- **Dashboard**: Visualization errors, data loading failures
- **I/O**: File access issues, network problems
- **Validation**: Schema validation, constraint violations

#### Error Severity Levels
- **Low**: Minor issues that can be easily corrected
- **Medium**: Issues requiring intervention but not critical
- **High**: Serious issues that may affect processing
- **Critical**: System-threatening errors that require immediate attention

#### Recovery Strategies

**Data Quality Recovery:**
- Apply data corrections when available
- Use default values for missing/invalid data
- Skip invalid records with logging

**Memory Recovery:**
- Reduce chunk size automatically
- Force garbage collection
- Clear caches
- Use memory mapping for large files

**Processing Recovery:**
- Retry operations with exponential backoff
- Skip problematic data chunks
- Use fallback processing methods

**Dashboard Recovery:**
- Use cached data when fresh data unavailable
- Show error messages to users
- Use simpler fallback visualizations

### 2. Monitoring System (`monitoring.py`)

Comprehensive monitoring with real-time metrics collection:

#### System Monitor
- CPU and memory usage tracking
- Disk usage monitoring
- Background monitoring thread
- Resource usage alerts

#### Pipeline Monitor
- Operation timing and throughput
- Data quality metrics
- Performance metrics by category
- Error tracking and reporting

#### Features
- Real-time metrics collection
- Performance benchmarking
- Data quality tracking
- Error rate monitoring
- Resource usage optimization

### 3. Advanced Logging (`logging_config.py`)

Enhanced logging system with structured output:

#### Log Types
- **Console**: Simple format for development
- **Main Log**: Detailed format with context
- **Error Log**: Errors and critical issues only
- **Data Quality Log**: Data quality specific issues
- **Processing Log**: Processing stage information
- **JSON Log**: Structured logging for analysis

#### Features
- Automatic log rotation
- Performance context injection
- Categorized logging filters
- Metrics integration
- JSON structured output

### 4. Monitoring Dashboard (`monitoring_dashboard.py`)

Interactive Streamlit dashboard for real-time monitoring:

#### Dashboard Tabs
- **Overview**: Key metrics and pipeline status
- **Performance**: Throughput and processing metrics
- **Data Quality**: Error rates and validation results
- **Errors**: Error analysis and resolution status
- **System Health**: Resource usage and system metrics

#### Features
- Real-time updates
- Interactive visualizations
- Historical data analysis
- Alert notifications
- Export capabilities

## Usage Examples

### Basic Error Handling

```python
from src.utils.error_handler import handle_data_quality_error, DataQualityError

try:
    # Some data processing
    if invalid_data:
        raise DataQualityError(
            "Invalid quantity value",
            field="quantity",
            value="-5",
            correction=0
        )
except DataQualityError as e:
    corrected_value = handle_data_quality_error(e)
    # Use corrected_value or handle appropriately
```

### Monitoring Operations

```python
from src.utils.monitoring import time_operation, log_performance_metric

# Time an operation
with time_operation("data_cleaning", records_count=10000) as timer:
    # Perform data cleaning
    cleaned_data = clean_data(raw_data)
    timer.update_records_count(len(cleaned_data))

# Log performance metrics
log_performance_metric("processing_speed", 1250.5, "records/sec", "performance")
```

### Starting Monitoring

```python
from src.utils.monitoring import start_monitoring, stop_monitoring

# Start monitoring
start_monitoring()

try:
    # Run your pipeline
    run_pipeline()
finally:
    # Stop monitoring
    stop_monitoring()
```

### Running Monitoring Dashboard

```python
from src.utils.monitoring_dashboard import run_monitoring_dashboard

# Run the dashboard (Streamlit app)
run_monitoring_dashboard()
```

Or from command line:
```bash
streamlit run src/utils/monitoring_dashboard.py
```

## Configuration

The system uses the main pipeline configuration (`config.py`) for settings:

```python
# Error handling settings
anomaly_threshold: float = 3.0
anomaly_sensitivity: float = 0.95

# Logging settings
log_level: str = 'INFO'
log_file: str = 'pipeline.log'
log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Memory settings
chunk_size: int = 50000
max_memory_usage: int = 4 * 1024 * 1024 * 1024  # 4GB
```

## Integration with Pipeline Components

The error handling and monitoring system integrates seamlessly with all pipeline components:

1. **Data Ingestion**: Memory error handling, progress tracking
2. **Data Cleaning**: Data quality error recovery, validation metrics
3. **Transformation**: Processing error handling, performance monitoring
4. **Dashboard**: Error fallbacks, user experience optimization
5. **Export**: I/O error handling, completion tracking

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_error_handler.py -v
```

Run the demonstration:

```bash
python examples/error_handling_monitoring_demo.py
```

## Log Files

The system creates several log files in the `logs/` directory:

- `pipeline.log`: Main application log
- `errors.log`: Errors and critical issues
- `data_quality.log`: Data quality specific issues
- `processing.log`: Processing stage information
- `pipeline.json`: Structured JSON logs

## Reports

Monitoring reports are exported to the `reports/` directory:

- `monitoring_report_YYYYMMDD_HHMMSS.json`: Comprehensive monitoring data
- `error_log_YYYYMMDD_HHMMSS.json`: Detailed error history

## Best Practices

1. **Always use context managers** for timing operations
2. **Provide meaningful error messages** with context
3. **Use appropriate error severity levels**
4. **Monitor resource usage** during processing
5. **Export reports regularly** for analysis
6. **Configure appropriate log levels** for different environments
7. **Use the dashboard** for real-time monitoring during development

This system provides enterprise-grade error handling and monitoring capabilities, ensuring robust and observable data pipeline operations.