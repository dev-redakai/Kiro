# Scalable Data Pipeline - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying and operating the Scalable Data Engineering Pipeline for processing large e-commerce datasets.

## System Requirements

### Hardware Requirements
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **RAM**: Minimum 8GB, Recommended 16GB+ for large datasets
- **Storage**: Minimum 50GB free space, SSD recommended
- **Network**: Stable internet connection for package installation

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Git**: For version control and deployment

## Installation Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd scalable-data-pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Directory Structure Verification

Ensure the following directory structure exists:

```
scalable-data-pipeline/
├── src/                    # Source code
│   ├── pipeline/          # ETL pipeline modules
│   ├── data/              # Data processing utilities
│   ├── quality/           # Data quality and validation
│   ├── dashboard/         # Dashboard application
│   └── utils/             # Common utilities
├── data/                  # Data storage
│   ├── raw/               # Raw input data
│   ├── processed/         # Cleaned data
│   ├── output/            # Final outputs
│   └── temp/              # Temporary files
├── config/                # Configuration files
├── tests/                 # Test suite
├── docs/                  # Documentation
├── logs/                  # Log files
└── reports/               # Generated reports
```

### 3. Configuration Setup

#### Pipeline Configuration

Edit `config/pipeline_config.yaml`:

```yaml
# Processing Configuration
chunk_size: 50000                    # Records per chunk
max_memory_usage: 4294967296         # 4GB in bytes
anomaly_threshold: 3.0               # Z-score threshold

# Output Configuration
output_formats:
  - csv
  - parquet

# Data Paths
input_data_path: "data/raw"
processed_data_path: "data/processed"
output_data_path: "data/output"
temp_data_path: "data/temp"

# Logging
log_level: INFO
log_file: "logs/pipeline.log"

# Performance Settings
enable_monitoring: true
enable_recovery: true
enable_alerting: true
```

#### Environment Variables (Optional)

Create `.env` file for sensitive configurations:

```bash
# Database connections (if applicable)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ecommerce_data

# API Keys (if applicable)
MONITORING_API_KEY=your_key_here

# Performance Settings
MAX_WORKERS=8
MEMORY_LIMIT_GB=4
```

## Deployment Options

### Option 1: Local Development Deployment

For development and testing purposes:

```bash
# Generate test data
python -c "
import sys; sys.path.insert(0, 'src')
from utils.test_data_generator import TestDataGenerator
generator = TestDataGenerator()
data = generator.generate_clean_sample(size=10000)
data.to_csv('data/raw/sample_data.csv', index=False)
"

# Run the pipeline
python -m src.pipeline.etl_main --input data/raw/sample_data.csv

# Start the dashboard (requires streamlit)
pip install streamlit
streamlit run src/dashboard/dashboard_app.py
```

### Option 2: Production Deployment

For production environments:

#### Using Docker (Recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/output data/temp logs reports

# Set environment variables
ENV PYTHONPATH=/app/src

# Expose dashboard port
EXPOSE 8501

# Default command
CMD ["python", "-m", "src.pipeline.etl_main", "--help"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app/src
    command: python -m src.pipeline.etl_main --input data/raw/input.csv

  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app/src
    command: streamlit run src/dashboard/dashboard_app.py --server.port=8501 --server.address=0.0.0.0
    depends_on:
      - pipeline
```

Deploy with Docker:

```bash
# Build and run
docker-compose up --build

# Run pipeline only
docker-compose run pipeline python -m src.pipeline.etl_main --input data/raw/your_data.csv

# Access dashboard at http://localhost:8501
```

#### Using Systemd (Linux)

Create service file `/etc/systemd/system/data-pipeline.service`:

```ini
[Unit]
Description=Scalable Data Pipeline
After=network.target

[Service]
Type=simple
User=pipeline
WorkingDirectory=/opt/scalable-data-pipeline
Environment=PYTHONPATH=/opt/scalable-data-pipeline/src
ExecStart=/opt/scalable-data-pipeline/venv/bin/python -m src.pipeline.etl_main --input data/raw/input.csv
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable data-pipeline
sudo systemctl start data-pipeline
sudo systemctl status data-pipeline
```

## Operation Guide

### Running the Pipeline

#### Basic Usage

```bash
# Process a CSV file
python -m src.pipeline.etl_main --input data/raw/sales_data.csv

# With custom configuration
python -m src.pipeline.etl_main --input data/raw/sales_data.csv --config config/custom_config.yaml

# Skip certain stages
python -m src.pipeline.etl_main --input data/raw/sales_data.csv --skip-stages cleaning,export

# Dry run (validate without processing)
python -m src.pipeline.etl_main --input data/raw/sales_data.csv --dry-run
```

#### Advanced Usage

```bash
# Set log level
python -m src.pipeline.etl_main --input data/raw/sales_data.csv --log-level DEBUG

# Check pipeline status
python -m src.pipeline.etl_main --status-only

# Process with monitoring
python -m src.pipeline.etl_main --input data/raw/sales_data.csv --enable-monitoring
```

### Dashboard Operation

#### Starting the Dashboard

```bash
# Install Streamlit if not already installed
pip install streamlit

# Start dashboard
streamlit run src/dashboard/dashboard_app.py

# Start on specific port
streamlit run src/dashboard/dashboard_app.py --server.port 8502

# Start for external access
streamlit run src/dashboard/dashboard_app.py --server.address 0.0.0.0
```

#### Dashboard Features

- **Revenue Trends**: Monthly revenue and order volume analysis
- **Top Products**: Best performing products by revenue and units
- **Regional Analysis**: Sales performance by geographic region
- **Category Insights**: Discount effectiveness by product category
- **Anomaly Detection**: Suspicious transactions and patterns
- **Interactive Filters**: Date range, region, and category filtering

### Monitoring and Maintenance

#### Log Management

```bash
# View recent logs
tail -f logs/pipeline.log

# View error logs
grep ERROR logs/pipeline.log

# Rotate logs (Linux)
logrotate -f config/logrotate.conf
```

#### Performance Monitoring

```bash
# Monitor system resources
htop

# Monitor disk usage
df -h

# Monitor memory usage
free -h

# Monitor pipeline performance
python -c "
import sys; sys.path.insert(0, 'src')
from utils.monitoring import get_system_metrics
print(get_system_metrics())
"
```

#### Health Checks

```bash
# Check pipeline health
python -c "
import sys; sys.path.insert(0, 'src')
from utils.pipeline_health import pipeline_health_monitor
health = pipeline_health_monitor.get_health_status()
print(f'Overall Health: {health[\"overall_status\"]}')
"

# Validate configuration
python -m src.utils.config --validate

# Test data quality
python -m src.quality.automated_validation --test
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors

**Symptoms**: OutOfMemoryError, slow processing
**Solutions**:
- Reduce `chunk_size` in configuration
- Increase system RAM
- Enable memory optimization: `enable_memory_optimization: true`

#### 2. Import Errors

**Symptoms**: ModuleNotFoundError, ImportError
**Solutions**:
- Ensure virtual environment is activated
- Install missing dependencies: `pip install -r requirements.txt`
- Set PYTHONPATH: `export PYTHONPATH=/path/to/project/src`

#### 3. Data Quality Issues

**Symptoms**: Validation failures, cleaning errors
**Solutions**:
- Check input data format
- Review data quality rules in `src/quality/validation_engine.py`
- Enable data profiling: `--enable-profiling`

#### 4. Dashboard Not Loading

**Symptoms**: Streamlit errors, blank dashboard
**Solutions**:
- Install Streamlit: `pip install streamlit`
- Check data availability in `data/output/`
- Verify port availability: `netstat -an | grep 8501`

### Performance Optimization

#### For Large Datasets (100M+ records)

```yaml
# Optimized configuration
chunk_size: 100000
max_memory_usage: 8589934592  # 8GB
enable_parallel_processing: true
optimize_memory_usage: true
use_memory_mapping: true
```

#### For Limited Resources

```yaml
# Resource-constrained configuration
chunk_size: 10000
max_memory_usage: 2147483648  # 2GB
enable_parallel_processing: false
optimize_memory_usage: true
```

## Security Considerations

### Data Protection

1. **Sensitive Data**: Ensure PII is properly handled
2. **Access Control**: Implement proper file permissions
3. **Encryption**: Use encrypted storage for sensitive data
4. **Audit Logging**: Enable comprehensive logging

### Network Security

1. **Dashboard Access**: Use authentication for production
2. **API Security**: Implement rate limiting and authentication
3. **Firewall**: Configure appropriate network access

## Backup and Recovery

### Data Backup

```bash
# Backup processed data
tar -czf backup_$(date +%Y%m%d).tar.gz data/processed/ data/output/

# Backup configuration
cp -r config/ backup/config_$(date +%Y%m%d)/
```

### Recovery Procedures

```bash
# Restore from backup
tar -xzf backup_20231201.tar.gz

# Restart pipeline
python -m src.pipeline.etl_main --input data/raw/recovery_data.csv --enable-recovery
```

## Scaling Considerations

### Horizontal Scaling

For very large datasets, consider:

1. **Data Partitioning**: Split input files by date/region
2. **Parallel Processing**: Run multiple pipeline instances
3. **Distributed Storage**: Use distributed file systems
4. **Load Balancing**: Distribute dashboard load

### Vertical Scaling

For single-machine scaling:

1. **Increase RAM**: More memory for larger chunks
2. **SSD Storage**: Faster I/O operations
3. **CPU Cores**: Better parallel processing
4. **Network**: Faster data transfer

## Support and Maintenance

### Regular Maintenance Tasks

1. **Log Rotation**: Weekly log cleanup
2. **Data Archival**: Monthly data archival
3. **Performance Review**: Monthly performance analysis
4. **Security Updates**: Regular dependency updates

### Getting Help

1. **Documentation**: Check `docs/` directory
2. **Logs**: Review `logs/pipeline.log`
3. **Health Checks**: Run system diagnostics
4. **Community**: Check project repository for issues

## Appendix

### A. Configuration Reference

See `config/pipeline_config.yaml` for complete configuration options.

### B. API Reference

See `docs/API_DOCUMENTATION.md` for detailed API documentation.

### C. Performance Benchmarks

| Dataset Size | Processing Time | Memory Usage | Throughput |
|-------------|----------------|--------------|------------|
| 10K records | 30 seconds     | 500MB        | 333 rec/sec |
| 100K records| 5 minutes      | 2GB          | 333 rec/sec |
| 1M records  | 45 minutes     | 4GB          | 370 rec/sec |
| 10M records | 7 hours        | 8GB          | 400 rec/sec |

### D. Troubleshooting Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Configuration valid
- [ ] Input data accessible
- [ ] Output directories writable
- [ ] Sufficient disk space
- [ ] Adequate memory
- [ ] Network connectivity (if required)

---

For additional support, please refer to the project documentation or contact the development team.