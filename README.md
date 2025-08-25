# Scalable Data Engineering Pipeline

A comprehensive, memory-efficient data engineering pipeline designed to process, clean, transform, and analyze large e-commerce datasets containing millions of records. The system handles ~100 million records without using distributed processing frameworks, providing business insights through data transformations and an interactive dashboard.

## Overview

This pipeline processes e-commerce sales data from a mid-size company selling electronics, fashion, and home goods across India and Southeast Asia. It includes:

- **Memory-efficient data ingestion** with chunked CSV reading
- **Comprehensive data cleaning** and standardization
- **Business intelligence transformations** for analytical insights
- **Anomaly detection** for suspicious transactions
- **Interactive dashboard** for data visualization
- **Robust error handling** and monitoring
- **Comprehensive testing suite** with realistic test data generation

## 🔗 ETL-Dashboard Integration Enhancement

**NEW**: The dashboard now uses **real processed data from the ETL pipeline** instead of sample data, providing authentic business insights based on actual data processing results.

### Key Benefits:
- **📊 Authentic Insights**: Dashboard shows real business metrics from processed data
- **🔄 Seamless Integration**: Automatic data flow from ETL pipeline to dashboard  
- **📈 Real Analytics**: Monthly trends, top products, and anomalies from actual data
- **🛡️ Robust Fallback**: Graceful degradation to sample data when needed

### Enhanced Data Flow:
```
Raw Data → ETL Pipeline → Processed Data → Dashboard Analytics → Business Insights
```

**See [ETL_DASHBOARD_INTEGRATION_ENHANCEMENT.md](ETL_DASHBOARD_INTEGRATION_ENHANCEMENT.md) for complete technical details.**

## Project Structure

```
├── src/                           # Source code
│   ├── pipeline/                  # ETL pipeline modules
│   │   ├── ingestion.py          # Data loading and chunked reading
│   │   ├── transformation.py     # Analytical transformations
│   │   └── etl_main.py          # Main pipeline orchestrator
│   ├── data/                     # Data processing utilities
│   │   ├── data_cleaner.py       # Main data cleaning orchestrator
│   │   ├── field_standardizer.py # Field-specific cleaning rules
│   │   ├── data_exporter.py      # Data export functionality
│   │   └── data_persistence.py   # Data caching and persistence
│   ├── quality/                  # Data quality and validation
│   │   ├── anomaly_detector.py   # Anomaly detection engine
│   │   ├── statistical_analyzer.py # Statistical analysis methods
│   │   └── validation_engine.py  # Data validation rules
│   ├── dashboard/                # Interactive dashboard
│   │   ├── dashboard_app.py      # Main Streamlit application
│   │   └── data_provider.py      # Dashboard data provider
│   └── utils/                    # Common utilities and helpers
│       ├── config.py             # Configuration management
│       ├── memory_manager.py     # Memory optimization
│       ├── error_handler.py      # Error handling system
│       ├── logger.py             # Logging configuration
│       └── test_data_generator.py # Test data generation
├── data/                         # Data storage
│   ├── raw/                      # Raw input data files
│   ├── processed/                # Cleaned and transformed data
│   ├── output/                   # Final processed outputs
│   │   ├── csv/                  # CSV format outputs
│   │   └── parquet/              # Parquet format outputs
│   └── temp/                     # Temporary processing files
├── examples/                     # Usage examples and demos
│   ├── data_export_example.py    # Data export demonstration
│   ├── anomaly_detection_example.py # Anomaly detection demo
│   └── test_data_generation_example.py # Test data generation demo
├── tests/                        # Comprehensive test suite
│   ├── test_data_cleaner.py      # Data cleaning tests
│   ├── test_anomaly_detector.py  # Anomaly detection tests
│   ├── test_transformation.py    # Transformation tests
│   └── test_integration.py       # End-to-end integration tests
├── config/                       # Configuration files
│   └── pipeline_config.yaml      # Main pipeline configuration
├── logs/                         # Application logs
├── reports/                      # Generated reports and monitoring
├── requirements.txt              # Python dependencies
└── README.md                    # This documentation
```

## Data Schema

### Input Data Format

The pipeline expects CSV files with the following schema:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `order_id` | String | Unique order identifier | ORD_000001 |
| `product_name` | String | Product name (may contain variations) | Laptop, laptop, LAPTOP |
| `category` | String | Product category (may need standardization) | Electronics, electronics, elec |
| `quantity` | Integer | Number of items ordered | 1, 2, 3 |
| `unit_price` | Float | Price per unit in local currency | 299.99 |
| `discount_percent` | Float | Discount as decimal (0.0-1.0) | 0.15 (15% discount) |
| `region` | String | Sales region (may need standardization) | North, north, nort |
| `sale_date` | String | Sale date in various formats | 2023-01-15, 15/01/2023 |
| `customer_email` | String | Customer email (may have validation issues) | user@example.com |

### Output Data Format

After processing, the pipeline generates:

1. **Cleaned Dataset**: Original data with standardized fields and calculated revenue
2. **Analytical Tables**:
   - `monthly_sales_summary`: Revenue, quantity, and discount by month
   - `top_products`: Top 10 products by revenue and units sold
   - `region_wise_performance`: Sales metrics by region
   - `category_discount_map`: Average discount by product category
   - `anomaly_records`: Top 5 records with suspicious patterns

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- At least 4GB RAM (configurable)
- 10GB+ free disk space for large datasets

### 2. Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Validate Setup

Run the setup validation to ensure everything is configured correctly:

```bash
python -m src.utils.setup_validator
```

### 5. Generate Test Data (Optional)

Create sample datasets for testing:

```bash
python examples/test_data_generation_example.py
```

This generates:
- Clean sample data (1,000 records)
- Dirty sample data with quality issues (1,000 records)
- Anomaly sample data with suspicious patterns (1,000 records)
- Large performance test dataset (100,000 records)

## Configuration

The pipeline uses YAML configuration files located in the `config/` directory:

### Main Configuration (`config/pipeline_config.yaml`)

```yaml
# Data Processing Settings
data_processing:
  chunk_size: 50000                    # Records per chunk
  max_memory_usage: 4294967296         # 4GB in bytes
  parallel_processing: true            # Enable parallel processing
  
# File Paths
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  output_data: "data/output"
  temp_data: "data/temp"
  
# Output Settings
output:
  formats: ["csv", "parquet"]          # Output formats
  include_metadata: true               # Include processing metadata
  compression: "gzip"                  # Compression for outputs
  
# Data Quality Settings
data_quality:
  anomaly_threshold: 3.0               # Z-score threshold for anomalies
  duplicate_handling: "flag"           # How to handle duplicates
  null_value_strategy: "impute"        # Strategy for null values
  
# Logging Configuration
logging:
  level: "INFO"                        # Log level
  file_rotation: true                  # Enable log file rotation
  max_file_size: "10MB"               # Max log file size
  backup_count: 5                      # Number of backup files
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 50,000 | Number of records to process at once |
| `max_memory_usage` | 4GB | Maximum memory usage limit |
| `anomaly_threshold` | 3.0 | Statistical threshold for anomaly detection |
| `output_formats` | ['csv', 'parquet'] | List of output formats |
| `log_level` | INFO | Logging verbosity level |
| `parallel_processing` | true | Enable multi-threading where possible |

## Usage

### 1. Basic Pipeline Execution

```bash
# Run the complete ETL pipeline
python src/pipeline/etl_main.py --input data/raw/sales_data.csv

# Run with custom configuration
python src/pipeline/etl_main.py --input data/raw/sales_data.csv --config config/custom_config.yaml

# Run with specific chunk size for memory optimization
python src/pipeline/etl_main.py --input data/raw/sales_data.csv --chunk-size 25000
```

### 2. Individual Component Usage

#### Data Ingestion
```python
from src.pipeline.ingestion import DataIngestionManager

ingestion_manager = DataIngestionManager()
for chunk in ingestion_manager.ingest_data("data/raw/sales_data.csv"):
    # Process each chunk
    print(f"Processing chunk with {len(chunk)} records")
```

#### Data Cleaning
```python
from src.data.data_cleaner import DataCleaner

cleaner = DataCleaner()
cleaned_chunk = cleaner.clean_chunk(raw_chunk)
cleaning_report = cleaner.generate_cleaning_report()
```

#### Anomaly Detection
```python
from src.quality.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(z_score_threshold=3.0)
anomalies = detector.detect_revenue_anomalies(data)
anomaly_report = detector.generate_anomaly_report()
```

### 3. Dashboard Usage

Launch the interactive dashboard:

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/dashboard_app.py

# Or use the dashboard runner
python run_dashboard.py
```

The dashboard provides:
- Monthly revenue trend visualizations
- Top products analysis
- Regional sales performance
- Category discount effectiveness
- Anomaly records display

### 4. Test Data Generation

Generate realistic test datasets:

```python
from src.utils.test_data_generator import TestDataGenerator

generator = TestDataGenerator(seed=42)

# Generate clean data
clean_data = generator.generate_clean_sample(size=10000)

# Generate dirty data with quality issues
dirty_data = generator.generate_dirty_sample(size=10000, dirty_ratio=0.3)

# Generate data with anomalies
anomaly_data = generator.generate_anomaly_sample(size=10000, anomaly_ratio=0.05)
```

## API Documentation

### Core Classes and Methods

#### DataIngestionManager
```python
class DataIngestionManager:
    """Orchestrates data ingestion with memory-efficient chunked reading."""
    
    def ingest_data(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Ingest data from CSV file in chunks."""
        
    def validate_file_format(self, file_path: str) -> bool:
        """Validate input file format and structure."""
        
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics and metrics."""
```

#### DataCleaner
```python
class DataCleaner:
    """Main data cleaning orchestrator with field-specific rules."""
    
    def clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean a data chunk applying all cleaning rules."""
        
    def generate_cleaning_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleaning report."""
```

#### FieldStandardizer
```python
class FieldStandardizer:
    """Handles field-specific cleaning and standardization."""
    
    def standardize_product_names(self, names: pd.Series) -> pd.Series:
        """Clean and standardize product names."""
        
    def standardize_categories(self, categories: pd.Series) -> pd.Series:
        """Standardize product categories."""
        
    def standardize_regions(self, regions: pd.Series) -> pd.Series:
        """Standardize region names."""
        
    def parse_dates(self, dates: pd.Series) -> pd.Series:
        """Parse dates in multiple formats."""
```

#### AnomalyDetector
```python
class AnomalyDetector:
    """Statistical anomaly detection for suspicious transactions."""
    
    def detect_revenue_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect revenue-based anomalies using statistical methods."""
        
    def detect_quantity_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect unusual quantity patterns."""
        
    def generate_anomaly_report(self) -> Dict[str, Any]:
        """Generate detailed anomaly detection report."""
```

#### TestDataGenerator
```python
class TestDataGenerator:
    """Generate realistic test datasets for pipeline testing."""
    
    def generate_clean_sample(self, size: int, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate clean e-commerce data with realistic patterns."""
        
    def generate_dirty_sample(self, size: int, dirty_ratio: float = 0.3) -> pd.DataFrame:
        """Generate data with common quality issues."""
        
    def generate_anomaly_sample(self, size: int, anomaly_ratio: float = 0.05) -> pd.DataFrame:
        """Generate data with known suspicious patterns."""
```

## Logging and Monitoring

The pipeline includes comprehensive logging and monitoring:

### Log Levels and Files
- **Console Output**: Real-time processing status and progress
- **File Logging**: Detailed logs with automatic rotation
  - `logs/pipeline.log`: Main pipeline execution logs
  - `logs/data_quality.log`: Data quality and cleaning logs
  - `logs/errors.log`: Error and exception logs
  - `logs/processing.log`: Performance and memory usage logs

### Log Configuration
```yaml
logging:
  level: INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
  console_output: true           # Enable console logging
  file_output: true             # Enable file logging
  rotation:
    max_size: "10MB"            # Maximum log file size
    backup_count: 5             # Number of backup files to keep
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Monitoring Features
- Memory usage tracking and alerts
- Processing speed and ETA calculations
- Data quality metrics and trends
- Error rate monitoring and reporting
- Pipeline health checks and status updates

## Features

- **Memory-efficient processing**: Handles datasets with 100M+ records through chunked processing
- **Configurable data cleaning**: YAML-based cleaning and validation rules
- **Statistical anomaly detection**: Z-score and IQR methods for identifying suspicious transactions
- **Multiple output formats**: Support for CSV and Parquet formats with compression
- **Interactive dashboard**: Streamlit-based visualization of business metrics
- **Comprehensive testing**: Unit, integration, and performance tests with realistic test data
- **Robust error handling**: Graceful error recovery and detailed error reporting
- **Modular architecture**: Clean separation of concerns across components
- **Performance optimization**: Memory management and parallel processing capabilities

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Memory Issues

**Problem**: `MemoryError` or system running out of memory
```
MemoryError: Unable to allocate array with shape (X,) and data type float64
```

**Solutions**:
- Reduce chunk size in configuration:
  ```yaml
  data_processing:
    chunk_size: 25000  # Reduce from default 50,000
  ```
- Increase system memory limit:
  ```yaml
  data_processing:
    max_memory_usage: 2147483648  # 2GB instead of 4GB
  ```
- Enable garbage collection checkpoints:
  ```python
  from src.utils.memory_manager import MemoryManager
  memory_manager = MemoryManager()
  memory_manager.enable_gc_checkpoints()
  ```

#### 2. Data Quality Issues

**Problem**: High number of cleaning errors or validation failures

**Solutions**:
- Check data quality report:
  ```python
  cleaning_report = cleaner.generate_cleaning_report()
  print(cleaning_report['error_summary'])
  ```
- Adjust cleaning rules in configuration
- Use test data generator to create sample data for testing:
  ```bash
  python examples/test_data_generation_example.py
  ```

#### 3. Performance Issues

**Problem**: Slow processing speed or high CPU usage

**Solutions**:
- Enable parallel processing:
  ```yaml
  data_processing:
    parallel_processing: true
  ```
- Optimize chunk size based on your system:
  ```python
  # Test different chunk sizes
  for chunk_size in [10000, 25000, 50000, 100000]:
      # Measure processing time
  ```
- Use Parquet format for better I/O performance:
  ```yaml
  output:
    formats: ["parquet"]  # Remove CSV for better performance
  ```

#### 4. File Format Issues

**Problem**: Unable to read input files or encoding errors

**Solutions**:
- Check file encoding:
  ```python
  import chardet
  with open('data/raw/file.csv', 'rb') as f:
      result = chardet.detect(f.read())
      print(result['encoding'])
  ```
- Specify encoding in ingestion:
  ```python
  ingestion_manager = DataIngestionManager(encoding='utf-8')
  ```
- Validate file format:
  ```python
  is_valid = ingestion_manager.validate_file_format('data/raw/file.csv')
  ```

#### 5. Dashboard Issues

**Problem**: Dashboard not loading or displaying errors

**Solutions**:
- Check if analytical tables exist:
  ```bash
  ls data/output/csv/
  ```
- Restart the dashboard:
  ```bash
  streamlit run src/dashboard/dashboard_app.py --server.port 8502
  ```
- Check dashboard logs:
  ```bash
  tail -f logs/dashboard.log
  ```

### Log Analysis

#### Finding Errors
```bash
# Search for errors in logs
grep -i "error" logs/pipeline.log

# Check memory usage patterns
grep -i "memory" logs/processing.log

# View data quality issues
grep -i "validation" logs/data_quality.log
```

#### Performance Analysis
```bash
# Check processing speed
grep -i "processing time" logs/pipeline.log

# Monitor memory usage
grep -i "memory usage" logs/processing.log

# View chunk processing statistics
grep -i "chunk" logs/pipeline.log
```

## FAQ

### General Questions

**Q: What is the maximum dataset size the pipeline can handle?**
A: The pipeline is designed to handle datasets with 100M+ records. The actual limit depends on available system memory and disk space. With 8GB RAM, you can typically process datasets up to 500M records using appropriate chunk sizes.

**Q: Can I run the pipeline on multiple files simultaneously?**
A: Currently, the pipeline processes one file at a time. However, you can process multiple files sequentially by running the pipeline multiple times with different input files.

**Q: What file formats are supported for input?**
A: Currently, the pipeline supports CSV files. The files can have various encodings (UTF-8, Latin-1, etc.) and different delimiter formats.

### Data Processing Questions

**Q: How does the pipeline handle duplicate records?**
A: Duplicate order IDs are detected and flagged during the cleaning process. You can configure the handling strategy in the configuration file (flag, remove, or keep first occurrence).

**Q: What happens to records that fail validation?**
A: Invalid records are logged with detailed error information but are not removed from the dataset. Instead, they are flagged for manual review. You can configure the pipeline to either skip invalid records or attempt to clean them.

**Q: How accurate is the anomaly detection?**
A: The anomaly detection uses statistical methods (Z-score and IQR) with configurable thresholds. The default threshold of 3.0 standard deviations captures approximately 0.3% of records as anomalies, which is suitable for most business scenarios.

### Technical Questions

**Q: Can I customize the data cleaning rules?**
A: Yes, cleaning rules are defined in the `FieldStandardizer` class and can be customized. You can also add new cleaning rules by extending the class or modifying the configuration file.

**Q: How do I add new analytical transformations?**
A: New transformations can be added to the `AnalyticalTransformer` class. Follow the existing pattern of creating methods that return pandas DataFrames with the desired aggregations.

**Q: Is the pipeline thread-safe?**
A: Yes, the pipeline is designed to be thread-safe. However, parallel processing is currently limited to certain operations to ensure data consistency.

### Performance Questions

**Q: How can I optimize processing speed?**
A: Key optimization strategies include:
- Adjusting chunk size based on available memory
- Using Parquet format for better I/O performance
- Enabling parallel processing where supported
- Using SSD storage for better disk I/O

**Q: What are the recommended system requirements?**
A: Minimum requirements:
- Python 3.8+
- 4GB RAM (8GB+ recommended for large datasets)
- 10GB+ free disk space
- Multi-core CPU (4+ cores recommended)

**Q: How do I monitor pipeline performance?**
A: The pipeline includes built-in performance monitoring:
- Check `logs/processing.log` for detailed performance metrics
- Use the monitoring dashboard for real-time statistics
- Enable memory profiling in the configuration for detailed memory usage analysis

## Quick Start Guide

After completing the setup, follow these steps to process your first dataset:

### 1. Prepare Your Data
Place your CSV files in the `data/raw/` directory:
```bash
cp your_sales_data.csv data/raw/
```

### 2. Configure the Pipeline
Edit `config/pipeline_config.yaml` to match your requirements:
```yaml
data_processing:
  chunk_size: 50000  # Adjust based on your system memory
  
paths:
  raw_data: "data/raw"
  output_data: "data/output"
```

### 3. Run the Pipeline
```bash
# Process your data
python src/pipeline/etl_main.py --input data/raw/your_sales_data.csv

# Or use the complete pipeline with dashboard
python run_pipeline.py --input data/raw/your_sales_data.csv --dashboard
```

### 4. View Results
- **Processed Data**: Check `data/output/csv/` and `data/output/parquet/`
- **Analytical Tables**: Review generated business intelligence tables
- **Dashboard**: Access the interactive dashboard at `http://localhost:8501`
- **Reports**: Check `reports/` for data quality and processing reports

## Development and Testing

### Development Setup

For development and contributing:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/ examples/

# Type checking
mypy src/

# Linting
flake8 src/ tests/ examples/
```

### Running Tests

The project includes a comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_data_cleaner.py          # Data cleaning tests
pytest tests/test_anomaly_detector.py      # Anomaly detection tests
pytest tests/test_transformation.py        # Transformation tests
pytest tests/test_integration.py           # End-to-end tests

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/test_performance.py -v
```

### Test Data Generation

Generate test datasets for development:

```bash
# Generate various test datasets
python examples/test_data_generation_example.py

# Generate specific dataset types
python -c "
from src.utils.test_data_generator import TestDataGenerator
gen = TestDataGenerator()
data = gen.generate_dirty_sample(10000, dirty_ratio=0.4)
gen.save_dataset(data, 'dev_test_data.csv', 'data/raw')
"
```

### Code Quality Standards

- **Code Style**: Follow PEP 8 with Black formatting
- **Type Hints**: Use type hints for all public methods
- **Documentation**: Include docstrings for all classes and methods
- **Testing**: Maintain >90% test coverage
- **Error Handling**: Implement comprehensive error handling with logging

## Performance Benchmarks

### Processing Speed
- **Small datasets** (< 1M records): ~1,000 records/second
- **Medium datasets** (1M-10M records): ~800 records/second
- **Large datasets** (10M-100M records): ~600 records/second

### Memory Usage
- **Base memory**: ~200MB for pipeline initialization
- **Per chunk**: ~50-100MB depending on chunk size and data complexity
- **Peak memory**: Typically 2-3x the configured chunk size in memory

### Disk Space Requirements
- **Input data**: Original CSV file size
- **Processed data**: ~1.5x input size (CSV + Parquet outputs)
- **Temporary files**: ~0.5x input size during processing
- **Logs and reports**: ~10-50MB depending on dataset size

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 10GB free space
- **CPU**: 2+ cores

### Recommended Requirements
- **RAM**: 16GB+ for datasets >50M records
- **Storage**: SSD with 50GB+ free space
- **CPU**: 4+ cores with hyperthreading
- **Network**: For dashboard access and data downloads

## Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **streamlit**: Interactive dashboard framework
- **plotly**: Interactive visualizations
- **pyyaml**: Configuration file parsing
- **psutil**: System and process monitoring

### Optional Dependencies
- **pyarrow**: Parquet file format support
- **openpyxl**: Excel file support
- **sqlalchemy**: Database connectivity
- **redis**: Caching support

See `requirements.txt` for complete dependency list with versions.

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow code style** guidelines (Black, flake8, mypy)
4. **Update documentation** for any API changes
5. **Submit a pull request** with a clear description

### Contribution Areas
- Performance optimizations
- Additional data cleaning rules
- New analytical transformations
- Dashboard enhancements
- Test coverage improvements
- Documentation updates

## License and Acknowledgments

This project is part of an AI-assisted data engineering assignment, demonstrating best practices in:
- Scalable data processing
- Memory-efficient algorithms
- Comprehensive testing
- Interactive data visualization
- Production-ready error handling

**Acknowledgments**:
- Built with modern Python data science stack
- Inspired by industry best practices in data engineering
- Designed for educational and practical use cases

## Support and Contact

For questions, issues, or contributions:
- **Issues**: Use the GitHub issue tracker
- **Documentation**: Check this README and inline code documentation
- **Examples**: Review the `examples/` directory for usage patterns
- **Tests**: Examine the `tests/` directory for implementation details

---

**Last Updated**: August 2025  
**Version**: 1.0.0  
**Python Compatibility**: 3.8+