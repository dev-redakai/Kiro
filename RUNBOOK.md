# Scalable Data Pipeline - Complete Run-Book

## 🚀 Quick Start (One Command)

```bash
# Run everything automatically
./run_all.sh
```

## 📋 Step-by-Step Feature Guide

### 1. Environment Setup

#### 1.1 Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Purpose**: Activates the Python virtual environment with all required dependencies.

#### 1.2 Verify Installation
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(pandas|streamlit|numpy)"
```

**Purpose**: Ensures all required packages are installed and accessible.

### 2. Data Generation Features

#### 2.1 Generate Clean Sample Data
```bash
# Generate 10,000 clean records
python -c "
import sys; sys.path.insert(0, 'src')
from utils.test_data_generator import TestDataGenerator
generator = TestDataGenerator()
data = generator.generate_clean_sample(size=10000)
data.to_csv('data/raw/clean_sample.csv', index=False)
print(f'Generated {len(data)} clean records')
"
```

**Purpose**: Creates realistic, clean e-commerce data for testing and demonstration.
**Output**: `data/raw/clean_sample.csv`

#### 2.2 Generate Dirty Sample Data (with Quality Issues)
```bash
# Generate 5,000 dirty records with 30% quality issues
python -c "
import sys; sys.path.insert(0, 'src')
from utils.test_data_generator import TestDataGenerator
generator = TestDataGenerator()
data = generator.generate_dirty_sample(size=5000, dirty_ratio=0.3)
data.to_csv('data/raw/dirty_sample.csv', index=False)
print(f'Generated {len(data)} dirty records with quality issues')
"
```

**Purpose**: Creates data with common quality issues (missing values, invalid formats, etc.) to test cleaning capabilities.
**Output**: `data/raw/dirty_sample.csv`

#### 2.3 Generate Anomaly Test Data
```bash
# Generate data with known anomalies
python -c "
import sys; sys.path.insert(0, 'src')
from utils.test_data_generator import TestDataGenerator
generator = TestDataGenerator()
data = generator.generate_anomaly_sample(size=3000, anomaly_ratio=0.05)
data.to_csv('data/raw/anomaly_sample.csv', index=False)
print(f'Generated {len(data)} records with anomalies')
"
```

**Purpose**: Creates data with known anomalous patterns for testing anomaly detection algorithms.
**Output**: `data/raw/anomaly_sample.csv`

#### 2.4 Generate Large Performance Test Data
```bash
# Generate large dataset for performance testing
python -c "
import sys; sys.path.insert(0, 'src')
from utils.test_data_generator import TestDataGenerator
generator = TestDataGenerator()
data = generator.generate_clean_sample(size=100000)
data.to_csv('data/raw/large_sample.csv', index=False)
print(f'Generated {len(data)} records for performance testing')
"
```

**Purpose**: Creates large datasets to test pipeline performance and memory efficiency.
**Output**: `data/raw/large_sample.csv`

### 3. ETL Pipeline Operations

#### 3.1 Run Complete ETL Pipeline
```bash
# Process data through complete pipeline
python -m src.pipeline.etl_main --input data/raw/clean_sample.csv
```

**Purpose**: Executes the complete ETL workflow: ingestion → cleaning → transformation → anomaly detection → export.
**Output**: 
- Processed data in `data/output/csv/` and `data/output/parquet/`
- Analytical tables
- Processing logs in `logs/`

#### 3.2 Run Pipeline with Custom Configuration
```bash
# Use custom configuration file
python -m src.pipeline.etl_main --input data/raw/clean_sample.csv --config config/pipeline_config.yaml
```

**Purpose**: Runs pipeline with specific configuration settings (chunk size, memory limits, etc.).

#### 3.3 Run Pipeline with Debug Logging
```bash
# Enable detailed debug logging
python -m src.pipeline.etl_main --input data/raw/clean_sample.csv --log-level DEBUG
```

**Purpose**: Provides detailed logging for troubleshooting and monitoring pipeline execution.

#### 3.4 Dry Run (Validation Only)
```bash
# Validate configuration without processing
python -m src.pipeline.etl_main --input data/raw/clean_sample.csv --dry-run
```

**Purpose**: Validates input data and configuration without actually processing data.

#### 3.5 Skip Specific Pipeline Stages
```bash
# Skip cleaning and export stages
python -m src.pipeline.etl_main --input data/raw/clean_sample.csv --skip-stages cleaning,export
```

**Purpose**: Allows selective execution of pipeline stages for testing or partial processing.

#### 3.6 Check Pipeline Status
```bash
# View current pipeline status
python -m src.pipeline.etl_main --status-only
```

**Purpose**: Displays current pipeline status, performance metrics, and health information.

### 4. Data Quality and Validation

## 📋 Complete Validation Rules Reference

The data pipeline implements 13 comprehensive validation rules to ensure data quality and business logic integrity. Each rule is designed to catch specific data quality issues that could impact business decisions.

### 4.0 Validation Rules Overview

| Rule Name | Field | Type | Severity | Business Impact |
|-----------|-------|------|----------|-----------------|
| `order_id_not_null` | order_id | NOT_NULL | CRITICAL | Prevents orphaned transactions |
| `order_id_unique` | order_id | UNIQUE | ERROR | Ensures transaction integrity |
| `product_name_not_null` | product_name | NOT_NULL | ERROR | Maintains product catalog integrity |
| `product_name_length` | product_name | RANGE | WARNING | Ensures display compatibility |
| `category_enum` | category | ENUM | WARNING | Maintains category standardization |
| `quantity_positive` | quantity | RANGE | ERROR | Prevents invalid order quantities |
| `quantity_statistical` | quantity | STATISTICAL | WARNING | Detects unusual order patterns |
| `unit_price_positive` | unit_price | RANGE | ERROR | Ensures valid pricing |
| `discount_range` | discount_percent | RANGE | ERROR | Validates discount calculations |
| `region_enum` | region | ENUM | WARNING | Maintains geographic consistency |
| `email_format` | customer_email | FORMAT | WARNING | Ensures communication capability |
| `sale_date_range` | sale_date | RANGE | ERROR | Validates transaction timing |
| `revenue_calculation` | revenue | BUSINESS_LOGIC | ERROR | Ensures financial accuracy |

### 4.0.1 Detailed Validation Rules

#### **1. Order ID Not Null (`order_id_not_null`)**
- **Rule Type**: NOT_NULL
- **Severity**: CRITICAL
- **Field**: order_id
- **Implementation**: `src/quality/automated_validation.py` (lines 155-163)
- **Check**: Ensures every record has a valid order ID
- **Business Impact**: 
  - Prevents orphaned transactions that can't be tracked
  - Essential for order fulfillment and customer service
  - Critical for financial reconciliation
- **Failure Scenarios**: NULL, empty string, or missing order_id values
- **Dashboard Display**: Shows as CRITICAL error with red indicator

#### **2. Order ID Unique (`order_id_unique`)**
- **Rule Type**: UNIQUE
- **Severity**: ERROR
- **Field**: order_id
- **Implementation**: `src/quality/automated_validation.py` (lines 164-172)
- **Check**: Verifies each order ID appears only once in the dataset
- **Business Impact**:
  - Prevents duplicate order processing
  - Ensures accurate revenue calculations
  - Maintains data integrity for reporting
- **Failure Scenarios**: Duplicate order IDs in the same dataset
- **Dashboard Display**: Shows duplicate count and sample duplicate values

#### **3. Product Name Not Null (`product_name_not_null`)**
- **Rule Type**: NOT_NULL
- **Severity**: ERROR
- **Field**: product_name
- **Implementation**: `src/quality/automated_validation.py` (lines 174-182)
- **Check**: Ensures every product has a valid name
- **Business Impact**:
  - Essential for product catalog management
  - Required for customer-facing displays
  - Needed for inventory tracking
- **Failure Scenarios**: NULL, empty, or missing product names
- **Dashboard Display**: Shows count of records with missing product names

#### **4. Product Name Length (`product_name_length`)**
- **Rule Type**: RANGE
- **Severity**: WARNING
- **Field**: product_name
- **Implementation**: `src/quality/automated_validation.py` (lines 183-191)
- **Parameters**: min_length=1, max_length=200
- **Check**: Validates product name length is within acceptable range
- **Business Impact**:
  - Ensures compatibility with UI display limits
  - Prevents database field overflow
  - Maintains consistent product presentation
- **Failure Scenarios**: Names shorter than 1 or longer than 200 characters
- **Dashboard Display**: Shows distribution of name lengths and outliers

#### **5. Category Enum (`category_enum`)**
- **Rule Type**: ENUM
- **Severity**: WARNING
- **Field**: category
- **Implementation**: `src/quality/automated_validation.py` (lines 194-202)
- **Parameters**: valid_values=['Electronics', 'Fashion', 'Home & Garden', 'Unknown']
- **Check**: Ensures category values match predefined list
- **Business Impact**:
  - Maintains consistent product categorization
  - Enables accurate category-based reporting
  - Supports marketing and inventory strategies
- **Failure Scenarios**: Categories not in the allowed list (e.g., "Home Goods")
- **Dashboard Display**: Shows invalid categories and suggests corrections

#### **6. Quantity Positive (`quantity_positive`)**
- **Rule Type**: RANGE
- **Severity**: ERROR
- **Field**: quantity
- **Implementation**: `src/quality/automated_validation.py` (lines 205-213)
- **Parameters**: min_value=1, max_value=10000
- **Check**: Ensures order quantities are positive and reasonable
- **Business Impact**:
  - Prevents invalid orders (zero or negative quantities)
  - Protects against data entry errors
  - Ensures accurate inventory calculations
- **Failure Scenarios**: Quantities ≤ 0 or > 10,000
- **Dashboard Display**: Shows distribution of invalid quantities

#### **7. Quantity Statistical (`quantity_statistical`)**
- **Rule Type**: STATISTICAL
- **Severity**: WARNING
- **Field**: quantity
- **Implementation**: `src/quality/automated_validation.py` (lines 215-223)
- **Parameters**: z_score_threshold=3.0
- **Check**: Detects statistical outliers in order quantities
- **Business Impact**:
  - Identifies unusual ordering patterns
  - Helps detect potential fraud or data errors
  - Supports inventory planning and demand forecasting
- **Failure Scenarios**: Quantities with z-score > 3.0 (statistical outliers)
- **Dashboard Display**: Shows outlier values and statistical thresholds

#### **8. Unit Price Positive (`unit_price_positive`)**
- **Rule Type**: RANGE
- **Severity**: ERROR
- **Field**: unit_price
- **Implementation**: `src/quality/automated_validation.py` (lines 226-234)
- **Parameters**: min_value=0.01, max_value=100000.0
- **Check**: Ensures unit prices are positive and within reasonable range
- **Business Impact**:
  - Prevents free or negative-priced items (unless intentional)
  - Protects revenue calculations
  - Maintains pricing integrity
- **Failure Scenarios**: Prices ≤ 0 or > $100,000
- **Dashboard Display**: Shows price distribution and invalid values

#### **9. Discount Range (`discount_range`)**
- **Rule Type**: RANGE
- **Severity**: ERROR
- **Field**: discount_percent
- **Implementation**: `src/quality/automated_validation.py` (lines 237-245)
- **Parameters**: min_value=0.0, max_value=1.0
- **Check**: Validates discount percentages are between 0% and 100%
- **Business Impact**:
  - Ensures accurate discount calculations
  - Prevents negative discounts or over-100% discounts
  - Maintains pricing and revenue integrity
- **Failure Scenarios**: Discounts < 0.0 or > 1.0
- **Dashboard Display**: Shows discount distribution and invalid values

#### **10. Region Enum (`region_enum`)**
- **Rule Type**: ENUM
- **Severity**: WARNING
- **Field**: region
- **Implementation**: `src/quality/automated_validation.py` (lines 248-256)
- **Parameters**: valid_values=['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest', 'Unknown']
- **Check**: Ensures region values match predefined geographic areas
- **Business Impact**:
  - Maintains consistent geographic reporting
  - Enables accurate regional sales analysis
  - Supports logistics and shipping optimization
- **Failure Scenarios**: Regions not in the allowed list
- **Dashboard Display**: Shows invalid regions and geographic distribution

#### **11. Email Format (`email_format`)**
- **Rule Type**: FORMAT
- **Severity**: WARNING
- **Field**: customer_email
- **Implementation**: `src/quality/automated_validation.py` (lines 259-267)
- **Check**: Validates email addresses follow proper format (when present)
- **Business Impact**:
  - Ensures customer communication capability
  - Supports marketing campaigns and notifications
  - Maintains customer contact data quality
- **Failure Scenarios**: Invalid email formats (missing @, invalid domains, etc.)
- **Dashboard Display**: Shows email format validation results and common errors

#### **12. Sale Date Range (`sale_date_range`)**
- **Rule Type**: RANGE
- **Severity**: ERROR
- **Field**: sale_date
- **Implementation**: `src/quality/automated_validation.py` (lines 269-277)
- **Parameters**: min_date='2020-01-01', max_date='2030-12-31'
- **Check**: Ensures sale dates are within reasonable business range
- **Business Impact**:
  - Prevents future-dated or historically invalid transactions
  - Ensures accurate time-series analysis
  - Maintains temporal data integrity
- **Failure Scenarios**: Dates before 2020 or after 2030
- **Dashboard Display**: Shows date distribution and temporal outliers

#### **13. Revenue Calculation (`revenue_calculation`)**
- **Rule Type**: BUSINESS_LOGIC
- **Severity**: ERROR
- **Field**: revenue
- **Implementation**: `src/quality/automated_validation.py` (lines 280-288)
- **Check**: Validates revenue = quantity × unit_price × (1 - discount_percent)
- **Business Impact**:
  - Ensures financial calculation accuracy
  - Prevents revenue reporting errors
  - Maintains financial data integrity
- **Failure Scenarios**: Revenue doesn't match calculated value (within tolerance)
- **Dashboard Display**: Shows calculation discrepancies and financial impact

### 4.0.2 Validation Configuration

#### **Configuration File**: `config/pipeline_config.yaml`
```yaml
validation_rules:
  quantity:
    min_value: 1
    max_value: 10000
  unit_price:
    min_value: 0.01
    max_value: 100000.0
  discount_percent:
    min_value: 0.0
    max_value: 1.0
  statistical_outliers:
    z_score_threshold: 3.0
  date_range:
    min_date: '2020-01-01'
    max_date: '2030-12-31'
```

#### **Validation Engine Files**:
- **Primary Implementation**: `src/quality/automated_validation.py`
- **Supporting Engine**: `src/quality/validation_engine.py`
- **Dashboard Integration**: `src/dashboard/data_provider.py`
- **Configuration**: `config/pipeline_config.yaml`

### 4.0.3 Business Impact Summary

#### **Critical Rules (CRITICAL Severity)**
- **order_id_not_null**: Prevents complete transaction tracking failure
- **Impact**: System cannot function without valid order IDs

#### **Error Rules (ERROR Severity)**
- **order_id_unique**: Prevents duplicate processing and financial errors
- **product_name_not_null**: Ensures product catalog integrity
- **quantity_positive**: Prevents invalid order processing
- **unit_price_positive**: Maintains pricing integrity
- **discount_range**: Ensures accurate financial calculations
- **sale_date_range**: Maintains temporal data integrity
- **revenue_calculation**: Ensures financial accuracy
- **Impact**: These errors can cause significant business and financial issues

#### **Warning Rules (WARNING Severity)**
- **product_name_length**: Prevents display and database issues
- **category_enum**: Maintains categorization consistency
- **quantity_statistical**: Identifies unusual patterns
- **region_enum**: Maintains geographic consistency
- **email_format**: Ensures communication capability
- **Impact**: These warnings indicate data quality issues that should be addressed but don't prevent processing

### 4.0.4 Dashboard Integration

#### **Data Quality Dashboard Sections**:
1. **Quality Overview**: Overall score and rule execution statistics
2. **Validation Rules**: Detailed rule-by-rule results and performance
3. **Error Analysis**: Problematic values and error distribution
4. **Performance Metrics**: Rule execution times and system performance
5. **Quality Log**: Real-time validation events and history

#### **Accessing Validation Results**:
```bash
# View validation results in dashboard
streamlit run src/dashboard/dashboard_app.py
# Navigate to "Data Quality" tab
```

#### **Validation Report Files**:
- **Debug Reports**: `logs/validation_debug_report_*.json`
- **Quality Logs**: `logs/data_quality.log`
- **Error Reports**: `logs/validation_error_report.json`

#### 4.1 Run Data Quality Validation
```bash
# Validate data quality
python -c "
import sys; sys.path.insert(0, 'src')
from quality.automated_validation import validate_data
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
result = validate_data(data)
print(f'Validation Status: {result[\"status\"]}')
for r in result['results']:
    print(f'- {r[\"field_name\"]}: {\"PASS\" if r[\"passed\"] else \"FAIL\"} ({r[\"error_rate\"]:.2f}% error rate)')
"
```

**Purpose**: Runs comprehensive data quality validation with 12+ validation rules.
**Output**: Detailed validation report with pass/fail status for each rule.

#### 4.2 Run Anomaly Detection
```bash
# Detect anomalies in data
python -c "
import sys; sys.path.insert(0, 'src')
from quality.anomaly_detector import AnomalyDetector
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
if 'revenue' not in data.columns:
    data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])

detector = AnomalyDetector()
anomalies = detector.detect_all_anomalies(data)
anomaly_records = detector.generate_top_anomaly_records(anomalies)
print(f'Detected {len(anomaly_records)} anomalous records')
print(anomaly_records[['order_id', 'product_name', 'revenue', 'anomaly_score']].head())
"
```

**Purpose**: Identifies suspicious transactions and patterns using statistical and rule-based methods.
**Output**: List of anomalous records with anomaly scores.

#### 4.3 Generate Data Quality Report
```bash
# Generate comprehensive data quality report
python -c "
import sys; sys.path.insert(0, 'src')
from quality.statistical_analyzer import StatisticalAnalyzer
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
analyzer = StatisticalAnalyzer()
report = analyzer.generate_comprehensive_report(data)
print('Data Quality Report:')
for field, stats in report.items():
    print(f'- {field}: {stats}')
"
```

**Purpose**: Generates detailed statistical analysis and data profiling report.
**Output**: Comprehensive data quality metrics and statistics.

### 5. Data Transformation and Analytics

#### 5.1 Generate Monthly Sales Summary
```bash
# Create monthly sales analytics
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline.transformation import AnalyticalTransformer
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])

transformer = AnalyticalTransformer()
monthly_summary = transformer.create_monthly_sales_summary(data)
monthly_summary.to_csv('data/output/monthly_sales_summary.csv', index=False)
print(f'Generated monthly sales summary: {len(monthly_summary)} records')
print(monthly_summary.head())
"
```

**Purpose**: Creates monthly revenue, order count, and discount analysis.
**Output**: `data/output/monthly_sales_summary.csv`

#### 5.2 Generate Top Products Analysis
```bash
# Create top products ranking
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline.transformation import AnalyticalTransformer
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])

transformer = AnalyticalTransformer()
top_products = transformer.create_top_products_table(data)
top_products.to_csv('data/output/top_products.csv', index=False)
print(f'Generated top products analysis: {len(top_products)} records')
print(top_products.head())
"
```

**Purpose**: Identifies best-performing products by revenue and units sold.
**Output**: `data/output/top_products.csv`

#### 5.3 Generate Regional Performance Analysis
```bash
# Create regional sales analysis
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline.transformation import AnalyticalTransformer
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])

transformer = AnalyticalTransformer()
regional_perf = transformer.create_region_performance_table(data)
regional_perf.to_csv('data/output/region_wise_performance.csv', index=False)
print(f'Generated regional performance analysis: {len(regional_perf)} records')
print(regional_perf.head())
"
```

**Purpose**: Analyzes sales performance by geographic region with market share.
**Output**: `data/output/region_wise_performance.csv`

#### 5.4 Generate Category Discount Analysis
```bash
# Create category discount effectiveness analysis
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline.transformation import AnalyticalTransformer
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
data['revenue'] = data['quantity'] * data['unit_price'] * (1 - data['discount_percent'])

transformer = AnalyticalTransformer()
category_discounts = transformer.create_category_discount_map(data)
category_discounts.to_csv('data/output/category_discount_map.csv', index=False)
print(f'Generated category discount analysis: {len(category_discounts)} records')
print(category_discounts.head())
"
```

**Purpose**: Analyzes discount effectiveness and penetration by product category.
**Output**: `data/output/category_discount_map.csv`

### 6. Dashboard Operations

#### 6.1 Generate Dashboard Data from ETL Pipeline Output
```bash
# Generate analytical tables from ETL pipeline processed data
python generate_dashboard_from_etl.py
```

**Purpose**: Creates analytical tables using actual processed data from ETL pipeline.
**Output**: All analytical CSV files in `data/output/` based on real ETL results.
**Priority**: Uses latest processed data from ETL pipeline for authentic insights.

#### 6.2 Generate Sample Dashboard Data (Fallback)
```bash
# Generate sample analytical tables for testing
python generate_dashboard_data.py
```

**Purpose**: Creates sample analytical tables when ETL pipeline output isn't available.
**Output**: Sample analytical CSV files in `data/output/`
**Use Case**: Development, testing, or when ETL pipeline hasn't run yet.

#### 6.3 Start Dashboard (Interactive Mode)
```bash
# Start Streamlit dashboard
streamlit run src/dashboard/dashboard_app.py
```

**Purpose**: Launches interactive web dashboard for data visualization and analysis.
**Access**: http://localhost:8501

#### 6.4 Start Dashboard (Headless Mode)
```bash
# Start dashboard without browser
streamlit run src/dashboard/dashboard_app.py --server.headless true
```

**Purpose**: Runs dashboard in background mode for server deployments.

#### 6.5 Start Dashboard on Custom Port
```bash
# Start dashboard on port 8502
streamlit run src/dashboard/dashboard_app.py --server.port 8502
```

**Purpose**: Runs dashboard on specific port to avoid conflicts.

#### 6.6 Test Dashboard Data Provider
```bash
# Test dashboard data loading
python -c "
import sys; sys.path.insert(0, 'src')
from dashboard.data_provider import DataProvider

provider = DataProvider()
tables = provider.load_analytical_tables()
metrics = provider.get_dashboard_metrics()

print(f'Loaded {len(tables)} analytical tables')
print(f'Total Revenue: ${metrics.get(\"total_revenue\", 0):,.2f}')
print(f'Total Orders: {metrics.get(\"total_orders\", 0):,}')
"
```

**Purpose**: Validates dashboard data loading and metrics calculation.

### 7. Data Export and Output

#### 7.1 Export Data to CSV
```bash
# Export processed data to CSV
python -c "
import sys; sys.path.insert(0, 'src')
from data.data_exporter import DataExporter
from utils.config import ConfigManager
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
config = ConfigManager().config
exporter = DataExporter(config=config)
exporter.export_processed_data(data, 'manual_export')
print('Data exported to CSV format')
"
```

**Purpose**: Exports processed data in CSV format with timestamp.
**Output**: `data/output/csv/manual_export_YYYYMMDD_HHMMSS.csv`

#### 7.2 Export Data to Parquet
```bash
# Export processed data to Parquet
python -c "
import sys; sys.path.insert(0, 'src')
from data.data_exporter import DataExporter
from utils.config import ConfigManager
import pandas as pd

data = pd.read_csv('data/raw/clean_sample.csv')
config = ConfigManager().config
exporter = DataExporter(config=config)
exporter.export_processed_data(data, 'manual_export')
print('Data exported to Parquet format')
"
```

**Purpose**: Exports processed data in efficient Parquet format.
**Output**: `data/output/parquet/manual_export_YYYYMMDD_HHMMSS.parquet`

#### 7.3 Export Analytical Tables
```bash
# Export all analytical tables
python -c "
import sys; sys.path.insert(0, 'src')
from data.data_exporter import DataExporter
from utils.config import ConfigManager
import pandas as pd
from pathlib import Path

config = ConfigManager().config
exporter = DataExporter(config=config)

# Load analytical tables
tables = {}
output_dir = Path('data/output')
for file in output_dir.glob('*.csv'):
    if file.stem in ['monthly_sales_summary', 'top_products', 'region_wise_performance', 'category_discount_map']:
        tables[file.stem] = pd.read_csv(file)

exporter.export_analytical_tables(tables)
print(f'Exported {len(tables)} analytical tables')
"
```

**Purpose**: Exports all analytical tables in both CSV and Parquet formats.

### 8. Monitoring and Health Checks

#### 8.1 Check System Health
```bash
# Run system health check
python -c "
import sys; sys.path.insert(0, 'src')
from utils.pipeline_health import pipeline_health_monitor

health = pipeline_health_monitor.get_health_status()
print(f'Overall Health: {health[\"overall_status\"]}')
for component, status in health['component_health'].items():
    print(f'- {component}: {status[\"status\"]}')
"
```

**Purpose**: Checks system health including memory, CPU, disk, and data quality.

#### 8.2 Monitor Performance Metrics
```bash
# Get performance metrics
python -c "
import sys; sys.path.insert(0, 'src')
from utils.performance_manager import PerformanceManager

perf_manager = PerformanceManager()
metrics = perf_manager.get_performance_summary()
print('Performance Metrics:')
for metric, value in metrics.items():
    print(f'- {metric}: {value}')
"
```

**Purpose**: Displays current performance metrics and system utilization.

#### 8.3 Check Memory Usage
```bash
# Monitor memory usage
python -c "
import sys; sys.path.insert(0, 'src')
from utils.memory_manager import MemoryManager

memory_manager = MemoryManager()
usage = memory_manager.check_memory_usage()
print(f'Memory Usage: {usage[\"process_memory_mb\"]:.1f} MB')
print(f'System Memory: {usage[\"system_memory_percent\"]:.1f}%')
print(f'Status: {usage[\"status\"]}')
"
```

**Purpose**: Monitors current memory usage and provides optimization recommendations.

### 9. Testing and Validation

#### 9.1 Run Unit Tests
```bash
# Run all unit tests
pytest tests/ -v
```

**Purpose**: Executes comprehensive unit test suite for all components.

#### 9.2 Run Integration Tests
```bash
# Run integration tests
python integration_test_runner.py
```

**Purpose**: Validates end-to-end integration of all pipeline components.

#### 9.3 Run Performance Tests
```bash
# Run performance benchmarks
pytest tests/test_performance_manager.py -v
```

**Purpose**: Validates performance benchmarks and optimization effectiveness.

#### 9.4 Validate Pipeline Integration
```bash
# Validate complete pipeline integration
python validate_pipeline_integration.py
```

**Purpose**: Comprehensive validation of all pipeline components working together.

### 10. Configuration Management

#### 10.1 Validate Configuration
```bash
# Validate pipeline configuration
python -c "
import sys; sys.path.insert(0, 'src')
from utils.config import ConfigManager

config_manager = ConfigManager()
errors = config_manager.validate_config()
if errors:
    print('Configuration Errors:')
    for error in errors:
        print(f'- {error}')
else:
    print('Configuration is valid')
"
```

**Purpose**: Validates pipeline configuration for correctness and completeness.

#### 10.2 Display Current Configuration
```bash
# Show current configuration
python -c "
import sys; sys.path.insert(0, 'src')
from utils.config import ConfigManager

config = ConfigManager().config
print(f'Chunk Size: {config.chunk_size:,}')
print(f'Max Memory: {config.max_memory_usage / (1024**3):.1f} GB')
print(f'Anomaly Threshold: {config.anomaly_threshold}')
print(f'Output Formats: {config.output_formats}')
"
```

**Purpose**: Displays current pipeline configuration settings.

### 11. Logging and Debugging

#### 11.1 View Recent Logs
```bash
# View last 50 lines of pipeline logs
tail -n 50 logs/pipeline.log
```

**Purpose**: Shows recent pipeline execution logs for monitoring and debugging.

#### 11.2 Search for Errors in Logs
```bash
# Find error messages in logs
grep -i error logs/pipeline.log
```

**Purpose**: Identifies error messages and issues in pipeline execution.

#### 11.3 Monitor Real-time Logs
```bash
# Follow logs in real-time
tail -f logs/pipeline.log
```

**Purpose**: Monitors pipeline execution in real-time during processing.

### 12. Data Cleanup and Maintenance

#### 12.1 Clean Temporary Files
```bash
# Remove temporary processing files
rm -rf data/temp/*
echo "Temporary files cleaned"
```

**Purpose**: Cleans up temporary files created during pipeline processing.

#### 12.2 Archive Old Outputs
```bash
# Archive outputs older than 7 days
find data/output -name "*.csv" -mtime +7 -exec mv {} data/archive/ \;
echo "Old outputs archived"
```

**Purpose**: Archives old output files to prevent disk space issues.

#### 12.3 Rotate Log Files
```bash
# Rotate log files
mv logs/pipeline.log logs/pipeline_$(date +%Y%m%d).log
touch logs/pipeline.log
echo "Log files rotated"
```

**Purpose**: Rotates log files to prevent them from growing too large.

## 🔧 Troubleshooting Commands

### Fix Common Issues

#### Memory Issues
```bash
# Reduce chunk size for memory-constrained environments
python -m src.pipeline.etl_main --input data/raw/clean_sample.csv --config config/low_memory_config.yaml
```

#### Import Errors
```bash
# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m src.pipeline.etl_main --input data/raw/clean_sample.csv
```

#### Permission Issues
```bash
# Fix file permissions
chmod -R 755 data/
chmod -R 755 logs/
```

## 📊 Performance Optimization Commands

### Optimize for Large Datasets
```bash
# Use optimized configuration for large datasets
python -m src.pipeline.etl_main --input data/raw/large_sample.csv --config config/high_performance_config.yaml
```

### Enable Parallel Processing
```bash
# Enable multi-threading
python -c "
import sys; sys.path.insert(0, 'src')
from utils.config import ConfigManager
config = ConfigManager()
config.config.enable_parallel_processing = True
print('Parallel processing enabled')
"
```

---

## 🎯 Quick Reference

| Feature | Command | Purpose |
|---------|---------|---------|
| Generate Data | `python generate_dashboard_data.py` | Create sample data |
| Run Pipeline | `python -m src.pipeline.etl_main --input data/raw/sample.csv` | Process data |
| Start Dashboard | `streamlit run src/dashboard/dashboard_app.py` | Launch UI |
| Run Tests | `pytest tests/ -v` | Validate system |
| Check Health | `python validate_pipeline_integration.py` | System status |
| View Logs | `tail -f logs/pipeline.log` | Monitor execution |

---

**All commands assume you're in the project root directory with the virtual environment activated.**