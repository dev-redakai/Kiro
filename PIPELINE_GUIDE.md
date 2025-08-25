# 🚀 Scalable Data Pipeline - Complete Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Pipeline Execution](#pipeline-execution)
3. [Data Generation & Management](#data-generation--management)
4. [Testing & Validation](#testing--validation)
5. [Technical Documentation](#technical-documentation)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### **For New Users (5 Minutes)**
```powershell
# 1. Generate test data
python generate_simple_raw_data.py

# 2. Run complete pipeline
./run_all.ps1 -InputFile data/raw/clean/medium_clean_sample.csv

# 3. Access dashboard
# Open browser to http://localhost:8501
```

### **For Developers**
```bash
# 1. Generate test data
python generate_simple_raw_data.py

# 2. Test components
python validate_pipeline_integration.py

# 3. Run with specific data type
./run_all.ps1 -InputFile data/raw/dirty/dirty_sample_30pct.csv
```

---

## 🎯 Pipeline Execution

### **Main Execution Script: `run_all.ps1`**

#### **Command Syntax**
```powershell
./run_all.ps1 [OPTIONS]
```

#### **Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-InputFile` | String | \"\" | Path to input raw data file (optional) |
| `-SampleSize` | Integer | 10000 | Number of sample records to generate if no input file |
| `-DashboardPort` | Integer | 8501 | Port for the Streamlit dashboard |
| `-SkipETL` | Switch | False | Skip ETL pipeline execution |
| `-SkipDashboard` | Switch | False | Skip dashboard startup |
| `-HealthCheckOnly` | Switch | False | Run health checks only |
| `-Help` | Switch | False | Show help message |

#### **Usage Examples**
```powershell
# Show help and available options
./run_all.ps1 -Help

# Default execution (generates 10K sample records)
./run_all.ps1

# Use specific input file
./run_all.ps1 -InputFile data/raw/clean/medium_clean_sample.csv

# Test with dirty data
./run_all.ps1 -InputFile data/raw/dirty/dirty_sample_30pct.csv

# Test with anomaly data
./run_all.ps1 -InputFile data/raw/anomaly/anomaly_sample_5pct.csv

# Performance testing
./run_all.ps1 -InputFile data/raw/performance/performance_100k.csv

# Custom port and skip dashboard
./run_all.ps1 -InputFile data/raw/clean/small_clean_sample.csv -DashboardPort 8502 -SkipDashboard

# Health checks only
./run_all.ps1 -HealthCheckOnly
```

### **Cross-Platform Execution**

#### **Windows PowerShell**
```powershell
./run_all.ps1 -InputFile data/raw/clean/medium_clean_sample.csv
```

#### **Linux/macOS Bash**
```bash
chmod +x run_all.sh
./run_all.sh
```

---

## 📊 Data Generation & Management

### **Raw Data Generator: `generate_simple_raw_data.py`**

#### **Purpose**
Generate various types of raw data for comprehensive pipeline testing.

#### **Usage**
```bash
python generate_simple_raw_data.py
```

#### **Generated Datasets**

##### **🧹 Clean Data (High Quality)**
- `data/raw/clean/small_clean_sample.csv` - 1,000 records
- `data/raw/clean/medium_clean_sample.csv` - 10,000 records  
- `data/raw/clean/large_clean_sample.csv` - 50,000 records

**Characteristics:**
- ✅ All validation rules pass
- ✅ Realistic e-commerce patterns
- ✅ Proper data types and formats
- ✅ Seasonal and regional variations
- ✅ Business logic consistency

##### **🔧 Dirty Data (Quality Issues)**
- `data/raw/dirty/dirty_sample_30pct.csv` - 5,000 records (30% dirty)
- `data/raw/dirty/dirty_sample_50pct.csv` - 3,000 records (50% dirty)

**Common Quality Issues:**
- ❌ Product Names: Case variations, extra spaces, special characters
- ❌ Categories: Invalid values, typos, inconsistent formatting
- ❌ Regions: Spelling variations, case issues, invalid regions
- ❌ Quantities: Zero/negative values, non-numeric strings, nulls
- ❌ Prices: Zero/negative prices, non-numeric values, nulls
- ❌ Discounts: Values > 1.0 or < 0.0, invalid formats
- ❌ Emails: Invalid formats, missing @ symbols, malformed domains
- ❌ Dates: Invalid formats, impossible dates, out-of-range values
- ❌ Duplicates: Duplicate order IDs
- ❌ Nulls: Missing values in critical fields

##### **🚨 Anomaly Data (Suspicious Patterns)**
- `data/raw/anomaly/anomaly_sample_5pct.csv` - 4,000 records (5% anomalies)
- `data/raw/anomaly/anomaly_sample_10pct.csv` - 2,000 records (10% anomalies)

**Anomaly Patterns:**
- 🔴 Extreme Revenue: Transactions 10-50x normal values
- 🔴 Extreme Quantities: Orders of 100-2000+ items
- 🔴 Extreme Discounts: 80-99% discounts on high-value items
- 🔴 Suspicious Combinations: High-value + high-quantity + high-discount
- 🔴 Price Manipulation: Round number pricing, extremely low prices

##### **🔀 Mixed Data (Realistic Distribution)**
- `data/raw/mixed/realistic_mixed_sample.csv` - 8,000 records
  - 70% clean, 25% dirty, 5% anomaly

##### **⚡ Performance Data (Large Scale)**
- `data/raw/performance/performance_100k.csv` - 100,000 records

##### **🧪 Validation Test Data (Rule-Specific Testing)**
- `data/raw/validation_test/null_order_ids_test.csv` - Tests `order_id_not_null` rule
- `data/raw/validation_test/duplicate_order_ids_test.csv` - Tests `order_id_unique` rule
- `data/raw/validation_test/invalid_categories_test.csv` - Tests `category_enum` rule
- `data/raw/validation_test/invalid_quantities_test.csv` - Tests quantity validation rules
- `data/raw/validation_test/invalid_emails_test.csv` - Tests `email_format` rule

### **Dashboard Data Generation**

#### **ETL-Based Dashboard Data: `generate_dashboard_from_etl.py`**
```bash
python generate_dashboard_from_etl.py
```
- **Purpose**: Generate dashboard data from actual ETL pipeline output
- **Use Case**: After ETL pipeline execution for real insights
- **Output**: Creates analytical tables from processed ETL data

#### **Sample Dashboard Data: `generate_dashboard_data.py`**
```bash
python generate_dashboard_data.py
```
- **Purpose**: Generate sample analytical data for dashboard testing
- **Use Case**: When ETL pipeline hasn't run yet or for development
- **Output**: Creates sample analytical tables in `data/output/`

### **Data Schema**

#### **Raw Data Schema**
```csv
order_id,product_name,category,quantity,unit_price,discount_percent,region,sale_date,customer_email,revenue
ORD_000000,Cookware Set,Home Goods,1,199.66,0.1870,North,2023-02-12 00:00:00,customer_000000@example.com,162.34
```

#### **Field Descriptions**
| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| `order_id` | String | Unique order identifier | NOT_NULL, UNIQUE |
| `product_name` | String | Product name | NOT_NULL, LENGTH(1-200) |
| `category` | String | Product category | ENUM(Electronics, Fashion, Home & Garden, Unknown) |
| `quantity` | Integer | Order quantity | RANGE(1-10000), STATISTICAL |
| `unit_price` | Float | Price per unit | RANGE(0.01-100000) |
| `discount_percent` | Float | Discount percentage | RANGE(0.0-1.0) |
| `region` | String | Geographic region | ENUM(North, South, East, West, etc.) |
| `sale_date` | DateTime | Sale timestamp | RANGE(2020-01-01 to 2030-12-31) |
| `customer_email` | String | Customer email | FORMAT(email) |
| `revenue` | Float | Calculated revenue | BUSINESS_LOGIC |

---

## 🧪 Testing & Validation

### **Comprehensive Integration Testing**

#### **Main Integration Validator: `validate_pipeline_integration.py`**
```bash
python validate_pipeline_integration.py
```

**Features:**
- ETL pipeline integration
- Dashboard integration
- Component integration
- Configuration validation
- Data quality validation
- Performance validation

#### **End-to-End Testing: `final_integration_test.py`**
```bash
python final_integration_test.py
```
- **Purpose**: Final end-to-end pipeline testing
- **Use Case**: Pre-deployment validation

### **Component-Specific Testing**

#### **Dashboard Testing**
```bash
# Test dashboard components
python test_dashboard.py

# Launch dashboard only
python run_dashboard.py
```

#### **Data Processing Testing**
```bash
# Test chunked reading
python test_chunked_reader.py
python test_chunked_reader_edge_cases.py

# Test data ingestion
python test_ingestion.py

# Test ETL orchestrator
python test_etl_main.py
```

#### **Validation System Testing**
```bash
# Test validation logging
python test_enhanced_validation_logging.py

# Test monitoring system
python test_monitoring_system.py
```

### **Data Quality Validation Rules**

#### **Complete Validation Rules Reference**

| Rule Name | Field | Type | Severity | Business Impact |
|-----------|-------|------|----------|------------------|
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

#### **Expected Results by Data Type**

**Clean Data Results:**
- Data Quality Score: 95-100%
- Validation Rules: All pass
- Dashboard Status: All charts populated with clean data
- Processing Time: Fast (baseline performance)

**Dirty Data Results (30% dirty):**
- Data Quality Score: 70-85%
- Validation Rules: 3-5 rules fail
- Dashboard Status: Shows data quality issues in Data Quality tab
- Processing Time: Slightly slower due to cleaning operations

**Anomaly Data Results:**
- Data Quality Score: 85-95%
- Validation Rules: Most pass, statistical outlier warnings
- Dashboard Status: Anomaly Detection tab shows suspicious transactions
- Processing Time: Normal, with anomaly detection overhead

---

## 🔧 Technical Documentation

### **System Architecture**

#### **Pipeline Stages**
1. **Data Ingestion** - Chunked CSV reading with memory optimization
2. **Data Cleaning** - Field standardization and quality improvement
3. **Data Validation** - 13 comprehensive validation rules
4. **Data Transformation** - Analytical table generation
5. **Anomaly Detection** - Statistical and rule-based detection
6. **Data Export** - CSV and Parquet output formats

#### **Key Components**
- **ETL Orchestrator** (`src/pipeline/etl_main.py`)
- **Data Ingestion** (`src/pipeline/ingestion.py`)
- **Data Cleaning** (`src/pipeline/cleaning.py`)
- **Validation Engine** (`src/quality/automated_validation.py`)
- **Transformation Engine** (`src/pipeline/transformation.py`)
- **Anomaly Detector** (`src/quality/anomaly_detector.py`)
- **Dashboard App** (`src/dashboard/dashboard_app.py`)

### **Configuration**

#### **Main Configuration: `config/pipeline_config.yaml`**
```yaml
# Processing Configuration
processing:
  chunk_size: 10000
  max_memory_usage: 4294967296  # 4GB
  enable_parallel_processing: true
  max_workers: 8

# Validation Rules Configuration
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

# Output Configuration
output:
  formats: ['csv', 'parquet']
  compression: 'gzip'
  include_timestamp: true
```

### **Performance Specifications**

#### **Memory Requirements**
- **Small datasets (1K records)**: <100 MB RAM
- **Medium datasets (10K records)**: 100-500 MB RAM
- **Large datasets (50K records)**: 500 MB - 2 GB RAM
- **Performance datasets (100K records)**: 2-4 GB RAM

#### **Processing Time**
- **Small (1K records)**: ~10-30 seconds
- **Medium (10K records)**: ~30-60 seconds
- **Large (50K records)**: ~2-5 minutes
- **Performance (100K records)**: ~5-15 minutes

### **Output Files**

#### **Processed Data**
- `data/output/csv/processed_data_YYYYMMDD_HHMMSS.csv`
- `data/output/parquet/processed_data_YYYYMMDD_HHMMSS.parquet`

#### **Analytical Tables**
- `data/output/monthly_sales_summary.csv`
- `data/output/top_products.csv`
- `data/output/region_wise_performance.csv`
- `data/output/category_discount_map.csv`
- `data/output/anomaly_records.csv`

#### **Reports and Logs**
- `logs/pipeline.log` - Main pipeline execution log
- `logs/data_quality.log` - Data quality validation log
- `logs/validation_debug_report_*.json` - Detailed validation reports
- `logs/run_all.log` - Complete execution log

---

## 🔍 Troubleshooting

### **Common Issues**

#### **Input File Not Found**
```
ERROR: Input file not found: data\\raw\\clean\\medium_clean_sample.csv
```
**Solution**: Run `python generate_simple_raw_data.py` first to generate raw data files.

#### **Permission Errors**
```
ERROR: Access denied to logs\\run_all.log
```
**Solution**: Run PowerShell as Administrator or check file permissions.

#### **Python Environment Issues**
```
WARNING: Virtual environment not detected
```
**Solution**: Activate virtual environment first:
```powershell
venv\\Scripts\\Activate.ps1
```

#### **Dashboard Port Conflicts**
```
ERROR: Port 8501 is already in use
```
**Solution**: Use a different port:
```powershell
./run_all.ps1 -InputFile data/raw/clean/small_clean_sample.csv -DashboardPort 8502
```

#### **Memory Issues**
```
ERROR: Out of memory during processing
```
**Solution**: Use smaller datasets or increase available memory:
```powershell
# Use smaller dataset
./run_all.ps1 -InputFile data/raw/clean/small_clean_sample.csv

# Or reduce sample size
./run_all.ps1 -SampleSize 5000
```

### **Validation Debugging**

#### **Check Input Data**
```powershell
# Inspect first few rows
Get-Content data/raw/clean/small_clean_sample.csv | Select-Object -First 5
```

#### **Review Validation Logs**
```powershell
# Check validation debug report
Get-Content logs/validation_debug_report_*.json

# Check data quality log
Get-Content logs/data_quality.log
```

#### **Dashboard Data Quality Tab**
1. Open http://localhost:8501
2. Navigate to \"Data Quality\" tab
3. Review rule-by-rule results
4. Check error analysis and problematic values

### **Performance Optimization**

#### **For Large Datasets**
```powershell
# Use optimized configuration
./run_all.ps1 -InputFile data/raw/performance/performance_100k.csv
```

#### **Memory Optimization**
- Close other applications when running large datasets
- Monitor memory usage with Task Manager
- Use smaller chunk sizes if memory is limited

### **Log Analysis**

#### **View Recent Logs**
```powershell
# View last 50 lines of pipeline logs
Get-Content logs/pipeline.log -Tail 50

# Search for errors
Select-String -Path logs/pipeline.log -Pattern \"ERROR\"

# Monitor real-time logs
Get-Content logs/pipeline.log -Wait
```

---

## 📚 Quick Reference

### **Essential Commands**
| Task | Command | Purpose |
|------|---------|----------|
| **Generate Data** | `python generate_simple_raw_data.py` | Create test datasets |
| **Run Pipeline** | `./run_all.ps1 -InputFile [file]` | Complete pipeline execution |
| **Show Help** | `./run_all.ps1 -Help` | Show all options |
| **Test Integration** | `python validate_pipeline_integration.py` | Validate all components |
| **Test Dashboard** | `python test_dashboard.py` | Test dashboard components |
| **Launch Dashboard** | `python run_dashboard.py` | Dashboard only |
| **Health Check** | `./run_all.ps1 -HealthCheckOnly` | System health status |

### **File Locations**
| Type | Location | Description |
|------|----------|-------------|
| **Raw Data** | `data/raw/` | Input datasets |
| **Processed Data** | `data/output/` | Pipeline outputs |
| **Logs** | `logs/` | Execution and error logs |
| **Configuration** | `config/` | Pipeline configuration |
| **Source Code** | `src/` | Pipeline implementation |
| **Tests** | `tests/` | Unit and integration tests |

### **Data Quality Scores**
| Score Range | Status | Action Required |
|-------------|--------|------------------|
| **95-100%** | 🟢 Excellent | Production ready |
| **85-95%** | 🟡 Good | Minor issues to address |
| **70-85%** | 🟠 Fair | Moderate attention needed |
| **Below 70%** | 🔴 Poor | Immediate action required |

---

**Last Updated**: August 6, 2025  
**Version**: 2.0 (Consolidated)  
**Total Files**: 15 executable files, 1 comprehensive guide  
**Status**: Production ready with comprehensive testing
"