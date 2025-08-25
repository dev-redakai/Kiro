# Data Schema Documentation

This document provides comprehensive documentation of the data schemas used throughout the Scalable Data Engineering Pipeline.

## Table of Contents

1. [Input Data Schema](#input-data-schema)
2. [Processed Data Schema](#processed-data-schema)
3. [Analytical Tables Schema](#analytical-tables-schema)
4. [Data Quality Schema](#data-quality-schema)
5. [Configuration Schema](#configuration-schema)
6. [Error and Logging Schema](#error-and-logging-schema)

## Input Data Schema

### Raw E-commerce Sales Data

The pipeline expects CSV files with the following structure:

| Field Name | Data Type | Required | Description | Example Values | Validation Rules |
|------------|-----------|----------|-------------|----------------|------------------|
| `order_id` | String | Yes | Unique identifier for each order | ORD_000001, ORD_123456 | Must be unique, non-empty |
| `product_name` | String | Yes | Name of the product sold | Laptop, iPhone 13, T-Shirt | Non-empty, max 200 chars |
| `category` | String | Yes | Product category | Electronics, Fashion, Home Goods | Must map to standard categories |
| `quantity` | Integer | Yes | Number of items in the order | 1, 2, 5, 10 | Must be positive integer |
| `unit_price` | Float | Yes | Price per unit in local currency | 299.99, 1299.00, 25.50 | Must be positive number |
| `discount_percent` | Float | Yes | Discount as decimal (0.0-1.0) | 0.15, 0.25, 0.00 | Must be between 0.0 and 1.0 |
| `region` | String | Yes | Sales region | North, South, East, West | Must map to standard regions |
| `sale_date` | String | Yes | Date of sale | 2023-01-15, 15/01/2023 | Must be valid date format |
| `customer_email` | String | No | Customer email address | user@example.com | Must be valid email format if present |

### Data Quality Issues Handled

The pipeline handles common data quality issues in input data:

#### Product Name Variations
- **Case inconsistencies**: `Laptop`, `laptop`, `LAPTOP`
- **Extra whitespace**: `  Laptop  `, `Laptop   `
- **Special characters**: `Laptop-Pro`, `Laptop_Pro`
- **Abbreviations**: `Lap Top`, `Lap-top`

#### Category Variations
- **Standard**: Electronics, Fashion, Home Goods
- **Variations**: electronics, electronic, elec, tech, clothing, apparel, home, household

#### Region Variations
- **Standard**: North, South, East, West, Central
- **Variations**: north, nort, northern, n, NORTH

#### Date Format Variations
- **ISO Format**: 2023-01-15
- **European**: 15/01/2023, 15-01-2023, 15.01.2023
- **American**: 01/15/2023, 01-15-2023
- **With Time**: 2023-01-15 10:30:00, 2023-01-15T10:30:00Z

#### Common Data Issues
- **Missing values**: Empty strings, NULL values
- **Invalid data types**: String quantities, non-numeric prices
- **Out-of-range values**: Negative quantities, discounts > 1.0
- **Duplicate records**: Same order_id appearing multiple times
- **Invalid emails**: Malformed email addresses

## Processed Data Schema

### Cleaned Sales Data

After processing, the data conforms to this standardized schema:

| Field Name | Data Type | Description | Example Values | Transformations Applied |
|------------|-----------|-------------|----------------|------------------------|
| `order_id` | String | Standardized order identifier | ORD_000001 | Duplicate detection and flagging |
| `product_name` | String | Cleaned product name | Laptop | Case standardization, whitespace removal |
| `category` | String | Standardized category | Electronics | Mapped to standard categories |
| `quantity` | Integer | Validated quantity | 2 | Converted to integer, invalid values flagged |
| `unit_price` | Float | Validated unit price | 299.99 | Converted to float, negative values flagged |
| `discount_percent` | Float | Validated discount percentage | 0.15 | Range validation (0.0-1.0) |
| `region` | String | Standardized region | North | Mapped to standard regions |
| `sale_date` | DateTime | Parsed date | 2023-01-15 00:00:00 | Converted to datetime object |
| `customer_email` | String | Validated email | user@example.com | Email format validation |
| `revenue` | Float | Calculated revenue | 509.83 | quantity × unit_price × (1 - discount_percent) |

### Data Quality Flags

Additional columns added during processing:

| Field Name | Data Type | Description | Values |
|------------|-----------|-------------|---------|
| `data_quality_score` | Float | Overall quality score (0-100) | 95.5 |
| `has_quality_issues` | Boolean | Indicates if record has issues | True/False |
| `quality_issues` | String | List of quality issues found | "invalid_email,duplicate_id" |
| `cleaning_applied` | String | List of cleaning operations | "standardize_category,parse_date" |
| `is_anomaly` | Boolean | Anomaly detection flag | True/False |
| `anomaly_score` | Float | Anomaly score (higher = more suspicious) | 2.5 |
| `anomaly_reason` | String | Reason for anomaly classification | "extreme_revenue" |

## Analytical Tables Schema

### Monthly Sales Summary

Aggregated monthly sales performance data:

| Field Name | Data Type | Description | Example Values |
|------------|-----------|-------------|----------------|
| `month` | String | Month period (YYYY-MM) | 2023-01 |
| `total_revenue` | Float | Sum of revenue for the month | 125000.50 |
| `total_quantity` | Integer | Sum of quantities sold | 2500 |
| `avg_discount` | Float | Average discount percentage | 0.18 |
| `order_count` | Integer | Number of orders in the month | 1250 |
| `avg_order_value` | Float | Average revenue per order | 100.00 |
| `unique_customers` | Integer | Number of unique customers | 980 |
| `top_category` | String | Most popular category | Electronics |

### Top Products Analysis

Top-performing products by various metrics:

| Field Name | Data Type | Description | Example Values |
|------------|-----------|-------------|----------------|
| `product_name` | String | Product name | Laptop |
| `category` | String | Product category | Electronics |
| `total_revenue` | Float | Total revenue generated | 50000.00 |
| `total_units` | Integer | Total units sold | 125 |
| `avg_price` | Float | Average unit price | 400.00 |
| `avg_discount` | Float | Average discount applied | 0.12 |
| `rank_by_revenue` | Integer | Revenue ranking (1 = highest) | 1 |
| `rank_by_units` | Integer | Units sold ranking | 3 |
| `profit_margin` | Float | Estimated profit margin | 0.25 |

### Regional Performance

Sales performance by geographic region:

| Field Name | Data Type | Description | Example Values |
|------------|-----------|-------------|----------------|
| `region` | String | Region name | North |
| `total_revenue` | Float | Total revenue for region | 75000.00 |
| `total_orders` | Integer | Number of orders | 1500 |
| `avg_order_value` | Float | Average order value | 50.00 |
| `top_category` | String | Most popular category | Fashion |
| `customer_count` | Integer | Number of unique customers | 1200 |
| `avg_discount` | Float | Average discount in region | 0.15 |
| `market_share` | Float | Percentage of total sales | 0.25 |

### Category Discount Analysis

Discount effectiveness by product category:

| Field Name | Data Type | Description | Example Values |
|------------|-----------|-------------|----------------|
| `category` | String | Product category | Electronics |
| `avg_discount` | Float | Average discount percentage | 0.18 |
| `discount_frequency` | Float | Percentage of orders with discounts | 0.65 |
| `revenue_with_discount` | Float | Revenue from discounted orders | 45000.00 |
| `revenue_without_discount` | Float | Revenue from full-price orders | 25000.00 |
| `discount_impact` | Float | Revenue impact of discounts | -8000.00 |
| `order_count_discounted` | Integer | Number of discounted orders | 800 |
| `order_count_full_price` | Integer | Number of full-price orders | 450 |

### Anomaly Records

Records flagged as potentially suspicious:

| Field Name | Data Type | Description | Example Values |
|------------|-----------|-------------|----------------|
| `order_id` | String | Order identifier | ORD_123456 |
| `anomaly_type` | String | Type of anomaly detected | extreme_revenue |
| `anomaly_score` | Float | Anomaly score (higher = more suspicious) | 4.2 |
| `revenue` | Float | Order revenue | 25000.00 |
| `quantity` | Integer | Order quantity | 100 |
| `unit_price` | Float | Unit price | 250.00 |
| `discount_percent` | Float | Discount applied | 0.95 |
| `reason` | String | Detailed reason for flagging | "Revenue 4.2 std devs above mean" |
| `detected_at` | DateTime | When anomaly was detected | 2023-01-15 10:30:00 |
| `confidence` | Float | Confidence in anomaly detection | 0.95 |

## Data Quality Schema

### Data Quality Report

Comprehensive data quality assessment:

```json
{
  "report_id": "DQ_20230115_103000",
  "generated_at": "2023-01-15T10:30:00Z",
  "dataset_info": {
    "total_records": 1000000,
    "date_range": {
      "start": "2023-01-01",
      "end": "2023-12-31"
    },
    "file_size_mb": 250.5
  },
  "quality_metrics": {
    "overall_score": 87.5,
    "completeness": 92.3,
    "validity": 89.1,
    "consistency": 85.7,
    "accuracy": 88.9
  },
  "field_quality": {
    "order_id": {
      "completeness": 100.0,
      "uniqueness": 99.8,
      "validity": 100.0,
      "issues": ["2000 duplicate IDs"]
    },
    "product_name": {
      "completeness": 98.5,
      "validity": 95.2,
      "standardization_applied": 15000,
      "issues": ["1500 missing values", "4800 non-standard formats"]
    }
  },
  "cleaning_summary": {
    "records_cleaned": 150000,
    "cleaning_operations": {
      "standardize_categories": 25000,
      "parse_dates": 18000,
      "validate_emails": 12000,
      "calculate_revenue": 1000000
    }
  },
  "anomalies": {
    "total_detected": 500,
    "by_type": {
      "extreme_revenue": 200,
      "unusual_quantity": 150,
      "suspicious_discount": 100,
      "price_manipulation": 50
    }
  }
}
```

### Validation Rules Schema

Configuration for data validation rules:

```yaml
validation_rules:
  order_id:
    required: true
    unique: true
    pattern: "^ORD_[0-9]{6}$"
    
  product_name:
    required: true
    max_length: 200
    min_length: 2
    
  category:
    required: true
    allowed_values: ["Electronics", "Fashion", "Home Goods", "Beauty", "Sports", "Books"]
    
  quantity:
    required: true
    type: integer
    min_value: 1
    max_value: 10000
    
  unit_price:
    required: true
    type: float
    min_value: 0.01
    max_value: 100000.00
    
  discount_percent:
    required: true
    type: float
    min_value: 0.0
    max_value: 1.0
    
  region:
    required: true
    allowed_values: ["North", "South", "East", "West", "Central", "Southeast", "Northeast", "Northwest", "Southwest"]
    
  sale_date:
    required: true
    type: datetime
    min_date: "2020-01-01"
    max_date: "2030-12-31"
    
  customer_email:
    required: false
    type: email
    pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
```

## Configuration Schema

### Pipeline Configuration

Main configuration file structure (`config/pipeline_config.yaml`):

```yaml
# Data Processing Settings
data_processing:
  chunk_size: 50000                    # Integer: Records per chunk
  max_memory_usage: 4294967296         # Integer: Max memory in bytes
  parallel_processing: true            # Boolean: Enable parallel processing
  enable_caching: true                 # Boolean: Enable result caching
  
# File Paths Configuration
paths:
  raw_data: "data/raw"                 # String: Raw data directory
  processed_data: "data/processed"     # String: Processed data directory
  output_data: "data/output"           # String: Output directory
  temp_data: "data/temp"               # String: Temporary files directory
  logs: "logs"                         # String: Log files directory
  config: "config"                     # String: Configuration directory
  
# Output Settings
output:
  formats: ["csv", "parquet"]          # List: Output formats
  include_metadata: true               # Boolean: Include processing metadata
  compression: "gzip"                  # String: Compression method
  create_analytical_tables: true      # Boolean: Generate analytical tables
  
# Data Quality Settings
data_quality:
  anomaly_threshold: 3.0               # Float: Z-score threshold
  iqr_multiplier: 1.5                  # Float: IQR multiplier for outliers
  duplicate_handling: "flag"           # String: flag|remove|keep_first
  null_value_strategy: "impute"        # String: impute|remove|flag
  enable_validation: true              # Boolean: Enable data validation
  
# Cleaning Rules
cleaning_rules:
  standardize_categories: true         # Boolean: Apply category standardization
  standardize_regions: true            # Boolean: Apply region standardization
  parse_dates: true                    # Boolean: Parse date fields
  validate_emails: true                # Boolean: Validate email formats
  calculate_revenue: true              # Boolean: Calculate revenue field
  
# Anomaly Detection Settings
anomaly_detection:
  enabled: true                        # Boolean: Enable anomaly detection
  methods: ["zscore", "iqr"]           # List: Detection methods
  sensitivity: "medium"                # String: low|medium|high
  fields_to_analyze: ["revenue", "quantity", "discount_percent"]
  
# Dashboard Settings
dashboard:
  enabled: true                        # Boolean: Enable dashboard
  port: 8501                          # Integer: Dashboard port
  auto_refresh: true                   # Boolean: Auto-refresh data
  refresh_interval: 300                # Integer: Refresh interval in seconds
  
# Logging Configuration
logging:
  level: "INFO"                        # String: Log level
  console_output: true                 # Boolean: Enable console logging
  file_output: true                    # Boolean: Enable file logging
  structured_logging: true             # Boolean: Use structured logging
  rotation:
    enabled: true                      # Boolean: Enable log rotation
    max_size: "10MB"                   # String: Max log file size
    backup_count: 5                    # Integer: Number of backup files
    
# Performance Settings
performance:
  enable_profiling: false              # Boolean: Enable performance profiling
  memory_monitoring: true              # Boolean: Monitor memory usage
  progress_reporting: true             # Boolean: Report processing progress
  checkpoint_interval: 100000          # Integer: Records between checkpoints
```

## Error and Logging Schema

### Error Log Entry

Structure of error log entries:

```json
{
  "timestamp": "2023-01-15T10:30:00.123Z",
  "level": "ERROR",
  "logger": "data_cleaner",
  "message": "Failed to parse date field",
  "error_details": {
    "error_type": "ValueError",
    "error_message": "time data '2023/13/01' does not match any expected format",
    "stack_trace": "Traceback (most recent call last)...",
    "context": {
      "chunk_id": 123,
      "record_index": 45678,
      "field_name": "sale_date",
      "field_value": "2023/13/01"
    }
  },
  "recovery_action": "Set field to null and flag for manual review",
  "impact": "Single record affected, processing continues"
}
```

### Processing Log Entry

Structure of processing log entries:

```json
{
  "timestamp": "2023-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "pipeline.ingestion",
  "message": "Completed chunk processing",
  "metrics": {
    "chunk_id": 123,
    "records_processed": 50000,
    "processing_time_seconds": 45.2,
    "memory_usage_mb": 256.7,
    "records_per_second": 1106,
    "errors_encountered": 12,
    "warnings_generated": 3
  },
  "progress": {
    "total_chunks": 200,
    "completed_chunks": 123,
    "percentage_complete": 61.5,
    "estimated_time_remaining": "00:58:30"
  }
}
```

### Data Quality Log Entry

Structure of data quality log entries:

```json
{
  "timestamp": "2023-01-15T10:30:00.123Z",
  "level": "WARNING",
  "logger": "quality.validator",
  "message": "Data quality issues detected",
  "quality_metrics": {
    "chunk_id": 123,
    "total_records": 50000,
    "quality_score": 87.5,
    "issues_by_type": {
      "missing_values": 150,
      "invalid_format": 75,
      "out_of_range": 25,
      "duplicates": 10
    },
    "fields_affected": {
      "customer_email": 150,
      "sale_date": 75,
      "discount_percent": 25,
      "order_id": 10
    }
  },
  "recommendations": [
    "Review email validation rules",
    "Add additional date format support",
    "Investigate discount calculation logic"
  ]
}
```

## Schema Evolution and Versioning

### Version Information

All schemas include version information for compatibility tracking:

```json
{
  "schema_version": "1.0.0",
  "created_at": "2023-01-15T10:30:00Z",
  "last_modified": "2023-01-15T10:30:00Z",
  "compatibility": {
    "min_pipeline_version": "1.0.0",
    "max_pipeline_version": "1.x.x"
  }
}
```

### Backward Compatibility

The pipeline maintains backward compatibility for:
- Input data formats (with automatic detection)
- Configuration files (with default value fallbacks)
- Output formats (with version-specific serialization)

### Migration Support

When schemas change, the pipeline provides:
- Automatic migration utilities
- Validation of legacy formats
- Clear migration documentation
- Rollback capabilities

This comprehensive schema documentation ensures consistent data handling throughout the pipeline and provides clear specifications for integration and development.