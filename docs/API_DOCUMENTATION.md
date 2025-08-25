# API Documentation

This document provides detailed API documentation for all major classes and functions in the Scalable Data Engineering Pipeline.

## Table of Contents

1. [Data Ingestion](#data-ingestion)
2. [Data Cleaning](#data-cleaning)
3. [Data Transformation](#data-transformation)
4. [Quality Assurance](#quality-assurance)
5. [Dashboard Components](#dashboard-components)
6. [Utilities](#utilities)

## Data Ingestion

### ChunkedCSVReader

Handles memory-efficient CSV reading with configurable chunk sizes.

```python
class ChunkedCSVReader:
    def __init__(self, file_path: str, chunk_size: int = 50000, encoding: str = 'utf-8')
```

**Parameters:**
- `file_path` (str): Path to the CSV file to read
- `chunk_size` (int): Number of records per chunk (default: 50,000)
- `encoding` (str): File encoding (default: 'utf-8')

**Methods:**

#### `read_chunks() -> Iterator[pd.DataFrame]`
Returns an iterator of DataFrame chunks.

**Returns:**
- Iterator yielding pandas DataFrames

**Example:**
```python
reader = ChunkedCSVReader('data/raw/sales.csv', chunk_size=25000)
for chunk in reader.read_chunks():
    print(f"Processing {len(chunk)} records")
```

#### `get_total_rows() -> int`
Estimates the total number of rows in the file.

**Returns:**
- int: Estimated total row count

#### `estimate_memory_usage() -> int`
Estimates memory usage for the current chunk size.

**Returns:**
- int: Estimated memory usage in bytes

### DataIngestionManager

Orchestrates the data ingestion process with validation and monitoring.

```python
class DataIngestionManager:
    def __init__(self, config: Optional[PipelineConfig] = None)
```

**Parameters:**
- `config` (PipelineConfig, optional): Pipeline configuration object

**Methods:**

#### `ingest_data(file_path: str) -> Iterator[pd.DataFrame]`
Ingests data from a CSV file with validation and progress tracking.

**Parameters:**
- `file_path` (str): Path to the input CSV file

**Returns:**
- Iterator[pd.DataFrame]: Iterator of validated data chunks

**Raises:**
- `FileNotFoundError`: If the input file doesn't exist
- `ValueError`: If the file format is invalid

**Example:**
```python
manager = DataIngestionManager()
for chunk in manager.ingest_data('data/raw/sales.csv'):
    # Process validated chunk
    processed_chunk = process_chunk(chunk)
```

#### `validate_file_format(file_path: str) -> bool`
Validates the input file format and structure.

**Parameters:**
- `file_path` (str): Path to the file to validate

**Returns:**
- bool: True if file format is valid

#### `get_ingestion_stats() -> Dict[str, Any]`
Returns ingestion statistics and metrics.

**Returns:**
- Dict containing:
  - `total_records`: Total number of records processed
  - `processing_time`: Total processing time in seconds
  - `average_chunk_size`: Average chunk size
  - `memory_usage`: Peak memory usage during ingestion

## Data Cleaning

### DataCleaner

Main orchestrator for data cleaning operations.

```python
class DataCleaner:
    def __init__(self, config: Optional[PipelineConfig] = None)
```

**Methods:**

#### `clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame`
Applies all cleaning rules to a data chunk.

**Parameters:**
- `chunk` (pd.DataFrame): Raw data chunk to clean

**Returns:**
- pd.DataFrame: Cleaned data chunk with standardized fields

**Example:**
```python
cleaner = DataCleaner()
cleaned_chunk = cleaner.clean_chunk(raw_chunk)
```

#### `apply_cleaning_rules(chunk: pd.DataFrame) -> pd.DataFrame`
Applies specific cleaning rules based on configuration.

**Parameters:**
- `chunk` (pd.DataFrame): Data chunk to process

**Returns:**
- pd.DataFrame: Chunk with cleaning rules applied

#### `generate_cleaning_report() -> Dict[str, Any]`
Generates a comprehensive cleaning report.

**Returns:**
- Dict containing:
  - `total_records_processed`: Number of records processed
  - `cleaning_errors`: List of cleaning errors encountered
  - `field_statistics`: Statistics for each field
  - `data_quality_score`: Overall data quality score (0-100)

### FieldStandardizer

Handles field-specific cleaning and standardization.

```python
class FieldStandardizer:
    def __init__(self)
```

**Methods:**

#### `standardize_product_names(names: pd.Series) -> pd.Series`
Cleans and standardizes product names.

**Parameters:**
- `names` (pd.Series): Series of product names to standardize

**Returns:**
- pd.Series: Standardized product names

**Cleaning Rules:**
- Removes extra whitespace
- Standardizes capitalization
- Removes special characters
- Handles common abbreviations

#### `standardize_categories(categories: pd.Series) -> pd.Series`
Standardizes product categories using predefined mappings.

**Parameters:**
- `categories` (pd.Series): Series of categories to standardize

**Returns:**
- pd.Series: Standardized categories

**Standard Categories:**
- Electronics
- Fashion
- Home Goods
- Beauty
- Sports
- Books

#### `standardize_regions(regions: pd.Series) -> pd.Series`
Standardizes region names.

**Parameters:**
- `regions` (pd.Series): Series of region names to standardize

**Returns:**
- pd.Series: Standardized region names

**Standard Regions:**
- North
- South
- East
- West
- Central
- Southeast
- Northeast
- Northwest
- Southwest

#### `parse_dates(dates: pd.Series) -> pd.Series`
Parses dates in multiple formats.

**Parameters:**
- `dates` (pd.Series): Series of date strings to parse

**Returns:**
- pd.Series: Parsed datetime objects

**Supported Formats:**
- YYYY-MM-DD
- DD/MM/YYYY
- MM/DD/YYYY
- DD-MM-YYYY
- MM-DD-YYYY
- YYYY/MM/DD
- DD.MM.YYYY
- ISO format with time

#### `validate_quantities(quantities: pd.Series) -> pd.Series`
Validates and cleans quantity values.

**Parameters:**
- `quantities` (pd.Series): Series of quantity values

**Returns:**
- pd.Series: Validated quantities (invalid values set to NaN)

**Validation Rules:**
- Must be positive integers
- Converts string numbers to integers
- Flags zero and negative values

## Data Transformation

### AnalyticalTransformer

Creates business intelligence tables from cleaned data.

```python
class AnalyticalTransformer:
    def __init__(self, config: Optional[PipelineConfig] = None)
```

**Methods:**

#### `create_monthly_sales_summary(data: pd.DataFrame) -> pd.DataFrame`
Creates monthly sales summary table.

**Parameters:**
- `data` (pd.DataFrame): Cleaned sales data

**Returns:**
- pd.DataFrame with columns:
  - `month`: Month period
  - `total_revenue`: Sum of revenue for the month
  - `total_quantity`: Sum of quantities sold
  - `avg_discount`: Average discount percentage
  - `order_count`: Number of orders

#### `create_top_products_table(data: pd.DataFrame) -> pd.DataFrame`
Creates top products analysis table.

**Parameters:**
- `data` (pd.DataFrame): Cleaned sales data

**Returns:**
- pd.DataFrame with columns:
  - `product_name`: Product name
  - `category`: Product category
  - `total_revenue`: Total revenue generated
  - `total_units`: Total units sold
  - `avg_price`: Average unit price
  - `rank_by_revenue`: Revenue ranking
  - `rank_by_units`: Units sold ranking

#### `create_region_performance_table(data: pd.DataFrame) -> pd.DataFrame`
Creates regional performance analysis.

**Parameters:**
- `data` (pd.DataFrame): Cleaned sales data

**Returns:**
- pd.DataFrame with columns:
  - `region`: Region name
  - `total_revenue`: Total revenue for region
  - `total_orders`: Number of orders
  - `avg_order_value`: Average order value
  - `top_category`: Most popular category in region

#### `create_category_discount_map(data: pd.DataFrame) -> pd.DataFrame`
Creates category discount effectiveness analysis.

**Parameters:**
- `data` (pd.DataFrame): Cleaned sales data

**Returns:**
- pd.DataFrame with columns:
  - `category`: Product category
  - `avg_discount`: Average discount percentage
  - `discount_impact`: Revenue impact of discounts
  - `order_count`: Number of discounted orders

### MetricsCalculator

Computes business metrics and KPIs.

```python
class MetricsCalculator:
    def __init__(self)
```

**Methods:**

#### `calculate_revenue_metrics(data: pd.DataFrame) -> Dict[str, float]`
Calculates comprehensive revenue metrics.

**Parameters:**
- `data` (pd.DataFrame): Sales data

**Returns:**
- Dict containing:
  - `total_revenue`: Total revenue
  - `average_revenue`: Average revenue per order
  - `median_revenue`: Median revenue per order
  - `revenue_growth_rate`: Month-over-month growth rate

#### `calculate_discount_effectiveness(data: pd.DataFrame) -> Dict[str, float]`
Analyzes discount effectiveness.

**Parameters:**
- `data` (pd.DataFrame): Sales data

**Returns:**
- Dict containing:
  - `average_discount`: Overall average discount
  - `discount_penetration`: Percentage of orders with discounts
  - `revenue_impact`: Revenue impact of discounts

## Quality Assurance

### AnomalyDetector

Statistical anomaly detection for suspicious transactions.

```python
class AnomalyDetector:
    def __init__(self, z_score_threshold: float = 3.0, iqr_multiplier: float = 1.5)
```

**Parameters:**
- `z_score_threshold` (float): Z-score threshold for anomaly detection
- `iqr_multiplier` (float): IQR multiplier for outlier detection

**Methods:**

#### `detect_revenue_anomalies(data: pd.DataFrame) -> pd.DataFrame`
Detects revenue-based anomalies.

**Parameters:**
- `data` (pd.DataFrame): Sales data with revenue column

**Returns:**
- pd.DataFrame: Original data with anomaly flags and scores

**Anomaly Columns Added:**
- `revenue_anomaly_zscore`: Boolean flag for Z-score anomalies
- `revenue_anomaly_iqr`: Boolean flag for IQR anomalies
- `revenue_anomaly_score`: Anomaly score (higher = more anomalous)
- `revenue_anomaly_reason`: Reason for anomaly classification

#### `detect_quantity_anomalies(data: pd.DataFrame) -> pd.DataFrame`
Detects unusual quantity patterns.

**Parameters:**
- `data` (pd.DataFrame): Sales data with quantity column

**Returns:**
- pd.DataFrame: Data with quantity anomaly flags

#### `generate_anomaly_report() -> Dict[str, Any]`
Generates detailed anomaly detection report.

**Returns:**
- Dict containing:
  - `total_records_analyzed`: Number of records analyzed
  - `anomalies_detected`: Number of anomalies found
  - `anomaly_rate`: Percentage of anomalous records
  - `top_anomalies`: List of most anomalous records

### ValidationEngine

Comprehensive data validation system.

```python
class ValidationEngine:
    def __init__(self, config: Optional[PipelineConfig] = None)
```

**Methods:**

#### `validate_data_chunk(chunk: pd.DataFrame) -> Dict[str, Any]`
Validates a data chunk against business rules.

**Parameters:**
- `chunk` (pd.DataFrame): Data chunk to validate

**Returns:**
- Dict containing validation results:
  - `is_valid`: Boolean indicating overall validity
  - `validation_errors`: List of validation errors
  - `field_validations`: Per-field validation results

## Dashboard Components

### DashboardApp

Main Streamlit dashboard application.

```python
class DashboardApp:
    def __init__(self, config: Optional[PipelineConfig] = None)
```

**Methods:**

#### `create_revenue_trend_chart() -> plotly.graph_objects.Figure`
Creates monthly revenue trend visualization.

**Returns:**
- plotly.graph_objects.Figure: Interactive line chart

#### `create_top_products_chart() -> plotly.graph_objects.Figure`
Creates top products bar chart.

**Returns:**
- plotly.graph_objects.Figure: Interactive bar chart

#### `create_regional_sales_map() -> plotly.graph_objects.Figure`
Creates regional sales visualization.

**Returns:**
- plotly.graph_objects.Figure: Geographic or bar chart visualization

#### `display_anomaly_table() -> pd.DataFrame`
Displays anomaly records in tabular format.

**Returns:**
- pd.DataFrame: Formatted anomaly records table

### DataProvider

Supplies data to dashboard components.

```python
class DataProvider:
    def __init__(self, data_path: str)
```

**Methods:**

#### `load_analytical_tables() -> Dict[str, pd.DataFrame]`
Loads all analytical tables for dashboard display.

**Returns:**
- Dict mapping table names to DataFrames

#### `get_dashboard_metrics() -> Dict[str, Any]`
Retrieves key metrics for dashboard display.

**Returns:**
- Dict containing dashboard metrics and KPIs

## Utilities

### TestDataGenerator

Generates realistic test datasets for pipeline testing.

```python
class TestDataGenerator:
    def __init__(self, seed: Optional[int] = 42)
```

**Parameters:**
- `seed` (int, optional): Random seed for reproducible generation

**Methods:**

#### `generate_clean_sample(size: int, start_date: str = "2023-01-01", end_date: str = "2023-12-31") -> pd.DataFrame`
Generates clean e-commerce data with realistic patterns.

**Parameters:**
- `size` (int): Number of records to generate
- `start_date` (str): Start date for sales data (YYYY-MM-DD)
- `end_date` (str): End date for sales data (YYYY-MM-DD)

**Returns:**
- pd.DataFrame: Clean sample data

#### `generate_dirty_sample(size: int, dirty_ratio: float = 0.3) -> pd.DataFrame`
Generates data with common quality issues.

**Parameters:**
- `size` (int): Number of records to generate
- `dirty_ratio` (float): Proportion of records with quality issues

**Returns:**
- pd.DataFrame: Sample data with quality issues

**Quality Issues Introduced:**
- Invalid data types
- Missing values
- Inconsistent formatting
- Duplicate records
- Out-of-range values

#### `generate_anomaly_sample(size: int, anomaly_ratio: float = 0.05) -> pd.DataFrame`
Generates data with known suspicious patterns.

**Parameters:**
- `size` (int): Number of records to generate
- `anomaly_ratio` (float): Proportion of anomalous records

**Returns:**
- pd.DataFrame: Sample data with anomalies

**Anomaly Types:**
- Extreme revenue values
- Unusual quantity patterns
- Suspicious discount combinations
- Price manipulation indicators

### MemoryManager

Optimizes memory usage during processing.

```python
class MemoryManager:
    def __init__(self, max_memory_gb: float = 4.0)
```

**Parameters:**
- `max_memory_gb` (float): Maximum memory usage in GB

**Methods:**

#### `get_optimal_chunk_size(file_size: int) -> int`
Calculates optimal chunk size based on available memory.

**Parameters:**
- `file_size` (int): Size of input file in bytes

**Returns:**
- int: Recommended chunk size

#### `monitor_memory_usage() -> Dict[str, float]`
Monitors current memory usage.

**Returns:**
- Dict containing:
  - `current_usage_gb`: Current memory usage in GB
  - `available_gb`: Available memory in GB
  - `usage_percentage`: Memory usage percentage

### ErrorHandler

Comprehensive error handling and recovery system.

```python
class ErrorHandler:
    def __init__(self, config: Optional[PipelineConfig] = None)
```

**Methods:**

#### `handle_data_quality_error(error: DataQualityError) -> None`
Handles data quality errors with appropriate recovery strategies.

**Parameters:**
- `error` (DataQualityError): Data quality error to handle

#### `handle_memory_error(error: MemoryError) -> None`
Handles memory errors by reducing processing load.

**Parameters:**
- `error` (MemoryError): Memory error to handle

#### `handle_processing_error(error: ProcessingError) -> None`
Handles general processing errors with graceful continuation.

**Parameters:**
- `error` (ProcessingError): Processing error to handle

## Error Handling

All API methods include comprehensive error handling:

### Common Exceptions

- `DataQualityError`: Raised when data quality issues are detected
- `ProcessingError`: Raised during data processing failures
- `ConfigurationError`: Raised for configuration-related issues
- `MemoryError`: Raised when memory limits are exceeded
- `ValidationError`: Raised when data validation fails

### Error Response Format

All methods that can fail return error information in a consistent format:

```python
{
    "success": False,
    "error_type": "DataQualityError",
    "error_message": "Detailed error description",
    "error_context": {
        "chunk_id": 123,
        "record_count": 50000,
        "field_name": "quantity"
    },
    "suggested_action": "Reduce chunk size or check data quality"
}
```

## Performance Considerations

### Memory Usage

- All methods are designed to work with configurable memory limits
- Large datasets are processed in chunks to prevent memory overflow
- Garbage collection is performed at regular intervals

### Processing Speed

- Methods are optimized for vectorized operations using pandas
- Parallel processing is used where thread-safe
- Caching is implemented for frequently accessed data

### Scalability

- The API is designed to handle datasets from thousands to millions of records
- Processing time scales linearly with dataset size
- Memory usage remains constant regardless of dataset size (due to chunking)

## Version Compatibility

- **Python**: 3.8+
- **Pandas**: 1.3+
- **NumPy**: 1.20+
- **Streamlit**: 1.0+

For complete dependency information, see `requirements.txt`.