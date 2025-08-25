# Data Output and Storage System

## Overview

The Data Output and Storage System provides comprehensive functionality for exporting processed data, managing data persistence, caching intermediate results, and tracking data lineage. This system is designed to handle large-scale data processing workflows with efficiency and reliability.

## Components

### 1. Data Export Functionality (`src/data/data_exporter.py`)

#### DataExporter
- **Purpose**: Handles exporting processed data in multiple formats with versioning and metadata
- **Supported Formats**: CSV, Parquet
- **Features**:
  - Automatic timestamp generation for file versioning
  - Metadata creation with data integrity information
  - Memory usage tracking
  - Configurable export parameters

#### DataVersionManager
- **Purpose**: Manages data versioning and tracks export history
- **Features**:
  - Version history tracking in JSON format
  - Version metadata storage
  - Latest version retrieval
  - Dataset-specific version filtering

#### ExportManager
- **Purpose**: High-level coordinator for data export operations
- **Features**:
  - Complete dataset export (processed data + analytical tables)
  - Integrated versioning
  - Export summary generation
  - Multi-format export coordination

### 2. Data Integrity Validation (`src/data/data_integrity.py`)

#### DataIntegrityValidator
- **Purpose**: Validates data integrity of exported files
- **Features**:
  - File checksum calculation (MD5)
  - Data consistency validation between original and exported data
  - Shape, column, and data type validation
  - Sample-based comparison for large datasets
  - Comprehensive validation reporting

### 3. Data Persistence and Caching (`src/data/data_persistence.py`)

#### DataCache
- **Purpose**: Manages caching of intermediate processing results
- **Features**:
  - Parquet-based storage for efficiency
  - Automatic cache cleanup (TTL and size-based)
  - Cache key generation based on data identifier and parameters
  - Cache statistics and monitoring
  - Configurable cache size limits and TTL

#### CheckpointManager
- **Purpose**: Manages checkpoints for long-running processes
- **Features**:
  - Process state serialization
  - Data checkpointing with Parquet storage
  - Checkpoint metadata tracking
  - Automatic cleanup of old checkpoints
  - Process-specific checkpoint filtering

#### DataLineageTracker
- **Purpose**: Tracks data lineage and transformation history
- **Features**:
  - Dataset registration and metadata storage
  - Transformation tracking with input/output relationships
  - Lineage query capabilities
  - Comprehensive lineage reporting
  - JSON-based lineage storage

#### PersistenceManager
- **Purpose**: High-level manager coordinating all persistence operations
- **Features**:
  - Unified interface for caching, checkpointing, and lineage
  - System status monitoring
  - Integrated workflow management

## Key Features

### 1. Multi-Format Export
- **CSV Export**: Human-readable format with configurable parameters
- **Parquet Export**: Efficient columnar storage with compression
- **Metadata Generation**: Automatic metadata creation for all exports
- **Timestamp Management**: Automatic versioning with timestamps

### 2. Data Integrity Assurance
- **Checksum Validation**: MD5 checksums for file integrity
- **Data Consistency Checks**: Validation of data shape, columns, and content
- **Flexible Validation**: Support for acceptable data type conversions
- **Comprehensive Reporting**: Detailed validation reports with issue identification

### 3. Intelligent Caching
- **Memory-Efficient Storage**: Parquet-based caching for optimal storage
- **Automatic Cleanup**: TTL-based and size-based cache management
- **Parameter-Based Keys**: Cache keys generated from data identifiers and processing parameters
- **Cache Statistics**: Monitoring and reporting of cache usage

### 4. Robust Checkpointing
- **Process State Management**: Serialization of complex process states
- **Data Checkpointing**: Storage of intermediate data results
- **Recovery Support**: Easy restoration from checkpoints
- **Automatic Cleanup**: Configurable retention policies

### 5. Data Lineage Tracking
- **Dataset Registration**: Comprehensive dataset metadata tracking
- **Transformation History**: Complete audit trail of data transformations
- **Relationship Mapping**: Input/output relationships between datasets
- **Lineage Queries**: Easy retrieval of dataset lineage information

## Configuration

The system uses the `PipelineConfig` class for configuration:

```yaml
# Output formats
output_formats:
  - csv
  - parquet

# File paths
output_data_path: data/output
processed_data_path: data/processed
temp_data_path: data/temp

# Cache settings (in PersistenceManager)
max_cache_size_gb: 2.0
cache_ttl_hours: 24
```

## Usage Examples

### Basic Data Export

```python
from src.utils.config import PipelineConfig
from src.data.data_exporter import ExportManager

config = PipelineConfig.from_file("config/pipeline_config.yaml")
export_manager = ExportManager(config)

# Export complete dataset
export_summary = export_manager.export_complete_dataset(
    processed_data=processed_df,
    analytical_tables=analytical_dict,
    dataset_name="ecommerce_data"
)
```

### Data Caching

```python
from src.data.data_persistence import PersistenceManager

persistence_manager = PersistenceManager(config)

# Cache intermediate result
cache_key = persistence_manager.cache_intermediate_result(
    data=intermediate_df,
    process_name="data_pipeline",
    step_name="cleaning",
    parameters={"remove_outliers": True, "threshold": 3.0}
)

# Retrieve cached result
cached_data = persistence_manager.get_cached_result(
    process_name="data_pipeline",
    step_name="cleaning",
    parameters={"remove_outliers": True, "threshold": 3.0}
)
```

### Checkpoint Management

```python
# Create checkpoint
checkpoint_id = persistence_manager.create_process_checkpoint(
    process_name="etl_pipeline",
    current_step="transformation",
    completed_steps=["ingestion", "cleaning"],
    data=current_data,
    additional_state={"batch_id": "batch_001"}
)

# Load checkpoint
state, data = persistence_manager.checkpoint_manager.load_checkpoint(checkpoint_id)
```

### Data Lineage Tracking

```python
# Register dataset
persistence_manager.lineage_tracker.register_dataset(
    "raw_sales_data",
    {"source": "sales_db", "records": 1000000, "date": "2023-01-01"}
)

# Track transformation
persistence_manager.lineage_tracker.track_transformation(
    "sales_cleaning_001",
    ["raw_sales_data"],
    ["cleaned_sales_data"],
    "data_cleaning",
    {"remove_duplicates": True, "validate_emails": True}
)
```

## File Structure

```
data/
├── output/
│   ├── csv/                    # CSV exports
│   ├── parquet/               # Parquet exports
│   ├── analytical/            # Analytical table exports
│   ├── metadata/              # Export metadata files
│   ├── version_history.json   # Version tracking
│   └── data_lineage.json      # Lineage information
├── temp/
│   ├── cache/                 # Cached intermediate results
│   │   ├── *.parquet         # Cached data files
│   │   └── cache_metadata.json
│   └── checkpoints/           # Process checkpoints
│       └── {process_id}_{timestamp}/
│           ├── state.json
│           ├── data.parquet
│           └── metadata.json
```

## Testing

Comprehensive test suite available in `tests/test_data_export_persistence.py`:

- Data export functionality tests
- Data integrity validation tests
- Caching mechanism tests
- Checkpoint management tests
- Data lineage tracking tests
- Integration tests

Run tests with:
```bash
python -m pytest tests/test_data_export_persistence.py -v
```

## Performance Considerations

1. **Parquet Format**: Used for efficient storage and fast I/O operations
2. **Compression**: Snappy compression for optimal balance of speed and size
3. **Memory Management**: Automatic cleanup of cache and checkpoints
4. **Chunked Processing**: Support for large datasets through chunked operations
5. **Metadata Optimization**: Lightweight JSON metadata for fast queries

## Error Handling

- Graceful handling of file I/O errors
- Validation of data integrity with detailed error reporting
- Automatic recovery mechanisms for cache and checkpoint operations
- Comprehensive logging for debugging and monitoring

## Requirements Satisfied

This implementation satisfies the following requirements:

- **6.4**: Save transformed data in both CSV and Parquet formats
- **6.5**: Include all source code, tests, and configuration files
- **6.4**: Implement data integrity validation for outputs
- **6.4**: Create intermediate result caching for large datasets
- **6.4**: Add checkpoint functionality for long-running processes
- **6.4**: Build data lineage tracking and documentation

The system provides a robust, scalable foundation for data output and storage operations in large-scale data processing pipelines.