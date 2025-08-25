"""
Data Persistence and Caching Example

This example demonstrates how to use the data persistence functionality
including caching, checkpoints, and data lineage tracking.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import PipelineConfig
from src.data.data_persistence import PersistenceManager


def simulate_data_processing_pipeline():
    """Simulate a multi-step data processing pipeline with caching and checkpoints."""
    
    print("Data Persistence and Caching Example")
    print("=" * 50)
    
    # Load configuration
    config = PipelineConfig.from_file("config/pipeline_config.yaml")
    
    # Initialize persistence manager
    persistence_manager = PersistenceManager(config)
    
    # Step 1: Create initial dataset
    print("\nStep 1: Creating initial dataset...")
    np.random.seed(42)
    n_records = 5000
    
    raw_data = pd.DataFrame({
        'id': range(n_records),
        'value': np.random.randn(n_records),
        'category': np.random.choice(['A', 'B', 'C'], n_records),
        'timestamp': pd.date_range('2023-01-01', periods=n_records, freq='h')
    })
    
    # Register dataset in lineage
    persistence_manager.lineage_tracker.register_dataset(
        "raw_data_v1",
        {
            "source": "simulated",
            "records": len(raw_data),
            "columns": list(raw_data.columns),
            "created_at": pd.Timestamp.now().isoformat()
        }
    )
    
    print(f"Created raw dataset with {len(raw_data)} records")
    
    # Step 2: Data cleaning (with caching)
    print("\nStep 2: Data cleaning...")
    
    # Check if cached result exists
    cleaning_params = {"remove_outliers": True, "threshold": 3.0}
    cached_cleaned_data = persistence_manager.get_cached_result(
        "data_pipeline", "cleaning", cleaning_params
    )
    
    if cached_cleaned_data is not None:
        print("Using cached cleaned data")
        cleaned_data = cached_cleaned_data
    else:
        print("Performing data cleaning...")
        # Simulate cleaning process
        time.sleep(1)  # Simulate processing time
        
        # Remove outliers
        z_scores = np.abs((raw_data['value'] - raw_data['value'].mean()) / raw_data['value'].std())
        cleaned_data = raw_data[z_scores < cleaning_params["threshold"]].copy()
        
        # Cache the result
        cache_key = persistence_manager.cache_intermediate_result(
            cleaned_data, "data_pipeline", "cleaning", cleaning_params
        )
        print(f"Cached cleaned data with key: {cache_key}")
    
    # Track transformation
    persistence_manager.lineage_tracker.track_transformation(
        "cleaning_transform_001",
        ["raw_data_v1"],
        ["cleaned_data_v1"],
        "outlier_removal",
        cleaning_params
    )
    
    # Register cleaned dataset
    persistence_manager.lineage_tracker.register_dataset(
        "cleaned_data_v1",
        {
            "source": "cleaning_transform",
            "records": len(cleaned_data),
            "removed_outliers": len(raw_data) - len(cleaned_data)
        }
    )
    
    print(f"Cleaned data: {len(cleaned_data)} records (removed {len(raw_data) - len(cleaned_data)} outliers)")
    
    # Create checkpoint after cleaning
    checkpoint_id = persistence_manager.create_process_checkpoint(
        "data_pipeline",
        "aggregation",
        ["cleaning"],
        cleaned_data,
        {"cleaning_params": cleaning_params}
    )
    print(f"Created checkpoint: {checkpoint_id}")
    
    # Step 3: Data aggregation (with caching)
    print("\nStep 3: Data aggregation...")
    
    aggregation_params = {"group_by": "category", "agg_func": "mean"}
    cached_aggregated_data = persistence_manager.get_cached_result(
        "data_pipeline", "aggregation", aggregation_params
    )
    
    if cached_aggregated_data is not None:
        print("Using cached aggregated data")
        aggregated_data = cached_aggregated_data
    else:
        print("Performing data aggregation...")
        time.sleep(0.5)  # Simulate processing time
        
        # Aggregate by category
        aggregated_data = cleaned_data.groupby('category').agg({
            'value': ['mean', 'std', 'count'],
            'id': 'count'
        }).round(4)
        
        # Flatten column names
        aggregated_data.columns = ['_'.join(col).strip() for col in aggregated_data.columns]
        aggregated_data = aggregated_data.reset_index()
        
        # Cache the result
        cache_key = persistence_manager.cache_intermediate_result(
            aggregated_data, "data_pipeline", "aggregation", aggregation_params
        )
        print(f"Cached aggregated data with key: {cache_key}")
    
    # Track transformation
    persistence_manager.lineage_tracker.track_transformation(
        "aggregation_transform_001",
        ["cleaned_data_v1"],
        ["aggregated_data_v1"],
        "group_aggregation",
        aggregation_params
    )
    
    # Register aggregated dataset
    persistence_manager.lineage_tracker.register_dataset(
        "aggregated_data_v1",
        {
            "source": "aggregation_transform",
            "records": len(aggregated_data),
            "aggregation_level": "category"
        }
    )
    
    print(f"Aggregated data: {len(aggregated_data)} records")
    print(aggregated_data)
    
    # Final checkpoint
    final_checkpoint_id = persistence_manager.create_process_checkpoint(
        "data_pipeline",
        "completed",
        ["cleaning", "aggregation"],
        aggregated_data,
        {
            "cleaning_params": cleaning_params,
            "aggregation_params": aggregation_params,
            "final_records": len(aggregated_data)
        }
    )
    print(f"\nCreated final checkpoint: {final_checkpoint_id}")
    
    return aggregated_data, persistence_manager


def demonstrate_lineage_tracking(persistence_manager):
    """Demonstrate data lineage tracking functionality."""
    
    print("\n" + "=" * 50)
    print("Data Lineage Tracking")
    print("=" * 50)
    
    # Get lineage for a specific dataset
    lineage = persistence_manager.lineage_tracker.get_dataset_lineage("cleaned_data_v1")
    
    print("\nLineage for 'cleaned_data_v1':")
    print(f"Dataset info: {lineage['dataset_info']}")
    print(f"Produced by: {len(lineage['produced_by'])} transformations")
    print(f"Consumed by: {len(lineage['consumed_by'])} transformations")
    
    # Generate full lineage report
    lineage_report = persistence_manager.lineage_tracker.generate_lineage_report()
    print(f"\nFull lineage report:")
    print(f"Total datasets: {lineage_report['total_datasets']}")
    print(f"Total transformations: {lineage_report['total_transformations']}")


def demonstrate_checkpoint_recovery(persistence_manager):
    """Demonstrate checkpoint recovery functionality."""
    
    print("\n" + "=" * 50)
    print("Checkpoint Recovery")
    print("=" * 50)
    
    # List available checkpoints
    checkpoints = persistence_manager.checkpoint_manager.list_checkpoints("data_pipeline")
    
    print(f"\nAvailable checkpoints for 'data_pipeline': {len(checkpoints)}")
    for checkpoint in checkpoints[:3]:  # Show first 3
        print(f"  {checkpoint['checkpoint_id']} - {checkpoint['created_at']}")
        print(f"    Current step: {checkpoint['state']['current_step']}")
        print(f"    Completed steps: {checkpoint['state']['completed_steps']}")
    
    if checkpoints:
        # Load the most recent checkpoint
        latest_checkpoint = checkpoints[0]
        print(f"\nLoading checkpoint: {latest_checkpoint['checkpoint_id']}")
        
        state, data = persistence_manager.checkpoint_manager.load_checkpoint(
            latest_checkpoint['checkpoint_id']
        )
        
        print(f"Recovered state: {state}")
        if data is not None:
            print(f"Recovered data shape: {data.shape}")


def demonstrate_cache_management(persistence_manager):
    """Demonstrate cache management functionality."""
    
    print("\n" + "=" * 50)
    print("Cache Management")
    print("=" * 50)
    
    # Get cache statistics
    cache_stats = persistence_manager.cache.get_cache_stats()
    
    print(f"Cache statistics:")
    print(f"  Total entries: {cache_stats['total_entries']}")
    print(f"  Total size: {cache_stats['total_size_mb']} MB")
    print(f"  Cache directory: {cache_stats['cache_directory']}")
    print(f"  Max size limit: {cache_stats['max_size_gb']} GB")
    print(f"  TTL: {cache_stats['ttl_hours']} hours")


def main():
    """Main function to run the persistence example."""
    
    try:
        # Run the simulated pipeline
        final_data, persistence_manager = simulate_data_processing_pipeline()
        
        # Demonstrate different features
        demonstrate_lineage_tracking(persistence_manager)
        demonstrate_checkpoint_recovery(persistence_manager)
        demonstrate_cache_management(persistence_manager)
        
        # Show system status
        print("\n" + "=" * 50)
        print("System Status")
        print("=" * 50)
        
        status = persistence_manager.get_system_status()
        print(f"System status at: {status['timestamp']}")
        print(f"Cache: {status['cache']['total_entries']} entries, {status['cache']['total_size_mb']} MB")
        print(f"Checkpoints: {status['checkpoints']['total_checkpoints']} total")
        print(f"Lineage: {status['lineage']['total_datasets']} datasets, {status['lineage']['total_transformations']} transformations")
        
        print("\nData persistence example completed successfully!")
        
    except Exception as e:
        print(f"Error during persistence example: {e}")
        raise


if __name__ == "__main__":
    main()