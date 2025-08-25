"""
Test Data Generation Example

This example demonstrates how to use the TestDataGenerator to create
realistic test datasets for the e-commerce data processing pipeline.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.test_data_generator import TestDataGenerator
from src.utils.config import PipelineConfig


def main():
    """Main function to demonstrate test data generation."""
    
    print("Test Data Generation Example")
    print("=" * 50)
    
    try:
        # Initialize the test data generator
        generator = TestDataGenerator(seed=42)
        
        # Generate clean sample data
        print("\n1. Generating clean sample data...")
        clean_data = generator.generate_clean_sample(
            size=1000,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        print(f"Generated {len(clean_data)} clean records")
        print("\nClean data sample:")
        print(clean_data.head())
        
        # Save clean data
        clean_path = generator.save_dataset(
            clean_data, 
            "test_clean_sample.csv",
            "data/temp"
        )
        print(f"Saved clean data to: {clean_path}")
        
        # Generate dirty sample data
        print("\n2. Generating dirty sample data...")
        dirty_data = generator.generate_dirty_sample(
            size=1000,
            start_date="2023-01-01",
            end_date="2023-12-31",
            dirty_ratio=0.4
        )
        print(f"Generated {len(dirty_data)} dirty records")
        print("\nDirty data sample (showing potential issues):")
        print(dirty_data.head(10))
        
        # Save dirty data
        dirty_path = generator.save_dataset(
            dirty_data,
            "test_dirty_sample.csv", 
            "data/temp"
        )
        print(f"Saved dirty data to: {dirty_path}")
        
        # Generate anomaly sample data
        print("\n3. Generating anomaly sample data...")
        anomaly_data = generator.generate_anomaly_sample(
            size=1000,
            start_date="2023-01-01", 
            end_date="2023-12-31",
            anomaly_ratio=0.1
        )
        print(f"Generated {len(anomaly_data)} records with anomalies")
        
        # Show potential anomalies (high revenue records)
        high_revenue = anomaly_data.nlargest(5, 'revenue')
        print("\nTop 5 high revenue records (potential anomalies):")
        print(high_revenue[['order_id', 'product_name', 'quantity', 'unit_price', 'discount_percent', 'revenue']])
        
        # Save anomaly data
        anomaly_path = generator.save_dataset(
            anomaly_data,
            "test_anomaly_sample.csv",
            "data/temp"
        )
        print(f"Saved anomaly data to: {anomaly_path}")
        
        # Generate configurable mixed dataset
        print("\n4. Generating configurable mixed dataset...")
        config = {
            'size': 2000,
            'type': 'mixed',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'dirty_ratio': 0.3,
            'anomaly_ratio': 0.05
        }
        
        mixed_data = generator.generate_configurable_dataset(config)
        print(f"Generated {len(mixed_data)} mixed records")
        
        # Save mixed data
        mixed_path = generator.save_dataset(
            mixed_data,
            "test_mixed_sample.csv",
            "data/temp"
        )
        print(f"Saved mixed data to: {mixed_path}")
        
        # Generate summary statistics
        print("\n5. Dataset summaries...")
        
        datasets = [
            ("Clean", clean_data),
            ("Dirty", dirty_data), 
            ("Anomaly", anomaly_data),
            ("Mixed", mixed_data)
        ]
        
        for name, data in datasets:
            summary = generator.get_generation_summary(data)
            print(f"\n{name} Dataset Summary:")
            print(f"  Total records: {summary['total_records']}")
            print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            print(f"  Categories: {summary['categories']}")
            print(f"  Revenue stats: Mean=${summary['revenue_stats']['mean']:.2f}, "
                  f"Total=${summary['revenue_stats']['total']:.2f}")
            print(f"  Data quality issues: {summary['data_quality']['duplicate_order_ids']} duplicate order IDs")
        
        # Demonstrate large dataset generation
        print("\n6. Generating large dataset for performance testing...")
        large_config = {
            'size': 100000,  # 100K records
            'type': 'clean',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        
        large_data = generator.generate_configurable_dataset(large_config)
        large_path = generator.save_dataset(
            large_data,
            "test_large_sample.csv",
            "data/temp"
        )
        print(f"Generated and saved {len(large_data)} records to: {large_path}")
        
        # Show memory usage
        memory_usage = large_data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Memory usage: {memory_usage:.2f} MB")
        
        print("\nTest data generation example completed successfully!")
        print("\nGenerated files:")
        print(f"  - {clean_path}")
        print(f"  - {dirty_path}")
        print(f"  - {anomaly_path}")
        print(f"  - {mixed_path}")
        print(f"  - {large_path}")
        
    except Exception as e:
        print(f"Error during test data generation: {e}")
        raise


if __name__ == "__main__":
    main()