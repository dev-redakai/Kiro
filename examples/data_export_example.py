"""
Data Export Example

This example demonstrates how to use the data export functionality
to save processed data and analytical tables in multiple formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import PipelineConfig
from src.data.data_exporter import ExportManager
from src.data.data_integrity import DataIntegrityValidator


def create_sample_data():
    """Create sample processed data and analytical tables."""
    
    # Create sample processed data
    np.random.seed(42)
    n_records = 1000
    
    processed_data = pd.DataFrame({
        'order_id': [f'ORD_{i:06d}' for i in range(n_records)],
        'product_name': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones'], n_records),
        'category': np.random.choice(['Electronics', 'Fashion', 'Home & Garden'], n_records),
        'quantity': np.random.randint(1, 10, n_records),
        'unit_price': np.random.uniform(10, 1000, n_records),
        'discount_percent': np.random.uniform(0, 0.3, n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'sale_date': pd.date_range('2023-01-01', periods=n_records, freq='H'),
        'customer_email': [f'customer_{i}@example.com' for i in range(n_records)]
    })
    
    # Calculate revenue
    processed_data['revenue'] = (processed_data['quantity'] * 
                                processed_data['unit_price'] * 
                                (1 - processed_data['discount_percent']))
    
    # Create sample analytical tables
    analytical_tables = {}
    
    # Monthly sales summary
    processed_data['month'] = processed_data['sale_date'].dt.to_period('M')
    monthly_summary = processed_data.groupby('month').agg({
        'revenue': 'sum',
        'quantity': 'sum',
        'discount_percent': 'mean',
        'order_id': 'count'
    }).rename(columns={'order_id': 'order_count'}).reset_index()
    analytical_tables['monthly_sales_summary'] = monthly_summary
    
    # Top products
    top_products = processed_data.groupby(['product_name', 'category']).agg({
        'revenue': 'sum',
        'quantity': 'sum'
    }).reset_index().sort_values('revenue', ascending=False).head(10)
    analytical_tables['top_products'] = top_products
    
    # Region performance
    region_performance = processed_data.groupby('region').agg({
        'revenue': 'sum',
        'order_id': 'count',
        'unit_price': 'mean'
    }).rename(columns={'order_id': 'order_count', 'unit_price': 'avg_price'}).reset_index()
    analytical_tables['region_wise_performance'] = region_performance
    
    return processed_data, analytical_tables


def main():
    """Main function to demonstrate data export functionality."""
    
    print("Data Export Example")
    print("=" * 50)
    
    try:
        # Load configuration
        config = PipelineConfig.from_file("config/pipeline_config.yaml")
        
        # Create sample data
        print("Creating sample data...")
        processed_data, analytical_tables = create_sample_data()
        print(f"Created processed data with {len(processed_data)} records")
        print(f"Created {len(analytical_tables)} analytical tables")
        
        # Initialize export manager
        export_manager = ExportManager(config)
        
        # Export complete dataset
        print("\nExporting complete dataset...")
        export_summary = export_manager.export_complete_dataset(
            processed_data=processed_data,
            analytical_tables=analytical_tables,
            dataset_name="sample_ecommerce_data"
        )
        
        print(f"Export completed with version ID: {export_summary['version_id']}")
        print(f"Export timestamp: {export_summary['timestamp']}")
        
        # Display export paths
        print("\nExport paths:")
        for data_type, paths in export_summary['export_paths'].items():
            print(f"  {data_type}:")
            if isinstance(paths, dict):
                for format_type, path in paths.items():
                    print(f"    {format_type}: {path}")
            else:
                print(f"    {paths}")
        
        # Validate data integrity
        print("\nValidating data integrity...")
        validator = DataIntegrityValidator()
        
        # Validate processed data exports
        processed_paths = export_summary['export_paths']['processed_data']
        validation_results = validator.validate_exported_files(
            processed_paths, processed_data
        )
        
        print("Validation results:")
        for format_type, result in validation_results.items():
            status = "PASSED" if result.get('overall_valid', False) else "FAILED"
            print(f"  {format_type}: {status}")
            if not result.get('overall_valid', False):
                print(f"    Issues: {result.get('error', 'See detailed report')}")
        
        # List version history
        print("\nVersion history:")
        versions = export_manager.version_manager.list_versions("sample_ecommerce_data")
        for version in versions[:3]:  # Show last 3 versions
            print(f"  {version['version_id']} - {version['timestamp']}")
        
        print("\nData export example completed successfully!")
        
    except Exception as e:
        print(f"Error during export example: {e}")
        raise


if __name__ == "__main__":
    main()