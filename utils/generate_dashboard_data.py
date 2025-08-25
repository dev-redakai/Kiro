#!/usr/bin/env python3
"""
Generate sample analytical data for dashboard testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_analytical_data():
    """Generate sample analytical tables for dashboard testing."""
    
    # Set random seed for reproducible data
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path('data/output')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Monthly Sales Summary
    months = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
    monthly_sales = pd.DataFrame({
        'month': months.strftime('%Y-%m'),
        'total_revenue': np.random.uniform(50000, 200000, len(months)),
        'unique_orders': np.random.randint(500, 2000, len(months)),
        'avg_order_value': np.random.uniform(80, 150, len(months)),
        'avg_discount': np.random.uniform(0.05, 0.25, len(months))
    })
    monthly_sales.to_csv(output_dir / 'monthly_sales_summary.csv', index=False)
    print(f"Created monthly_sales_summary: {len(monthly_sales)} records")
    
    # 2. Top Products
    products = [
        'iPhone 13 Pro', 'Samsung Galaxy S21', 'Dell XPS 13', 'HP Pavilion',
        'Nike Air Max', 'Adidas Ultraboost', 'Sony WH-1000XM4', 'Apple AirPods',
        'LG OLED TV', 'Samsung QLED TV'
    ]
    
    top_products = pd.DataFrame({
        'product_name': products,
        'total_revenue': np.random.uniform(10000, 50000, len(products)),
        'total_units': np.random.randint(100, 1000, len(products)),
        'revenue_rank': range(1, len(products) + 1),
        'units_rank': np.random.permutation(range(1, len(products) + 1))
    })
    top_products = top_products.sort_values('total_revenue', ascending=False)
    top_products.to_csv(output_dir / 'top_products.csv', index=False)
    print(f"Created top_products: {len(top_products)} records")
    
    # 3. Regional Performance
    regions = ['North', 'South', 'East', 'West', 'Central']
    regional_performance = pd.DataFrame({
        'region': regions,
        'total_revenue': np.random.uniform(80000, 150000, len(regions)),
        'unique_orders': np.random.randint(800, 1500, len(regions)),
        'avg_order_value': np.random.uniform(90, 140, len(regions)),
        'market_share_pct': np.random.uniform(15, 25, len(regions))
    })
    regional_performance.to_csv(output_dir / 'region_wise_performance.csv', index=False)
    print(f"Created region_wise_performance: {len(regional_performance)} records")
    
    # 4. Category Discount Map
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Books']
    category_discounts = pd.DataFrame({
        'category': categories,
        'avg_discount': np.random.uniform(0.05, 0.30, len(categories)),
        'discount_penetration_pct': np.random.uniform(40, 80, len(categories)),
        'total_revenue': np.random.uniform(30000, 100000, len(categories)),
        'discount_effectiveness_score': np.random.uniform(0.6, 0.9, len(categories))
    })
    category_discounts.to_csv(output_dir / 'category_discount_map.csv', index=False)
    print(f"Created category_discount_map: {len(category_discounts)} records")
    
    # 5. Anomaly Records
    anomaly_records = pd.DataFrame({
        'order_id': [f'ORD_{i:06d}' for i in range(1001, 1021)],
        'product_name': np.random.choice(products, 20),
        'revenue': np.random.uniform(5000, 15000, 20),  # High revenue anomalies
        'quantity': np.random.randint(50, 200, 20),     # High quantity anomalies
        'anomaly_score': np.random.uniform(3.5, 8.0, 20),
        'anomaly_type': np.random.choice(['revenue_outlier', 'quantity_outlier', 'pattern_anomaly'], 20)
    })
    anomaly_records.to_csv(output_dir / 'anomaly_records.csv', index=False)
    print(f"Created anomaly_records: {len(anomaly_records)} records")
    
    print("\nAll analytical tables created successfully!")
    print(f"Dashboard data available in: {output_dir.absolute()}")
    
    return {
        'monthly_sales_summary': monthly_sales,
        'top_products': top_products,
        'region_wise_performance': regional_performance,
        'category_discount_map': category_discounts,
        'anomaly_records': anomaly_records
    }

if __name__ == '__main__':
    tables = generate_sample_analytical_data()
    
    # Display summary
    print("\nData Summary:")
    for table_name, table_data in tables.items():
        print(f"- {table_name}: {len(table_data)} records")
        if len(table_data) > 0:
            print(f"  Columns: {list(table_data.columns)}")