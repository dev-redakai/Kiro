#!/usr/bin/env python3
"""
Test script for the dashboard application.

This script tests the dashboard components without running the full Streamlit app.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_data_provider():
    """Test the DataProvider class."""
    print("Testing DataProvider...")
    
    from dashboard.data_provider import DataProvider
    
    # Initialize data provider
    provider = DataProvider()
    
    # Test loading analytical tables
    tables = provider.load_analytical_tables()
    print(f"Loaded {len(tables)} tables:")
    for name, df in tables.items():
        print(f"  - {name}: {len(df)} rows, {len(df.columns)} columns")
    
    # Test dashboard metrics
    metrics = provider.get_dashboard_metrics()
    print(f"\nDashboard metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    
    # Test data freshness
    freshness = provider.get_data_freshness()
    print(f"\nData freshness:")
    print(f"  - Cache valid: {freshness['cache_valid']}")
    print(f"  - Available tables: {freshness['available_tables']}")
    print(f"  - Missing tables: {freshness['missing_tables']}")
    
    return True

def test_dashboard_components():
    """Test dashboard chart creation."""
    print("\nTesting Dashboard Components...")
    
    from dashboard.dashboard_app import DashboardApp
    
    # Initialize dashboard
    dashboard = DashboardApp()
    
    # Create sample data for testing
    monthly_data = pd.DataFrame({
        'month': ['2024-01', '2024-02', '2024-03'],
        'total_revenue': [100000, 120000, 110000],
        'unique_orders': [1000, 1200, 1100],
        'avg_order_value': [100, 100, 100],
        'avg_discount': [0.1, 0.12, 0.11]
    })
    
    products_data = pd.DataFrame({
        'product_name': ['Product A', 'Product B', 'Product C'],
        'total_revenue': [50000, 40000, 30000],
        'total_units': [500, 600, 400],
        'revenue_rank': [1, 2, 3],
        'units_rank': [2, 1, 3]
    })
    
    regional_data = pd.DataFrame({
        'region': ['North', 'South', 'East'],
        'total_revenue': [80000, 70000, 60000],
        'unique_orders': [800, 700, 600],
        'avg_order_value': [100, 100, 100],
        'market_share_pct': [40, 35, 25]
    })
    
    category_data = pd.DataFrame({
        'category': ['Electronics', 'Fashion', 'Home'],
        'avg_discount': [0.15, 0.20, 0.10],
        'discount_penetration_pct': [60, 70, 50],
        'total_revenue': [100000, 80000, 60000],
        'discount_effectiveness_score': [15000, 16000, 6000]
    })
    
    # Test chart creation
    try:
        revenue_chart = dashboard.create_revenue_trend_chart(monthly_data)
        print("✅ Revenue trend chart created successfully")
        
        products_chart = dashboard.create_top_products_chart(products_data)
        print("✅ Top products chart created successfully")
        
        regional_chart = dashboard.create_regional_sales_chart(regional_data)
        print("✅ Regional sales chart created successfully")
        
        heatmap_chart = dashboard.create_category_heatmap(category_data)
        print("✅ Category heatmap created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Dashboard Application")
    print("=" * 50)
    
    try:
        # Test data provider
        if not test_data_provider():
            print("❌ DataProvider tests failed")
            return False
        
        # Test dashboard components
        if not test_dashboard_components():
            print("❌ Dashboard component tests failed")
            return False
        
        print("\n✅ All tests passed successfully!")
        print("\n🚀 To run the dashboard, use: streamlit run run_dashboard.py")
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)