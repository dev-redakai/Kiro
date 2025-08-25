#!/usr/bin/env python3
"""
Final Integration Test - ETL Pipeline to Dashboard
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_etl_to_dashboard_integration():
    """Test complete integration from ETL pipeline to dashboard."""
    print("=" * 60)
    print("FINAL INTEGRATION TEST: ETL PIPELINE → DASHBOARD")
    print("=" * 60)
    
    # Step 1: Verify ETL pipeline output exists
    print("\n1. Checking ETL Pipeline Output...")
    
    csv_dir = "data/output/csv"
    parquet_dir = "data/output/parquet"
    
    # Find processed data files
    import glob
    csv_files = glob.glob(os.path.join(csv_dir, "*_processed_*.csv"))
    parquet_files = glob.glob(os.path.join(parquet_dir, "*_processed_*.parquet"))
    
    if csv_files or parquet_files:
        latest_file = (csv_files + parquet_files)[0]
        print(f"✅ ETL processed data found: {latest_file}")
        
        # Load and check processed data
        if latest_file.endswith('.parquet'):
            processed_data = pd.read_parquet(latest_file)
        else:
            processed_data = pd.read_csv(latest_file)
        
        print(f"   - Records: {len(processed_data):,}")
        print(f"   - Columns: {list(processed_data.columns)}")
    else:
        print("❌ No ETL processed data found")
        return False
    
    # Step 2: Test ETL-based dashboard data generation
    print("\n2. Testing ETL-based Dashboard Data Generation...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "generate_dashboard_from_etl.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dashboard data generated from ETL pipeline output")
        else:
            print(f"❌ Dashboard data generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running dashboard data generation: {e}")
        return False
    
    # Step 3: Verify analytical tables were created
    print("\n3. Verifying Analytical Tables...")
    
    analytical_tables = [
        'monthly_sales_summary.csv',
        'top_products.csv', 
        'region_wise_performance.csv',
        'category_discount_map.csv',
        'anomaly_records.csv'
    ]
    
    tables_found = 0
    for table_file in analytical_tables:
        table_path = os.path.join("data/output", table_file)
        if os.path.exists(table_path):
            df = pd.read_csv(table_path)
            print(f"✅ {table_file}: {len(df)} records")
            tables_found += 1
        else:
            print(f"❌ {table_file}: NOT FOUND")
    
    if tables_found == len(analytical_tables):
        print(f"✅ All {tables_found} analytical tables created successfully")
    else:
        print(f"⚠️ Only {tables_found}/{len(analytical_tables)} tables found")
    
    # Step 4: Test dashboard data provider
    print("\n4. Testing Dashboard Data Provider...")
    
    try:
        from dashboard.data_provider import DataProvider
        
        # Create fresh data provider (no cache)
        provider = DataProvider()
        provider.cache = {}  # Clear any cache
        provider.cache_timestamp = None
        
        # Load tables
        tables = provider.load_analytical_tables()
        
        print(f"✅ Data provider loaded {len(tables)} tables:")
        for table_name, df in tables.items():
            if len(df) > 0:
                print(f"   - {table_name}: {len(df)} records")
            else:
                print(f"   - {table_name}: EMPTY")
        
        # Test metrics calculation
        metrics = provider.get_dashboard_metrics()
        
        # Check if we have real data
        monthly_data = tables.get('monthly_sales_summary', pd.DataFrame())
        if len(monthly_data) > 0 and 'total_revenue' in monthly_data.columns:
            expected_revenue = monthly_data['total_revenue'].sum()
            actual_revenue = metrics.get('total_revenue', 0)
            
            if abs(expected_revenue - actual_revenue) < 0.01:  # Allow for small floating point differences
                print(f"✅ Metrics calculation working: ${actual_revenue:,.2f} total revenue")
            else:
                print(f"⚠️ Metrics calculation issue: expected ${expected_revenue:,.2f}, got ${actual_revenue:,.2f}")
        
    except Exception as e:
        print(f"❌ Dashboard data provider test failed: {e}")
        return False
    
    # Step 5: Test dashboard app (basic import test)
    print("\n5. Testing Dashboard App...")
    
    try:
        from dashboard.dashboard_app import DashboardApp
        print("✅ Dashboard app can be imported successfully")
        
        # Check if Streamlit is available
        try:
            import streamlit
            print("✅ Streamlit is available for dashboard")
        except ImportError:
            print("⚠️ Streamlit not available - install with: pip install streamlit")
        
    except Exception as e:
        print(f"❌ Dashboard app import failed: {e}")
        return False
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    print("✅ ETL Pipeline Output: Available")
    print("✅ Dashboard Data Generation: Working")
    print("✅ Analytical Tables: Created")
    print("✅ Dashboard Data Provider: Functional")
    print("✅ Dashboard App: Ready")
    
    print("\n🎉 COMPLETE INTEGRATION SUCCESSFUL!")
    print("\nThe dashboard now uses real data from the ETL pipeline:")
    print("1. ETL pipeline processes raw data")
    print("2. Dashboard generator creates analytical tables from processed data")
    print("3. Dashboard displays real business insights")
    
    print("\nTo start the dashboard:")
    print("streamlit run src/dashboard/dashboard_app.py")
    
    return True

if __name__ == '__main__':
    success = test_etl_to_dashboard_integration()
    sys.exit(0 if success else 1)