@echo off
REM =============================================================================
REM Scalable Data Pipeline - All-in-One Execution Script (Windows)
REM =============================================================================
REM This script handles the complete pipeline workflow:
REM 1. Environment setup and validation
REM 2. Raw data generation
REM 3. ETL pipeline execution
REM 4. Dashboard data preparation
REM 5. Dashboard launch
REM =============================================================================

setlocal enabledelayedexpansion

REM Configuration
set DEFAULT_SAMPLE_SIZE=10000
set DEFAULT_DASHBOARD_PORT=8501
set LOG_FILE=logs\run_all.log

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Initialize log file
echo Pipeline execution started at %date% %time% > "%LOG_FILE%"

echo.
echo ===============================================================================
echo                    SCALABLE DATA PIPELINE - ALL-IN-ONE EXECUTION
echo ===============================================================================
echo.

REM Function to print status messages
:print_status
echo [INFO] %~1
echo [INFO] %~1 >> "%LOG_FILE%"
goto :eof

:print_success
echo [SUCCESS] %~1
echo [SUCCESS] %~1 >> "%LOG_FILE%"
goto :eof

:print_warning
echo [WARNING] %~1
echo [WARNING] %~1 >> "%LOG_FILE%"
goto :eof

:print_error
echo [ERROR] %~1
echo [ERROR] %~1 >> "%LOG_FILE%"
goto :eof

:print_header
echo.
echo === %~1 ===
echo === %~1 === >> "%LOG_FILE%"
echo.
goto :eof

REM Check Python environment
call :print_header "CHECKING ENVIRONMENT"
call :print_status "Checking Python environment..."

python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is not installed or not in PATH"
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
call :print_success "Python version: %PYTHON_VERSION%"

REM Check if virtual environment is activated
if defined VIRTUAL_ENV (
    call :print_success "Virtual environment is active: %VIRTUAL_ENV%"
) else (
    call :print_warning "Virtual environment not detected. Attempting to activate..."
    if exist "venv\Scripts\activate.bat" (
        call venv\Scripts\activate.bat
        call :print_success "Activated virtual environment"
    ) else (
        call :print_warning "No virtual environment found. Continuing with system Python..."
    )
)

REM Create directory structure
call :print_status "Creating directory structure..."
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\output\csv mkdir data\output\csv
if not exist data\output\parquet mkdir data\output\parquet
if not exist data\temp mkdir data\temp
if not exist data\archive mkdir data\archive
if not exist logs mkdir logs
if not exist reports mkdir reports
if not exist config mkdir config
call :print_success "Directory structure created"

REM Install dependencies
call :print_status "Checking and installing dependencies..."
if not exist requirements.txt (
    call :print_warning "requirements.txt not found. Creating basic requirements..."
    (
        echo pandas^>=1.3.0
        echo numpy^>=1.21.0
        echo streamlit^>=1.28.0
        echo plotly^>=5.0.0
        echo pyarrow^>=5.0.0
        echo pytest^>=6.0.0
        echo pyyaml^>=5.4.0
    ) > requirements.txt
)

call :print_status "Installing Python packages..."
pip install -r requirements.txt --quiet
if errorlevel 1 (
    call :print_warning "Some packages may have failed to install, continuing..."
) else (
    call :print_success "Dependencies installed successfully"
)

REM Step 1: Generate raw data
call :print_header "STEP 1: GENERATING RAW DATA"
call :print_status "Generating sample data with %DEFAULT_SAMPLE_SIZE% records..."

python -c "
import sys
sys.path.insert(0, 'src')
try:
    from utils.test_data_generator import TestDataGenerator
    generator = TestDataGenerator()
    
    # Generate clean data
    print('Generating clean sample data...')
    clean_data = generator.generate_clean_sample(size=%DEFAULT_SAMPLE_SIZE%)
    clean_data.to_csv('data/raw/clean_sample.csv', index=False)
    print(f'Generated clean_sample.csv with {len(clean_data)} records')
    
    # Generate dirty data for testing
    print('Generating dirty sample data...')
    dirty_data = generator.generate_dirty_sample(size=int(%DEFAULT_SAMPLE_SIZE% * 0.3), dirty_ratio=0.3)
    dirty_data.to_csv('data/raw/dirty_sample.csv', index=False)
    print(f'Generated dirty_sample.csv with {len(dirty_data)} records')
    
    # Generate anomaly data
    print('Generating anomaly test data...')
    anomaly_data = generator.generate_anomaly_sample(size=int(%DEFAULT_SAMPLE_SIZE% * 0.2), anomaly_ratio=0.05)
    anomaly_data.to_csv('data/raw/anomaly_sample.csv', index=False)
    print(f'Generated anomaly_sample.csv with {len(anomaly_data)} records')
    
    print('Raw data generation completed successfully!')
    
except Exception as e:
    print(f'Error generating data: {e}')
    # Fallback: create simple sample data
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print('Using fallback data generation...')
    np.random.seed(42)
    
    data = {
        'order_id': [f'ORD_{i:06d}' for i in range(%DEFAULT_SAMPLE_SIZE%)],
        'product_name': np.random.choice(['iPhone 13', 'Samsung TV', 'Nike Shoes', 'Dell Laptop'], %DEFAULT_SAMPLE_SIZE%),
        'category': np.random.choice(['Electronics', 'Fashion', 'Home'], %DEFAULT_SAMPLE_SIZE%),
        'quantity': np.random.randint(1, 10, %DEFAULT_SAMPLE_SIZE%),
        'unit_price': np.random.uniform(50, 500, %DEFAULT_SAMPLE_SIZE%),
        'discount_percent': np.random.uniform(0, 0.3, %DEFAULT_SAMPLE_SIZE%),
        'region': np.random.choice(['North', 'South', 'East', 'West'], %DEFAULT_SAMPLE_SIZE%),
        'sale_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(%DEFAULT_SAMPLE_SIZE%)],
        'customer_email': [f'customer_{i}@example.com' for i in range(%DEFAULT_SAMPLE_SIZE%)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/clean_sample.csv', index=False)
    print(f'Generated fallback clean_sample.csv with {len(df)} records')
"

if errorlevel 1 (
    call :print_error "Failed to generate raw data"
    pause
    exit /b 1
) else (
    call :print_success "Raw data generation completed"
)

REM Step 2: Run ETL pipeline
call :print_header "STEP 2: RUNNING ETL PIPELINE"
call :print_status "Starting ETL pipeline processing..."

if not exist "data\raw\clean_sample.csv" (
    call :print_error "Input file data\raw\clean_sample.csv not found"
    pause
    exit /b 1
)

call :print_status "Processing data through ETL pipeline..."
set PYTHONPATH=%PYTHONPATH%;%CD%\src

python -m src.pipeline.etl_main --input data/raw/clean_sample.csv --log-level INFO
if errorlevel 1 (
    call :print_warning "ETL pipeline encountered issues, but continuing..."
) else (
    call :print_success "ETL pipeline completed successfully"
)

REM Check if output files were created
if exist "data\output\csv\*" (
    call :print_success "Output files created in data\output\csv\"
    dir data\output\csv\*.csv /b
) else (
    call :print_warning "No CSV output files found, will generate sample data for dashboard"
)

REM Step 3: Generate dashboard data
call :print_header "STEP 3: GENERATING DASHBOARD DATA"
call :print_status "Preparing analytical data for dashboard from ETL pipeline output..."

python generate_dashboard_from_etl.py
if errorlevel 1 (
    call :print_warning "ETL-based dashboard data generation failed, trying sample data generator..."
    
    python generate_dashboard_data.py
    if errorlevel 1 (
        call :print_warning "Dashboard data generation failed, creating minimal sample data..."
    
    python -c "
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path('data/output')
output_dir.mkdir(exist_ok=True)

# Minimal monthly sales data
months = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
monthly_data = pd.DataFrame({
    'month': months,
    'total_revenue': np.random.uniform(50000, 100000, len(months)),
    'unique_orders': np.random.randint(500, 1000, len(months)),
    'avg_order_value': np.random.uniform(80, 120, len(months)),
    'avg_discount': np.random.uniform(0.1, 0.2, len(months))
})
monthly_data.to_csv(output_dir / 'monthly_sales_summary.csv', index=False)

# Minimal top products data
products = ['iPhone 13', 'Samsung TV', 'Nike Shoes', 'Dell Laptop', 'iPad Pro']
products_data = pd.DataFrame({
    'product_name': products,
    'total_revenue': np.random.uniform(10000, 30000, len(products)),
    'total_units': np.random.randint(100, 500, len(products)),
    'revenue_rank': range(1, len(products) + 1),
    'units_rank': range(1, len(products) + 1)
})
products_data.to_csv(output_dir / 'top_products.csv', index=False)

print('Minimal dashboard data created')
"
) else (
    call :print_success "Dashboard data generated successfully"
)

REM Step 4: Health checks
call :print_header "HEALTH CHECKS"
call :print_status "Running system health checks..."

REM Check data files
if exist "data\raw\clean_sample.csv" (
    for /f %%i in ('type "data\raw\clean_sample.csv" ^| find /c /v ""') do set RECORD_COUNT=%%i
    set /a RECORD_COUNT=!RECORD_COUNT!-1
    call :print_success "Raw data file exists with !RECORD_COUNT! records"
) else (
    call :print_warning "Raw data file not found"
)

if exist "data\output\*.csv" (
    for /f %%i in ('dir "data\output\*.csv" /b 2^>nul ^| find /c /v ""') do set OUTPUT_COUNT=%%i
    call :print_success "Dashboard data files: !OUTPUT_COUNT! files"
) else (
    call :print_warning "Dashboard data files not found"
)

REM Check Python packages
call :print_status "Checking critical Python packages..."
python -c "import pandas" 2>nul && (
    call :print_success "pandas is available"
) || (
    call :print_warning "pandas is not available"
)

python -c "import streamlit" 2>nul && (
    call :print_success "streamlit is available"
) || (
    call :print_warning "streamlit is not available"
)

REM Step 5: Display summary
call :print_header "EXECUTION SUMMARY"
echo.
echo ┌─────────────────────────────────────────────────────────────┐
echo │                    PIPELINE SUMMARY                        │
echo ├─────────────────────────────────────────────────────────────┤

if exist "data\raw\clean_sample.csv" (
    echo │ ✅ Raw Data Generated: Records available                   │
) else (
    echo │ ❌ Raw Data Generation: Failed                             │
)

if exist "data\output\csv\*" (
    echo │ ✅ ETL Pipeline: Completed                                 │
) else (
    echo │ ⚠️  ETL Pipeline: Partial completion                       │
)

if exist "data\output\monthly_sales_summary.csv" (
    echo │ ✅ Dashboard Data: Ready                                   │
) else (
    echo │ ❌ Dashboard Data: Not available                           │
)

echo ├─────────────────────────────────────────────────────────────┤
echo │                      NEXT STEPS                            │
echo ├─────────────────────────────────────────────────────────────┤
echo │ 1. Start dashboard: streamlit run src/dashboard/dashboard_app.py │
echo │ 2. View logs: type logs\run_all.log                       │
echo │ 3. Check outputs: dir data\output\                        │
echo │ 4. Access dashboard: http://localhost:%DEFAULT_DASHBOARD_PORT%                │
echo └─────────────────────────────────────────────────────────────┘
echo.

REM Step 6: Start dashboard
call :print_header "STEP 4: STARTING DASHBOARD"
call :print_status "Launching Streamlit dashboard on port %DEFAULT_DASHBOARD_PORT%..."

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    call :print_warning "Streamlit not found, installing..."
    pip install streamlit --quiet
)

if not exist "src\dashboard\dashboard_app.py" (
    call :print_error "Dashboard application not found at src\dashboard\dashboard_app.py"
    pause
    exit /b 1
)

call :print_status "Starting dashboard server..."
call :print_success "Dashboard will be available at: http://localhost:%DEFAULT_DASHBOARD_PORT%"
call :print_status "Press Ctrl+C to stop the dashboard"

REM Create dashboard status file
echo Dashboard Status: Running > logs\dashboard_status.txt
echo Port: %DEFAULT_DASHBOARD_PORT% >> logs\dashboard_status.txt
echo URL: http://localhost:%DEFAULT_DASHBOARD_PORT% >> logs\dashboard_status.txt

REM Start dashboard
streamlit run src/dashboard/dashboard_app.py --server.port %DEFAULT_DASHBOARD_PORT%

REM Cleanup on exit
echo Dashboard Status: Stopped > logs\dashboard_status.txt
call :print_success "Pipeline execution completed!"

pause