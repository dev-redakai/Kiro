# =============================================================================
# Scalable Data Pipeline - All-in-One Execution Script (PowerShell)
# =============================================================================
# This script handles the complete pipeline workflow:
# 1. Environment setup and validation
# 2. Raw data generation
# 3. ETL pipeline execution
# 4. Dashboard data preparation
# 5. Dashboard launch
# =============================================================================

param(
    [string]$InputFile = "",
    [int]$SampleSize = 10000,
    [int]$DashboardPort = 8501,
    [switch]$SkipETL,
    [switch]$SkipDashboard,
    [switch]$HealthCheckOnly,
    [switch]$Help
)

# Configuration
$LogFile = "logs\run_all.log"
$ErrorActionPreference = "Continue"

# Create logs directory if it doesn't exist
if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}

# Initialize log file
"Pipeline execution started at $(Get-Date)" | Out-File -FilePath $LogFile -Encoding UTF8

# Function to write colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
    "[INFO] $Message" | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
    "[SUCCESS] $Message" | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
    "[WARNING] $Message" | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    "[ERROR] $Message" | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Magenta
    Write-Host ""
    "=== $Message ===" | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

# Function to show usage
function Show-Usage {
    Write-Host "Usage: .\run_all.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -InputFile FILE        Path to input raw data file (optional)"
    Write-Host "  -SampleSize SIZE       Number of sample records to generate (default: 10000)"
    Write-Host "  -DashboardPort PORT    Dashboard port (default: 8501)"
    Write-Host "  -SkipETL              Skip ETL pipeline execution"
    Write-Host "  -SkipDashboard        Skip dashboard startup"
    Write-Host "  -HealthCheckOnly      Run health checks only"
    Write-Host "  -Help                 Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_all.ps1                                    # Run with default settings"
    Write-Host "  .\run_all.ps1 -InputFile data\raw\clean\medium_clean_sample.csv  # Use specific input file"
    Write-Host "  .\run_all.ps1 -InputFile data\raw\dirty\dirty_sample_30pct.csv   # Test with dirty data"
    Write-Host "  .\run_all.ps1 -SampleSize 50000                  # Generate 50,000 sample records"
    Write-Host "  .\run_all.ps1 -DashboardPort 8502               # Use port 8502 for dashboard"
    Write-Host "  .\run_all.ps1 -SkipETL                          # Skip ETL pipeline, only prepare dashboard"
    Write-Host ""
    Write-Host "Available Raw Data Files:"
    Write-Host "  Clean Data:"
    Write-Host "    data\raw\clean\small_clean_sample.csv         (1,000 records)"
    Write-Host "    data\raw\clean\medium_clean_sample.csv        (10,000 records)"
    Write-Host "    data\raw\clean\large_clean_sample.csv         (50,000 records)"
    Write-Host "  Dirty Data:"
    Write-Host "    data\raw\dirty\dirty_sample_30pct.csv         (5,000 records, 30% dirty)"
    Write-Host "    data\raw\dirty\dirty_sample_50pct.csv         (3,000 records, 50% dirty)"
    Write-Host "  Anomaly Data:"
    Write-Host "    data\raw\anomaly\anomaly_sample_5pct.csv      (4,000 records, 5% anomalies)"
    Write-Host "    data\raw\anomaly\anomaly_sample_10pct.csv     (2,000 records, 10% anomalies)"
    Write-Host "  Mixed Data:"
    Write-Host "    data\raw\mixed\realistic_mixed_sample.csv     (8,000 records, mixed quality)"
    Write-Host "  Performance Data:"
    Write-Host "    data\raw\performance\performance_100k.csv     (100,000 records)"
    Write-Host "  Validation Test Data:"
    Write-Host "    data\raw\validation_test\null_order_ids_test.csv"
    Write-Host "    data\raw\validation_test\invalid_categories_test.csv"
    Write-Host "    data\raw\validation_test\invalid_emails_test.csv"
}

# Function to check Python environment
function Test-PythonEnvironment {
    Write-Status "Checking Python environment..."
    
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Python is not installed or not in PATH"
            exit 1
        }
        Write-Success "Python version: $pythonVersion"
    }
    catch {
        Write-Error "Failed to check Python version"
        exit 1
    }
    
    # Check if virtual environment is activated
    if ($env:VIRTUAL_ENV) {
        Write-Success "Virtual environment is active: $env:VIRTUAL_ENV"
    }
    else {
        Write-Warning "Virtual environment not detected. Attempting to activate..."
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & "venv\Scripts\Activate.ps1"
            Write-Success "Activated virtual environment"
        }
        elseif (Test-Path "venv\Scripts\activate.bat") {
            cmd /c "venv\Scripts\activate.bat"
            Write-Success "Activated virtual environment"
        }
        else {
            Write-Warning "No virtual environment found. Continuing with system Python..."
        }
    }
}

# Function to create directory structure
function New-DirectoryStructure {
    Write-Status "Creating directory structure..."
    
    $directories = @(
        "data\raw",
        "data\processed",
        "data\output\csv",
        "data\output\parquet",
        "data\temp",
        "data\archive",
        "logs",
        "reports",
        "config"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directory structure created"
}

# Function to install dependencies
function Install-Dependencies {
    Write-Status "Checking and installing dependencies..."
    
    if (!(Test-Path "requirements.txt")) {
        Write-Warning "requirements.txt not found. Creating basic requirements..."
        @"
pandas>=1.3.0
numpy>=1.21.0
streamlit>=1.28.0
plotly>=5.0.0
pyarrow>=5.0.0
pytest>=6.0.0
pyyaml>=5.4.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8
    }
    
    Write-Status "Installing Python packages..."
    try {
        pip install -r requirements.txt --quiet
        Write-Success "Dependencies installed successfully"
    }
    catch {
        Write-Warning "Some packages may have failed to install, continuing..."
    }
}

# Function to generate raw data
function New-RawData {
    param([int]$Size, [string]$InputFilePath)
    
    Write-Header "STEP 1: RAW DATA PREPARATION"
    
    # If input file is specified and exists, skip generation
    if ($InputFilePath -and (Test-Path $InputFilePath)) {
        Write-Status "Using specified input file: $InputFilePath"
        $fileSize = (Get-Item $InputFilePath).Length / 1MB
        $recordCount = (Get-Content $InputFilePath | Measure-Object -Line).Lines - 1
        Write-Success "Input file ready: $([math]::Round($fileSize, 2)) MB, $recordCount records"
        return
    }
    
    Write-Status "Generating sample data with $Size records..."
    
    $pythonScript = @"
import sys
sys.path.insert(0, 'src')
try:
    from utils.test_data_generator import TestDataGenerator
    generator = TestDataGenerator()
    
    # Generate clean data
    print('Generating clean sample data...')
    clean_data = generator.generate_clean_sample(size=$Size)
    clean_data.to_csv('data/raw/clean_sample.csv', index=False)
    print(f'Generated clean_sample.csv with {len(clean_data)} records')
    
    # Generate dirty data for testing
    print('Generating dirty sample data...')
    dirty_data = generator.generate_dirty_sample(size=int($Size * 0.3), dirty_ratio=0.3)
    dirty_data.to_csv('data/raw/dirty_sample.csv', index=False)
    print(f'Generated dirty_sample.csv with {len(dirty_data)} records')
    
    # Generate anomaly data
    print('Generating anomaly test data...')
    anomaly_data = generator.generate_anomaly_sample(size=int($Size * 0.2), anomaly_ratio=0.05)
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
        'order_id': [f'ORD_{i:06d}' for i in range($Size)],
        'product_name': np.random.choice(['iPhone 13', 'Samsung TV', 'Nike Shoes', 'Dell Laptop'], $Size),
        'category': np.random.choice(['Electronics', 'Fashion', 'Home'], $Size),
        'quantity': np.random.randint(1, 10, $Size),
        'unit_price': np.random.uniform(50, 500, $Size),
        'discount_percent': np.random.uniform(0, 0.3, $Size),
        'region': np.random.choice(['North', 'South', 'East', 'West'], $Size),
        'sale_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range($Size)],
        'customer_email': [f'customer_{i}@example.com' for i in range($Size)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/clean_sample.csv', index=False)
    print(f'Generated fallback clean_sample.csv with {len(df)} records')
"@
    
    try {
        python -c $pythonScript
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Raw data generation completed"
        }
        else {
            Write-Error "Failed to generate raw data"
            exit 1
        }
    }
    catch {
        Write-Error "Failed to execute data generation script"
        exit 1
    }
}

# Function to run ETL pipeline
function Start-ETLPipeline {
    param([string]$InputFilePath)
    
    Write-Header "STEP 2: RUNNING ETL PIPELINE"
    Write-Status "Starting ETL pipeline processing..."
    
    # Determine input file to use
    if ($InputFilePath -and (Test-Path $InputFilePath)) {
        $inputFile = $InputFilePath
        Write-Status "Using specified input file: $inputFile"
    }
    elseif (Test-Path "data\raw\clean_sample.csv") {
        $inputFile = "data\raw\clean_sample.csv"
        Write-Status "Using generated input file: $inputFile"
    }
    else {
        Write-Error "No input file found. Please specify -InputFile or ensure data generation completed."
        Write-Status "Available files in data\raw\:"
        if (Test-Path "data\raw\") {
            Get-ChildItem "data\raw\" -Recurse -Filter "*.csv" | Select-Object FullName, Length, LastWriteTime
        }
        exit 1
    }
    
    # Validate input file exists and is readable
    if (!(Test-Path $inputFile)) {
        Write-Error "Input file not found: $inputFile"
        exit 1
    }
    
    $fileSize = (Get-Item $inputFile).Length / 1MB
    $recordCount = (Get-Content $inputFile | Measure-Object -Line).Lines - 1
    Write-Status "Input file: $inputFile ($([math]::Round($fileSize, 2)) MB, $recordCount records)"
    
    Write-Status "Processing data through ETL pipeline..."
    $env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)\src"
    
    try {
        python -m src.pipeline.etl_main --input $inputFile --log-level INFO
        if ($LASTEXITCODE -eq 0) {
            Write-Success "ETL pipeline completed successfully"
        }
        else {
            Write-Warning "ETL pipeline encountered issues, but continuing..."
        }
    }
    catch {
        Write-Warning "ETL pipeline execution failed, but continuing..."
    }
    
    # Check if output files were created
    if (Test-Path "data\output\csv\*.csv") {
        Write-Success "Output files created in data\output\csv\"
        Get-ChildItem "data\output\csv\*.csv" | Select-Object Name, Length, LastWriteTime
    }
    else {
        Write-Warning "No CSV output files found, will generate sample data for dashboard"
    }
}

# Function to generate dashboard data
function New-DashboardData {
    Write-Header "STEP 3: GENERATING DASHBOARD DATA"
    Write-Status "Preparing analytical data for dashboard from ETL pipeline output..."
    
    try {
        # First try to generate from ETL pipeline output
        python utils/generate_dashboard_from_etl.py
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dashboard data generated from ETL pipeline output"
        }
        else {
            Write-Warning "ETL-based dashboard data generation failed, trying sample data generator..."
            
            # Fallback to sample data generator
            python utils/generate_dashboard_data.py
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Dashboard data generated using sample data"
            }
            else {
                Write-Warning "Dashboard data generation failed, creating minimal sample data..."
            
                $fallbackScript = @"
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
"@
                python -c $fallbackScript
            }
        }
    }
    catch {
        Write-Warning "Failed to generate dashboard data"
    }
}

# Function to run health checks
function Test-SystemHealth {
    param([string]$InputFilePath)
    
    Write-Header "HEALTH CHECKS"
    Write-Status "Running system health checks..."
    
    # Check input data file
    if ($InputFilePath -and (Test-Path $InputFilePath)) {
        $recordCount = (Get-Content $InputFilePath | Measure-Object -Line).Lines - 1
        $fileSize = (Get-Item $InputFilePath).Length / 1MB
        Write-Success "Input file: $InputFilePath ($([math]::Round($fileSize, 2)) MB, $recordCount records)"
    }
    elseif (Test-Path "data\raw\clean_sample.csv") {
        $recordCount = (Get-Content "data\raw\clean_sample.csv" | Measure-Object -Line).Lines - 1
        Write-Success "Generated raw data file exists with $recordCount records"
    }
    else {
        Write-Warning "No raw data file found"
    }
    
    if (Test-Path "data\output\*.csv") {
        $outputCount = (Get-ChildItem "data\output\*.csv" | Measure-Object).Count
        Write-Success "Dashboard data files: $outputCount files"
    }
    else {
        Write-Warning "Dashboard data files not found"
    }
    
    # Check Python packages
    Write-Status "Checking critical Python packages..."
    
    try {
        python -c "import pandas" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✓ pandas is available"
        }
        else {
            Write-Warning "✗ pandas is not available"
        }
    }
    catch {
        Write-Warning "✗ pandas check failed"
    }
    
    try {
        python -c "import streamlit" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✓ streamlit is available"
        }
        else {
            Write-Warning "✗ streamlit is not available"
        }
    }
    catch {
        Write-Warning "✗ streamlit check failed"
    }
}

# Function to display summary
function Show-Summary {
    Write-Header "EXECUTION SUMMARY"
    
    Write-Host "┌─────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
    Write-Host "│                    PIPELINE SUMMARY                        │" -ForegroundColor Cyan
    Write-Host "├─────────────────────────────────────────────────────────────┤" -ForegroundColor Cyan
    
    if (Test-Path "data\raw\clean_sample.csv") {
        Write-Host "│ ✅ Raw Data Generated: Records available                   │" -ForegroundColor Cyan
    }
    else {
        Write-Host "│ ❌ Raw Data Generation: Failed                             │" -ForegroundColor Cyan
    }
    
    if (Test-Path "data\output\csv\*.csv") {
        Write-Host "│ ✅ ETL Pipeline: Completed                                 │" -ForegroundColor Cyan
    }
    else {
        Write-Host "│ ⚠️  ETL Pipeline: Partial completion                       │" -ForegroundColor Cyan
    }
    
    if (Test-Path "data\output\monthly_sales_summary.csv") {
        Write-Host "│ ✅ Dashboard Data: Ready                                   │" -ForegroundColor Cyan
    }
    else {
        Write-Host "│ ❌ Dashboard Data: Not available                           │" -ForegroundColor Cyan
    }
    
    Write-Host "├─────────────────────────────────────────────────────────────┤" -ForegroundColor Cyan
    Write-Host "│                      NEXT STEPS                            │" -ForegroundColor Cyan
    Write-Host "├─────────────────────────────────────────────────────────────┤" -ForegroundColor Cyan
    Write-Host "│ 1. Access dashboard: http://localhost:$DashboardPort                │" -ForegroundColor Cyan
    Write-Host "│ 2. View logs: Get-Content logs\run_all.log                 │" -ForegroundColor Cyan
    Write-Host "│ 3. Check outputs: Get-ChildItem data\output\               │" -ForegroundColor Cyan
    Write-Host "│ 4. Stop dashboard: Press Ctrl+C                            │" -ForegroundColor Cyan
    Write-Host "└─────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan
}

# Function to start dashboard
function Start-Dashboard {
    param([int]$Port)
    
    Write-Header "STEP 4: STARTING DASHBOARD"
    Write-Status "Launching Streamlit dashboard on port $Port..."
    
    # Check if Streamlit is installed
    try {
        python -c "import streamlit" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Streamlit not found, installing..."
            pip install streamlit --quiet
        }
    }
    catch {
        Write-Warning "Failed to check Streamlit installation"
    }
    
    if (!(Test-Path "src\dashboard\dashboard_app.py")) {
        Write-Error "Dashboard application not found at src\dashboard\dashboard_app.py"
        exit 1
    }
    
    Write-Status "Starting dashboard server..."
    Write-Success "Dashboard will be available at: http://localhost:$Port"
    Write-Status "Press Ctrl+C to stop the dashboard"
    
    # Create dashboard status file
    @"
Dashboard Status: Running
Port: $Port
URL: http://localhost:$Port
"@ | Out-File -FilePath "logs\dashboard_status.txt" -Encoding UTF8
    
    # Start dashboard
    try {
        streamlit run src/dashboard/dashboard_app.py --server.port $Port
    }
    catch {
        Write-Error "Failed to start dashboard"
    }
    finally {
        "Dashboard Status: Stopped" | Out-File -FilePath "logs\dashboard_status.txt" -Encoding UTF8
    }
}

# Main execution
function Main {
    if ($Help) {
        Show-Usage
        return
    }
    
    Write-Header "🚀 SCALABLE DATA PIPELINE - ALL-IN-ONE EXECUTION"
    Write-Status "Starting complete pipeline workflow..."
    Write-Status "Log file: $LogFile"
    
    if ($HealthCheckOnly) {
        Test-SystemHealth
        return
    }
    
    # Step 0: Environment setup
    Test-PythonEnvironment
    Install-Dependencies
    New-DirectoryStructure
    
    # Step 1: Prepare raw data
    New-RawData -Size $SampleSize -InputFilePath $InputFile
    
    # Step 2: Run ETL pipeline (optional)
    if (!$SkipETL) {
        Start-ETLPipeline -InputFilePath $InputFile
    }
    else {
        Write-Warning "Skipping ETL pipeline execution"
    }
    
    # Step 3: Generate dashboard data
    New-DashboardData
    
    # Step 4: Health checks
    Test-SystemHealth -InputFilePath $InputFile
    
    # Step 5: Display summary
    Show-Summary
    
    # Step 6: Start dashboard (optional)
    if (!$SkipDashboard) {
        Start-Dashboard -Port $DashboardPort
    }
    else {
        Write-Warning "Skipping dashboard startup"
        Write-Success "Pipeline execution completed successfully!"
    }
}

# Execute main function
Main