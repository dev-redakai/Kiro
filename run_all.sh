#!/bin/bash

# =============================================================================
# Scalable Data Pipeline - All-in-One Execution Script
# =============================================================================
# This script handles the complete pipeline workflow:
# 1. Environment setup and validation
# 2. Raw data generation
# 3. ETL pipeline execution
# 4. Dashboard data preparation
# 5. Dashboard launch
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_SAMPLE_SIZE=10000
DEFAULT_DASHBOARD_PORT=8501
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="logs/run_all.log"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${PURPLE}$1${NC}" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python environment
check_python_env() {
    print_status "Checking Python environment..."
    
    if ! command_exists python; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    print_success "Python version: $python_version"
    
    # Check if virtual environment is activated
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Virtual environment is active: $VIRTUAL_ENV"
    else
        print_warning "Virtual environment not detected. Attempting to activate..."
        if [[ -f "venv/bin/activate" ]]; then
            source venv/bin/activate
            print_success "Activated virtual environment"
        elif [[ -f "venv/Scripts/activate" ]]; then
            source venv/Scripts/activate
            print_success "Activated virtual environment (Windows)"
        else
            print_warning "No virtual environment found. Continuing with system Python..."
        fi
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Checking and installing dependencies..."
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        print_warning "requirements.txt not found. Creating basic requirements..."
        cat > requirements.txt << EOF
pandas>=1.3.0
numpy>=1.21.0
streamlit>=1.28.0
plotly>=5.0.0
pyarrow>=5.0.0
pytest>=6.0.0
pyyaml>=5.4.0
EOF
    fi
    
    # Install requirements
    print_status "Installing Python packages..."
    pip install -r requirements.txt --quiet
    print_success "Dependencies installed successfully"
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "data/raw"
        "data/processed"
        "data/output/csv"
        "data/output/parquet"
        "data/temp"
        "data/archive"
        "logs"
        "reports"
        "config"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "Directory structure created"
}

# Function to generate raw data
generate_raw_data() {
    local sample_size=${1:-$DEFAULT_SAMPLE_SIZE}
    
    print_header "=== STEP 1: GENERATING RAW DATA ==="
    print_status "Generating sample data with $sample_size records..."
    
    # Generate clean sample data
    python -c "
import sys
sys.path.insert(0, 'src')
try:
    from utils.test_data_generator import TestDataGenerator
    generator = TestDataGenerator()
    
    # Generate clean data
    print('Generating clean sample data...')
    clean_data = generator.generate_clean_sample(size=$sample_size)
    clean_data.to_csv('data/raw/clean_sample.csv', index=False)
    print(f'✓ Generated clean_sample.csv with {len(clean_data)} records')
    
    # Generate dirty data for testing
    print('Generating dirty sample data...')
    dirty_data = generator.generate_dirty_sample(size=int($sample_size * 0.3), dirty_ratio=0.3)
    dirty_data.to_csv('data/raw/dirty_sample.csv', index=False)
    print(f'✓ Generated dirty_sample.csv with {len(dirty_data)} records')
    
    # Generate anomaly data
    print('Generating anomaly test data...')
    anomaly_data = generator.generate_anomaly_sample(size=int($sample_size * 0.2), anomaly_ratio=0.05)
    anomaly_data.to_csv('data/raw/anomaly_sample.csv', index=False)
    print(f'✓ Generated anomaly_sample.csv with {len(anomaly_data)} records')
    
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
        'order_id': [f'ORD_{i:06d}' for i in range($sample_size)],
        'product_name': np.random.choice(['iPhone 13', 'Samsung TV', 'Nike Shoes', 'Dell Laptop'], $sample_size),
        'category': np.random.choice(['Electronics', 'Fashion', 'Home'], $sample_size),
        'quantity': np.random.randint(1, 10, $sample_size),
        'unit_price': np.random.uniform(50, 500, $sample_size),
        'discount_percent': np.random.uniform(0, 0.3, $sample_size),
        'region': np.random.choice(['North', 'South', 'East', 'West'], $sample_size),
        'sale_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range($sample_size)],
        'customer_email': [f'customer_{i}@example.com' for i in range($sample_size)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/clean_sample.csv', index=False)
    print(f'✓ Generated fallback clean_sample.csv with {len(df)} records')
"
    
    if [[ $? -eq 0 ]]; then
        print_success "Raw data generation completed"
    else
        print_error "Failed to generate raw data"
        exit 1
    fi
}

# Function to run ETL pipeline
run_etl_pipeline() {
    print_header "=== STEP 2: RUNNING ETL PIPELINE ==="
    print_status "Starting ETL pipeline processing..."
    
    # Check if input file exists
    if [[ ! -f "data/raw/clean_sample.csv" ]]; then
        print_error "Input file data/raw/clean_sample.csv not found"
        exit 1
    fi
    
    # Run the ETL pipeline
    print_status "Processing data through ETL pipeline..."
    
    # Set Python path
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    
    # Run pipeline with error handling
    if python -m src.pipeline.etl_main --input data/raw/clean_sample.csv --log-level INFO 2>&1 | tee -a "$LOG_FILE"; then
        print_success "ETL pipeline completed successfully"
    else
        print_warning "ETL pipeline encountered issues, but continuing..."
        # Don't exit here as we can still generate dashboard data
    fi
    
    # Check if output files were created
    if [[ -d "data/output/csv" ]] && [[ $(ls -A data/output/csv 2>/dev/null | wc -l) -gt 0 ]]; then
        print_success "Output files created in data/output/csv/"
        ls -la data/output/csv/ | tail -5 | tee -a "$LOG_FILE"
    else
        print_warning "No CSV output files found, will generate sample data for dashboard"
    fi
}

# Function to generate dashboard data
generate_dashboard_data() {
    print_header "=== STEP 3: GENERATING DASHBOARD DATA ==="
    print_status "Preparing analytical data for dashboard from ETL pipeline output..."
    
    # First try to generate from ETL pipeline output
    if python generate_dashboard_from_etl.py 2>&1 | tee -a "$LOG_FILE"; then
        print_success "Dashboard data generated from ETL pipeline output"
    else
        print_warning "ETL-based dashboard data generation failed, trying sample data generator..."
        
        # Fallback to sample data generator
        if python generate_dashboard_data.py 2>&1 | tee -a "$LOG_FILE"; then
            print_success "Dashboard data generated using sample data"
        else
            print_warning "Dashboard data generation failed, creating minimal sample data..."
        
        # Create minimal sample data as fallback
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
    fi
    
    # Verify dashboard data files exist
    dashboard_files=("monthly_sales_summary.csv" "top_products.csv" "region_wise_performance.csv" "category_discount_map.csv" "anomaly_records.csv")
    missing_files=()
    
    for file in "${dashboard_files[@]}"; do
        if [[ ! -f "data/output/$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -eq 0 ]]; then
        print_success "All dashboard data files are ready"
    else
        print_warning "Missing dashboard files: ${missing_files[*]}"
    fi
}

# Function to start dashboard
start_dashboard() {
    local port=${1:-$DEFAULT_DASHBOARD_PORT}
    
    print_header "=== STEP 4: STARTING DASHBOARD ==="
    print_status "Launching Streamlit dashboard on port $port..."
    
    # Check if Streamlit is installed
    if ! command_exists streamlit; then
        print_warning "Streamlit not found, installing..."
        pip install streamlit --quiet
    fi
    
    # Check if dashboard app exists
    if [[ ! -f "src/dashboard/dashboard_app.py" ]]; then
        print_error "Dashboard application not found at src/dashboard/dashboard_app.py"
        exit 1
    fi
    
    print_status "Starting dashboard server..."
    print_success "Dashboard will be available at: http://localhost:$port"
    print_status "Press Ctrl+C to stop the dashboard"
    
    # Start dashboard in the background and capture PID
    streamlit run src/dashboard/dashboard_app.py --server.port "$port" --server.headless false &
    DASHBOARD_PID=$!
    
    # Wait a moment for dashboard to start
    sleep 3
    
    # Check if dashboard is running
    if kill -0 $DASHBOARD_PID 2>/dev/null; then
        print_success "Dashboard started successfully (PID: $DASHBOARD_PID)"
        
        # Create a simple status check
        echo "Dashboard Status: Running" > logs/dashboard_status.txt
        echo "PID: $DASHBOARD_PID" >> logs/dashboard_status.txt
        echo "Port: $port" >> logs/dashboard_status.txt
        echo "URL: http://localhost:$port" >> logs/dashboard_status.txt
        
        # Wait for dashboard (this will keep the script running)
        wait $DASHBOARD_PID
    else
        print_error "Failed to start dashboard"
        exit 1
    fi
}

# Function to run health checks
run_health_checks() {
    print_header "=== HEALTH CHECKS ==="
    print_status "Running system health checks..."
    
    # Check disk space
    disk_usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        print_warning "Disk usage is high: ${disk_usage}%"
    else
        print_success "Disk usage: ${disk_usage}%"
    fi
    
    # Check memory
    if command_exists free; then
        memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
        print_success "Memory usage: ${memory_usage}%"
    fi
    
    # Check Python packages
    print_status "Checking critical Python packages..."
    critical_packages=("pandas" "numpy" "streamlit")
    
    for package in "${critical_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_success "✓ $package is available"
        else
            print_warning "✗ $package is not available"
        fi
    done
    
    # Check data files
    print_status "Checking data files..."
    if [[ -f "data/raw/clean_sample.csv" ]]; then
        record_count=$(wc -l < data/raw/clean_sample.csv)
        print_success "✓ Raw data file exists with $((record_count-1)) records"
    else
        print_warning "✗ Raw data file not found"
    fi
    
    if [[ -d "data/output" ]] && [[ $(ls -A data/output 2>/dev/null | wc -l) -gt 0 ]]; then
        output_count=$(ls -1 data/output/*.csv 2>/dev/null | wc -l)
        print_success "✓ Dashboard data files: $output_count files"
    else
        print_warning "✗ Dashboard data files not found"
    fi
}

# Function to display summary
display_summary() {
    print_header "=== EXECUTION SUMMARY ==="
    
    echo -e "${CYAN}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│                    PIPELINE SUMMARY                        │${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────────────────┤${NC}"
    
    # Check what was completed
    if [[ -f "data/raw/clean_sample.csv" ]]; then
        record_count=$(wc -l < data/raw/clean_sample.csv)
        echo -e "${CYAN}│${NC} ✅ Raw Data Generated: $((record_count-1)) records                    ${CYAN}│${NC}"
    else
        echo -e "${CYAN}│${NC} ❌ Raw Data Generation: Failed                          ${CYAN}│${NC}"
    fi
    
    if [[ -d "data/output/csv" ]] && [[ $(ls -A data/output/csv 2>/dev/null | wc -l) -gt 0 ]]; then
        output_count=$(ls -1 data/output/csv/*.csv 2>/dev/null | wc -l)
        echo -e "${CYAN}│${NC} ✅ ETL Pipeline: Completed ($output_count output files)              ${CYAN}│${NC}"
    else
        echo -e "${CYAN}│${NC} ⚠️  ETL Pipeline: Partial completion                    ${CYAN}│${NC}"
    fi
    
    if [[ -f "data/output/monthly_sales_summary.csv" ]]; then
        echo -e "${CYAN}│${NC} ✅ Dashboard Data: Ready                                ${CYAN}│${NC}"
    else
        echo -e "${CYAN}│${NC} ❌ Dashboard Data: Not available                        ${CYAN}│${NC}"
    fi
    
    if [[ -f "logs/dashboard_status.txt" ]]; then
        dashboard_url=$(grep "URL:" logs/dashboard_status.txt | cut -d' ' -f2)
        echo -e "${CYAN}│${NC} ✅ Dashboard: Running at $dashboard_url        ${CYAN}│${NC}"
    else
        echo -e "${CYAN}│${NC} ❌ Dashboard: Not started                               ${CYAN}│${NC}"
    fi
    
    echo -e "${CYAN}├─────────────────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│                      NEXT STEPS                            │${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│${NC} 1. Access dashboard: http://localhost:$DEFAULT_DASHBOARD_PORT                ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC} 2. View logs: tail -f logs/run_all.log                  ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC} 3. Check outputs: ls -la data/output/                   ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC} 4. Stop dashboard: Press Ctrl+C                         ${CYAN}│${NC}"
    echo -e "${CYAN}└─────────────────────────────────────────────────────────────┘${NC}"
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    if [[ -n $DASHBOARD_PID ]]; then
        kill $DASHBOARD_PID 2>/dev/null || true
    fi
    echo "Dashboard Status: Stopped" > logs/dashboard_status.txt
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --size SIZE        Number of sample records to generate (default: $DEFAULT_SAMPLE_SIZE)"
    echo "  -p, --port PORT        Dashboard port (default: $DEFAULT_DASHBOARD_PORT)"
    echo "  -h, --help             Show this help message"
    echo "  --skip-etl             Skip ETL pipeline execution"
    echo "  --skip-dashboard       Skip dashboard startup"
    echo "  --health-check-only    Run health checks only"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run with default settings"
    echo "  $0 -s 50000           # Generate 50,000 sample records"
    echo "  $0 -p 8502            # Use port 8502 for dashboard"
    echo "  $0 --skip-etl         # Skip ETL pipeline, only prepare dashboard"
    echo ""
}

# Main execution function
main() {
    local sample_size=$DEFAULT_SAMPLE_SIZE
    local dashboard_port=$DEFAULT_DASHBOARD_PORT
    local skip_etl=false
    local skip_dashboard=false
    local health_check_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--size)
                sample_size="$2"
                shift 2
                ;;
            -p|--port)
                dashboard_port="$2"
                shift 2
                ;;
            --skip-etl)
                skip_etl=true
                shift
                ;;
            --skip-dashboard)
                skip_dashboard=true
                shift
                ;;
            --health-check-only)
                health_check_only=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set up signal handlers
    trap cleanup EXIT INT TERM
    
    # Start execution
    print_header "🚀 SCALABLE DATA PIPELINE - ALL-IN-ONE EXECUTION"
    print_status "Starting complete pipeline workflow..."
    print_status "Log file: $LOG_FILE"
    echo "Execution started at $(date)" >> "$LOG_FILE"
    
    # Health check only mode
    if [[ $health_check_only == true ]]; then
        run_health_checks
        exit 0
    fi
    
    # Step 0: Environment setup
    check_python_env
    install_dependencies
    create_directories
    
    # Step 1: Generate raw data
    generate_raw_data "$sample_size"
    
    # Step 2: Run ETL pipeline (optional)
    if [[ $skip_etl == false ]]; then
        run_etl_pipeline
    else
        print_warning "Skipping ETL pipeline execution"
    fi
    
    # Step 3: Generate dashboard data
    generate_dashboard_data
    
    # Step 4: Health checks
    run_health_checks
    
    # Step 5: Display summary
    display_summary
    
    # Step 6: Start dashboard (optional)
    if [[ $skip_dashboard == false ]]; then
        start_dashboard "$dashboard_port"
    else
        print_warning "Skipping dashboard startup"
        print_success "Pipeline execution completed successfully!"
    fi
}

# Execute main function with all arguments
main "$@"